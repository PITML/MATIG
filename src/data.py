import torch
import json
import numpy as np
import random
from utils.data import random_rotate_z, normalize_pc, augment_pc
from torch.utils.data import Dataset, DataLoader
# import MinkowskiEngine as ME
import logging
import copy
from torch.utils.data.distributed import DistributedSampler


class Four(Dataset):
    def __init__(self, config, phase):
        self.phase = phase
        if phase == "train":
            self.split = json.load(open(config.dataset.train_split, "r"))
            if config.dataset.train_partial > 0:
                self.split = self.split[:config.dataset.train_partial]
        else:
            raise NotImplementedError("Phase %s not supported." % phase)
        self.phase = phase
        self.y_up = config.dataset.y_up
        self.random_z_rotate = config.dataset.random_z_rotate
        self.num_points = config.dataset.num_points
        self.use_color = config.dataset.use_color
        self.normalize = config.dataset.normalize
        self.rgb_random_drop_prob = config.dataset.rgb_random_drop_prob
        self.text_source = config.dataset.text_source
        self.augment = config.dataset.augment
        self.use_knn_negative_sample = config.dataset.use_knn_negative_sample
        self.use_text_filtering = config.dataset.use_text_filtering
        self.use_prompt_engineering = config.dataset.use_prompt_engineering
        self.text_embed_version = "prompt_avg" if self.use_prompt_engineering else "original"
        if self.use_knn_negative_sample:
            self.negative_sample_num = config.dataset.negative_sample_num
            self.knn = np.load(config.dataset.knn_path, allow_pickle=True).item()
            self.uid_to_index = {}
            for i, item in enumerate(self.split):
                self.uid_to_index[item['id']] = i
        if self.use_text_filtering:
            self.gpt4_filtering = json.load(open(config.dataset.gpt4_filtering_path, "r"))
        self.text_guidance_num = config.dataset.text_guidance_num
        self.image_guidance_num = config.dataset.image_guidance_num
        logging.info("Phase %s: %d samples" % (phase, len(self.split)))

    def get_objaverse(self, meta):
        uid = meta["id"]
        data = np.load(meta['data_path'], allow_pickle=True).item()
        n = data['xyz'].shape[0]
        idx = random.sample(range(n), self.num_points)
        xyz = data['xyz'][idx]
        rgb = data['rgb'][idx]
        
        if self.y_up:
            # swap y and z axis
            xyz[:, [1, 2]] = xyz[:, [2, 1]]
        if self.normalize:
            xyz = normalize_pc(xyz)
        if self.phase == "train" and self.augment:
            xyz = augment_pc(xyz)
        if self.phase == "train" and self.random_z_rotate:
            xyz = random_rotate_z(xyz)
        if self.use_color:
            if self.phase == "train" and np.random.rand() < self.rgb_random_drop_prob:
                rgb = np.ones_like(rgb) * 0.4
            features = np.concatenate([xyz, rgb], axis=1)
        else:
            features = xyz

        text_feat, text_feat1, text_feat2, text_feat3 = [], [], [], []
        texts, texts1, texts2, texts3 = [], [], [], []
        if 'text' in self.text_source:
            if not (self.use_text_filtering and self.gpt4_filtering[uid]["flag"] == "N"):
                texts1.append(data["text"][0])
                text_feat1.append(data["text_feat"][0][self.text_embed_version])

        if 'caption' in self.text_source:
            texts2.append(data["blip_caption"])
            text_feat2.append(data["blip_caption_feat"][self.text_embed_version])
            texts2.append(data["msft_caption"])
            text_feat2.append(data["msft_caption_feat"][self.text_embed_version])

        if 'retrieval_text' in self.text_source:
            if len(data["retrieval_text"]) > 0:
                for i in range(len(data['retrieval_text'])):
                    texts3.append(data["retrieval_text"][i])
                    text_feat3.append(data['retrieval_text_feat'][i]["original"])

        if self.text_guidance_num == 3:
            if len(texts1) == 1:
                texts.extend(texts1)
                text_feat.extend(text_feat1)
            if len(texts) == 1:
                if np.random.rand() < 0.5:
                    texts.append(texts2[0])
                    text_feat.append(text_feat2[0])
                else:
                    texts.append(texts2[1])
                    text_feat.append(text_feat2[1])
            else:
                texts.extend(texts2)
                text_feat.extend(text_feat2)

            if len(texts3) >= 1:
                idx = np.random.randint(len(texts3))
                texts.append(texts3[idx])
                text_feat.append(text_feat3[idx]) # no prompt engineering for retrieval text

        elif self.text_guidance_num > 3:
            while len(texts) < self.text_guidance_num:
                texts.extend(texts1)
                text_feat.extend(text_feat1)
                texts.extend(texts2)
                text_feat.extend(text_feat2)

                idx = list(range(len(texts3)))
                random.shuffle(idx)
                for i in range(len(texts3)):
                    texts.append(texts3[idx[i]])
                    text_feat.append(text_feat3[idx[i]])
            texts = texts[:self.text_guidance_num]
            text_feat = text_feat[:self.text_guidance_num]
        else:
            exit(0)

        img_feats = []
        if self.image_guidance_num == 3:
            if np.random.rand() < 0.5:
                img_feats.append(data['thumbnail_feat'])
                idx = random.sample(range(data['image_feat'].shape[0]), 2)
                img_feats.extend(data['image_feat'][idx])
            else:
                idx = random.sample(range(data['image_feat'].shape[0]), 3)
                img_feats.extend(data['image_feat'][idx])
        elif self.image_guidance_num > 3:
            while len(img_feats) < self.image_guidance_num:
                img_feats.append(data['thumbnail_feat'])
                idx = list(range(data['image_feat'].shape[0]))
                random.shuffle(idx)
                for i in idx:
                    img_feats.append(data['image_feat'][i])
            img_feats = img_feats[:self.image_guidance_num]
        else:
            exit(0)

        assert len(text_feat) == len(texts) == self.text_guidance_num and len(img_feats) == self.image_guidance_num, "len of text and imgs is valid"
        assert not np.isnan(xyz).any(), "xyz(point) are none!"

        return {
            "xyz": torch.from_numpy(xyz).type(torch.float32),
            "features": torch.from_numpy(features).type(torch.float32),
            "img_feat": torch.from_numpy(np.array(img_feats)).type(torch.float32).reshape(self.image_guidance_num, -1),
            "text_feat": torch.from_numpy(np.array(text_feat)).type(torch.float32).reshape(self.text_guidance_num, -1),
            "dataset": "Objaverse",
            "group": meta["group"],
            "name":  uid,
            "texts": texts,
            "has_text": text_feat is not None,
        }
    
    def get_others(self, meta):
        data = np.load(meta['data_path'], allow_pickle=True).item()
        n = data['xyz'].shape[0]
        idx = random.sample(range(n), self.num_points)
        xyz = data['xyz'][idx]
        rgb = data['rgb'][idx]

        if self.y_up:
            # swap y and z axis
            xyz[:, [1, 2]] = xyz[:, [2, 1]]
        if self.normalize:
            xyz = normalize_pc(xyz)
        if self.phase == "train" and self.augment:
            xyz = augment_pc(xyz)
        if self.phase == "train" and self.random_z_rotate:
            xyz = random_rotate_z(xyz)
        if self.use_color:
            if self.phase == "train" and np.random.rand() < self.rgb_random_drop_prob:
                rgb = np.ones_like(rgb) * 0.4
            features = np.concatenate([xyz, rgb], axis=1)
        else:
            features = xyz

        text_feat = []
        texts = []

        if self.text_guidance_num == 3:
            if 'text' in self.text_source:
                idx = np.random.randint(len(data["text"]))
                texts.append(data["text"][idx])
                text_feat.append(data["text_feat"][idx][self.text_embed_version])
            if 'caption' in self.text_source:
                if np.random.rand() < 0.5:
                    if len(data["blip_caption"]) > 0:
                        texts.append(data["blip_caption"])
                        text_feat.append(data["blip_caption_feat"][self.text_embed_version])
                else:
                    if len(data["msft_caption"]) > 0:
                        texts.append(data["msft_caption"])
                        text_feat.append(data["msft_caption_feat"][self.text_embed_version])

            if 'retrieval_text' in self.text_source:
                if len(data["retrieval_text"]) > 0:
                    idx = np.random.randint(len(data["retrieval_text"]))
                    texts.append(data["retrieval_text"][idx])
                    text_feat.append(data["retrieval_text_feat"][idx]["original"]) # no prompt engineering for retrieval text
        elif self.text_guidance_num > 3:
            while len(texts) < self.text_guidance_num:
                if 'text' in self.text_source:
                    idx = np.random.randint(len(data["text"]))
                    texts.append(data["text"][idx])
                    text_feat.append(data["text_feat"][idx][self.text_embed_version])
                if 'caption' in self.text_source:
                    texts.append(data["blip_caption"])
                    text_feat.append(data["blip_caption_feat"][self.text_embed_version])
                    texts.append(data["msft_caption"])
                    text_feat.append(data["msft_caption_feat"][self.text_embed_version])
                if 'retrieval_text' in self.text_source:
                    if len(data["retrieval_text"]) > 0:
                        idx = list(range(len(data["retrieval_text"])))
                        random.shuffle(idx)
                        for i in idx:
                            texts.append(data["retrieval_text"][i])
                            text_feat.append(data["retrieval_text_feat"][i]["original"])
            texts = texts[:self.text_guidance_num]
            text_feat = text_feat[:self.image_guidance_num]
        else:
            exit(0)

        img_feats = []
        while len(img_feats) < self.image_guidance_num:
            idx = list(range(data['image_feat'].shape[0]))
            random.shuffle(idx)
            for i in idx:
                img_feats.append(data['image_feat'][i])
        img_feats = img_feats[:self.image_guidance_num]

        assert len(text_feat) == len(texts) == self.text_guidance_num and len(img_feats) == self.image_guidance_num, 'len of text and imgs is valid'
        assert not np.isnan(xyz).any(), "xyz(point) are none!"

        return {
            "xyz": torch.from_numpy(xyz).type(torch.float32),
            "features": torch.from_numpy(features).type(torch.float32),
            "img_feat": torch.from_numpy(np.array(img_feats)).type(torch.float32).reshape(self.image_guidance_num, -1),
            "text_feat": torch.from_numpy(np.array(text_feat)).type(torch.float32).reshape(self.text_guidance_num, -1),
            "dataset": meta["dataset"],
            "group": meta["group"],
            "name":  meta["id"],
            "texts": texts,
            "has_text": text_feat is not None,
        }

    def __getitem__(self, index: int):
        if self.use_knn_negative_sample == False:
            if self.split[index]['dataset'] == "Objaverse":
                return self.get_objaverse(self.split[index])
            else:
                return self.get_others(self.split[index])
        else:
            data_list = []
            # random select a seed shape from split
            index = random.randint(0, len(self.split) - 1)
            uid = self.split[index]['id']
            # randomly pick (negative_sample_num - 1) neighbors from 31 nearest neighbors
            knn_idx = [0] + (np.random.choice(31, self.negative_sample_num - 1, replace=False) + 1).tolist() 
            for i in knn_idx:
                idx = self.uid_to_index[self.knn['name'][self.knn['index'][uid][i]]]
                if self.split[idx]['dataset'] == "Objaverse":
                    data_list.append(self.get_objaverse(self.split[idx]))
                else:
                    data_list.append(self.get_others(self.split[idx]))
            return data_list

    def __len__(self):
        if self.use_knn_negative_sample == False:
            return len(self.split)
        else:
            return len(self.split) // self.negative_sample_num
    
def minkowski_collate_fn(list_data):
    if isinstance(list_data[0], list):
        merged_list = []
        for data in list_data:
            merged_list += data
        list_data = merged_list
    return {
        "features": torch.cat([data["features"] for data in list_data], dim=0),
        "xyz_dense": torch.stack([data["xyz"] for data in list_data]).float(),
        "features_dense": torch.stack([data["features"] for data in list_data]),
        "img_feat": torch.stack([data["img_feat"] for data in list_data]),
        "text_feat": torch.stack([data["text_feat"] for data in list_data if data["text_feat"] is not None]),
        "dataset": [data["dataset"] for data in list_data],
        "group": [data["group"] for data in list_data],
        "name": [data["name"] for data in list_data],
        "texts": [data["texts"] for data in list_data],
        "has_text_idx": [i for i, data in enumerate(list_data) if data["text_feat"] is not None],
    }

def make(config, phase, rank, world_size):
    if config.dataset.name == "Four":
        dataset = Four(config, phase,)
        if phase == "train":
            batch_size = config.dataset.train_batch_size
            sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, drop_last=True)
        else:
            raise NotImplementedError("Phase %s not supported." % phase)
        data_loader = DataLoader(
            dataset,
            num_workers=config.dataset.num_workers,
            collate_fn=minkowski_collate_fn,
            batch_size=batch_size,
            pin_memory = True,
            drop_last=True,
            sampler=sampler
        )
    else:
        raise NotImplementedError("Dataset %s not supported." % config.dataset.name)
    return data_loader

class ModelNet40Test(Dataset):
    def __init__(self, config):
        self.split = json.load(open(config.modelnet40.test_split, "r"))
        self.pcs = np.load(config.modelnet40.test_pc, allow_pickle=True)
        self.num_points = config.modelnet40.num_points
        self.use_color = config.dataset.use_color
        self.y_up = config.modelnet40.y_up
        clip_feat = np.load(config.modelnet40.clip_feat_path, allow_pickle=True).item()
        self.categories = list(clip_feat.keys())
        self.clip_cat_feat = []
        self.category2idx = {}
        for i, category in enumerate(self.categories):
            self.category2idx[category] = i
            self.clip_cat_feat.append(clip_feat[category]["open_clip_text_feat"])
        self.clip_cat_feat = np.concatenate(self.clip_cat_feat, axis=0)

        logging.info("ModelNet40Test: %d samples" % len(self.split))
        logging.info("----clip feature shape: %s" % str(self.clip_cat_feat.shape))

    def __getitem__(self, index: int):
        pc = copy.deepcopy(self.pcs[index])
        n = pc['xyz'].shape[0]
        idx = random.sample(range(n), self.num_points)
        xyz = pc['xyz'][idx]
        rgb = pc['rgb'][idx]
        rgb = rgb / 255.0 # 100, scale to 0.4 to make it consistent with the training data
        if self.y_up:
            # swap y and z axis
            xyz[:, [1, 2]] = xyz[:, [2, 1]]
        
        xyz = normalize_pc(xyz)

        if self.use_color:
            features = np.concatenate([xyz, rgb], axis=1)
        else:
            features = xyz
        
        assert not np.isnan(xyz).any()
        
        return {
            "xyz": torch.from_numpy(xyz).type(torch.float32),
            "features": torch.from_numpy(features).type(torch.float32),
            "name": self.split[index]["name"],
            "category": self.category2idx[self.split[index]["category"]],
        }

    def __len__(self):
        return len(self.split)

def minkowski_modelnet40_collate_fn(list_data):
    return {
        "features": torch.cat([data["features"] for data in list_data], dim=0),
        "xyz_dense": torch.stack([data["xyz"] for data in list_data]).float(),
        "features_dense": torch.stack([data["features"] for data in list_data]),
        "name": [data["name"] for data in list_data],
        "category": torch.tensor([data["category"] for data in list_data], dtype = torch.int32),
    }

def make_modelnet40test(config):
    dataset = ModelNet40Test(config)
    data_loader = DataLoader(
        dataset, \
        num_workers=config.modelnet40.num_workers, \
        collate_fn=minkowski_modelnet40_collate_fn, \
        batch_size=config.modelnet40.test_batch_size, \
        pin_memory = True, \
        shuffle=False
    )
    return data_loader

class ObjaverseLVIS(Dataset):
    def __init__(self, config):
        self.split = json.load(open(config.objaverse_lvis.split, "r"))
        self.y_up = config.objaverse_lvis.y_up
        self.num_points = config.objaverse_lvis.num_points
        self.use_color = config.objaverse_lvis.use_color
        self.normalize = config.objaverse_lvis.normalize
        self.categories = sorted(np.unique([data['category'] for data in self.split]))
        self.category2idx = {self.categories[i]: i for i in range(len(self.categories))}
        self.clip_cat_feat = np.load(config.objaverse_lvis.clip_feat_path, allow_pickle=True)
        
        logging.info("ObjaverseLVIS: %d samples" % (len(self.split)))
        logging.info("----clip feature shape: %s" % str(self.clip_cat_feat.shape))

    def __getitem__(self, index: int):
        data = np.load(self.split[index]['data_path'], allow_pickle=True).item()
        n = data['xyz'].shape[0]
        idx = random.sample(range(n), self.num_points)
        xyz = data['xyz'][idx]
        rgb = data['rgb'][idx]

        if self.y_up:
            # swap y and z axis
            xyz[:, [1, 2]] = xyz[:, [2, 1]]
        if self.normalize:
            xyz = normalize_pc(xyz)
        if self.use_color:
            features = np.concatenate([xyz, rgb], axis=1)
        else:
            features = xyz
        
        assert not np.isnan(xyz).any()

        return {
            "xyz": torch.from_numpy(xyz).type(torch.float32),
            "features": torch.from_numpy(features).type(torch.float32),
            "group": self.split[index]['group'],
            "name":  self.split[index]['uid'],
            "category": self.category2idx[self.split[index]["category"]],
        }

    def __len__(self):
        return len(self.split)
    
def minkowski_objaverse_lvis_collate_fn(list_data):
    return {
        # "xyz": ME.utils.batched_coordinates([data["xyz"] for data in list_data], dtype=torch.float32),
        "features": torch.cat([data["features"] for data in list_data], dim=0),
        "xyz_dense": torch.stack([data["xyz"] for data in list_data]).float(),
        "features_dense": torch.stack([data["features"] for data in list_data]),
        "group": [data["group"] for data in list_data],
        "name": [data["name"] for data in list_data],
        "category": torch.tensor([data["category"] for data in list_data], dtype = torch.int32),
    }


def make_objaverse_lvis(config):
    return DataLoader(
        ObjaverseLVIS(config), \
        num_workers=config.objaverse_lvis.num_workers, \
        collate_fn=minkowski_objaverse_lvis_collate_fn, \
        batch_size=config.objaverse_lvis.batch_size, \
        pin_memory = True, \
        shuffle=False 
    )

class ScanObjectNNTest(Dataset):
    def __init__(self, config):
        self.data = np.load(config.scanobjectnn.data_path, allow_pickle=True).item()
        self.num_points = config.scanobjectnn.num_points
        self.use_color = config.dataset.use_color
        self.y_up = config.scanobjectnn.y_up
        clip_feat = np.load(config.scanobjectnn.clip_feat_path, allow_pickle=True).item()
        self.categories = ["bag", "bin", "box", "cabinet", "chair", "desk", "display", "door", "shelf", "table", "bed", "pillow", "sink", "sofa", "toilet"]
        self.clip_cat_feat = []
        self.category2idx = {}
        for i, category in enumerate(self.categories):
            self.category2idx[category] = i
            self.clip_cat_feat.append(clip_feat[category]["open_clip_text_feat"])
        self.clip_cat_feat = np.concatenate(self.clip_cat_feat, axis=0)
        logging.info("ScanObjectNNTest: %d samples" % self.__len__())
        logging.info("----clip feature shape: %s" % str(self.clip_cat_feat.shape))

    def __getitem__(self, index: int):
        xyz = copy.deepcopy(self.data['xyz'][index])
        if 'rgb' not in self.data:
            rgb = np.ones_like(xyz) * 0.4
        else:
            rgb = self.data['rgb'][index]
        label = self.data['label'][index]
        n = xyz.shape[0]
        if n != self.num_points:
            idx = np.random.choice(n, self.num_points) #random.sample(range(n), self.num_points)
            xyz = xyz[idx]
            rgb = rgb[idx]
        if self.y_up:
            # swap y and z axis
            xyz[:, [1, 2]] = xyz[:, [2, 1]]
        
        xyz = normalize_pc(xyz)
        if self.use_color:
            features = np.concatenate([xyz, rgb], axis=1)
        else:
            features = xyz
        
        assert not np.isnan(xyz).any()
        return {
            "xyz": torch.from_numpy(xyz).type(torch.float32),
            "features": torch.from_numpy(features).type(torch.float32),
            "name": str(index),
            "category": label,
        }

    def __len__(self):
        return len(self.data['xyz'])
    
def make_scanobjectnntest(config):
    dataset = ScanObjectNNTest(config)
    data_loader = DataLoader(
        dataset, \
        num_workers=config.scanobjectnn.num_workers, \
        collate_fn=minkowski_modelnet40_collate_fn, \
        batch_size=config.scanobjectnn.test_batch_size, \
        pin_memory = True, \
        shuffle=False
    )
    return data_loader
