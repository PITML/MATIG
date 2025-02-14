import torch
import numpy as np
import logging
import os
import torch.distributed.nn
import torch.nn.functional as F
from tqdm import tqdm
from utils.extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2

class Trainer(object):
    def __init__(self, rank, config, model, logit_scale, image_proj, text_proj, optimizer, scheduler, train_loader, \
                  modelnet40_loader, objaverse_lvis_loader=None, scanobjectnn_loader=None):
        self.rank = rank
        self.config = config
        self.model = model
        self.logit_scale = logit_scale
        self.image_proj = image_proj
        self.text_proj = text_proj
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.modelnet40_loader = modelnet40_loader
        self.objaverse_lvis_loader = objaverse_lvis_loader
        self.scanobjectnn_loader = scanobjectnn_loader
        self.epoch = 0
        self.step = 0
        self.best_img_contras_acc = 0
        self.best_text_contras_acc = 0
        self.best_modelnet40_overall_acc = 0
        self.best_modelnet40_class_acc = 0
        self.best_lvis_acc = 0
        if 'rebuild' in self.config.model.loss['type']:
            self.build_loss_func(self.config.model.loss['rebuild_loss'])

    def load_from_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['state_dict'])
        if self.config.training.use_text_proj:
            self.text_proj.load_state_dict(checkpoint['text_proj'])
        if self.config.training.use_image_proj:
            self.image_proj.load_state_dict(checkpoint['image_proj'])
        self.logit_scale.load_state_dict(checkpoint['logit_scale']) #module.logit_scale = checkpoint['logit_scale']
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if self.config.training.use_openclip_optimizer_scheduler == False:
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.epoch = checkpoint['epoch']
        self.step = checkpoint['step']
        self.best_img_contras_acc = checkpoint['best_img_contras_acc']
        self.best_text_contras_acc = checkpoint['best_text_contras_acc']
        self.best_modelnet40_overall_acc = checkpoint['best_modelnet40_overall_acc']
        self.best_modelnet40_class_acc = checkpoint['best_modelnet40_class_acc']
        self.best_lvis_acc = checkpoint['best_lvis_acc']

        logging.info("Loaded checkpoint from {}".format(path))
        logging.info("----Epoch: {0} Step: {1}".format(self.epoch, self.step))
        logging.info("----Best img contras acc: {}".format(self.best_img_contras_acc))
        logging.info("----Best text contras acc: {}".format(self.best_text_contras_acc))
        logging.info("----Best modelnet40 overall acc: {}".format(self.best_modelnet40_overall_acc))
        logging.info("----Best modelnet40 class acc: {}".format(self.best_modelnet40_class_acc))
        logging.info("----Best lvis acc: {}".format(self.best_lvis_acc))

    def contras_loss(self, feat1, feat2, logit_scale=1, mask = None):
        if self.config.ngpu > 1:
            feat1 = F.normalize(feat1, dim=1)
            feat2 = F.normalize(feat2, dim=1)
            all_feat1 = torch.cat(torch.distributed.nn.all_gather(feat1), dim=0)
            all_feat2 = torch.cat(torch.distributed.nn.all_gather(feat2), dim=0)
            logits = logit_scale * all_feat1 @ all_feat2.T
        else:
            logits = logit_scale * F.normalize(feat1, dim=1) @ F.normalize(feat2, dim=1).T
        if mask is not None:
            logits = logits * mask
        labels = torch.arange(logits.shape[0]).to(self.config.device)
        accuracy = (logits.argmax(dim=1) == labels).float().mean()
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
        return loss, accuracy

    def build_loss_func(self, loss_type):
        if loss_type == "cdl1":
            self.rebuild_loss_func = ChamferDistanceL1().cuda()
        elif loss_type =='cdl2':
            self.rebuild_loss_func = ChamferDistanceL2().cuda()
        else:
            raise NotImplementedError


    def train_one_epoch(self):
        self.model.train()
        if self.config.training.use_text_proj:
            self.text_proj.train()
        if self.config.training.use_image_proj:
            self.image_proj.train()

        text_contras_acc_list = []
        img_contras_acc_list = []
        if self.config.training.use_mask:
            k = self.config.dataset.negative_sample_num
            s = self.config.dataset.train_batch_size
            mask1 = np.eye(k * s).astype(np.bool)
            mask2 = np.kron(np.eye(s), np.ones((k, k))).astype(np.bool)
            mask_other = torch.from_numpy(np.logical_or(mask1, 1 - mask2)).bool().to(self.config.device)

        for _, data in enumerate(self.train_loader):
            self.step += 1
            self.optimizer.zero_grad()
            loss = 0
            if not self.config.model.get("use_dense", False):
                pred_feat = self.model(data['xyz'], data['features'], \
                                        device = self.config.device, \
                                        quantization_size = self.config.model.voxel_size)
            else:
                pred_feat, rebuild_points, gt_mask_points = self.model(data['xyz_dense'], data['features_dense'], data['img_feat'], data['text_feat'])
            logit_scale = self.logit_scale(None)
            idx = data['has_text_idx']

            text_feat = data['text_feat'].to(self.config.device)
            img_feat = data['img_feat'].to(self.config.device)

            if 'contrasive' in self.config.model.loss['type']:
                if self.config.training.use_mask:
                    img_text_sim = F.normalize(img_feat, dim=-1) @ F.normalize(text_feat, dim=-1).T
                    mask = torch.diagonal(img_text_sim).reshape(-1, 1) - img_text_sim > self.config.training.mask_threshold
                    mask = torch.logical_or(mask, mask_other).detach()
                else:
                    mask = None

                if len(idx) > 0:
                    if self.config.training.use_text_proj:
                        text_feat = self.text_proj(text_feat)
                    text_contras_loss_sum = 0
                    for i in range(text_feat.shape[1]):
                        text_feat_e = text_feat[:, i, :]
                        text_contras_loss, text_contras_acc = self.contras_loss(pred_feat[idx], text_feat_e, logit_scale=logit_scale, mask=mask)
                        text_contras_loss_sum += text_contras_loss * self.config.training.lambda_text_contras
                        loss += text_contras_loss * self.config.training.lambda_text_contras
                        text_contras_acc_list.append(text_contras_acc.item())

                if self.config.training.use_image_proj:
                    img_feat = self.image_proj(img_feat)
                img_contras_loss_sum = 0
                for i in range(img_feat.shape[1]):
                    img_feat_e = img_feat[:, i, :]
                    img_contras_loss, img_contras_acc = self.contras_loss(pred_feat, img_feat_e, logit_scale=logit_scale, mask=mask)
                    img_contras_loss_sum += img_contras_loss * self.config.training.lambda_img_contras
                    loss += img_contras_loss * self.config.training.lambda_img_contras
                    img_contras_acc_list.append(img_contras_acc.item())

            rebuild_loss = 0.0
            if 'rebuild' in self.config.model.loss['type']:
                rebuild_loss = self.rebuild_loss_func(rebuild_points, gt_mask_points)
                rebuild_loss = rebuild_loss * self.config.training.lambda_rebuild

                loss += rebuild_loss

            loss.backward()
            self.optimizer.step()
            if self.config.training.use_openclip_optimizer_scheduler:
                self.scheduler(self.step)
            else:
                self.scheduler.step()

            logging.info('epoch {0} step {1} '
                         'train/loss: {2}, '
                         'train/text_contras_loss_sum: {3}, '
                         'train/img_contras_loss_sum: {4}, '
                         'train/rebuild_loss: {5}, '
                         'train/text_contras_acc: {6}, '
                         'train/img_contras_acc: {7},'
                         'train/lr: {8}, '
                         'train/epoch: {9}, '
                         'train/step: {10}, '
                         'train/logit_scale: {11}, '
                         'train/has_text: {12}, '
                         'train/filtered_pair: {13}'
                         .format(self.epoch,
                                 _,
                                 loss.item(),
                                 text_contras_loss_sum.item() if len(idx) > 0 else 0,
                                 img_contras_loss_sum.item(),
                                 0.0 if isinstance(rebuild_loss, float) else rebuild_loss.item() ,
                                 np.mean(text_contras_acc_list), # text_contras_acc.item() if len(idx) > 0 else 0,
                                 np.mean(img_contras_acc_list), # img_contras_acc.item(),
                                 self.optimizer.param_groups[0]['lr'],
                                 self.epoch,
                                 self.step,
                                 logit_scale,
                                 len(idx),
                                 (mask == False).sum() if mask is not None else 0
                                )
                         )
        if self.rank == 0: 
            logging.info('Train: text_cotras_acc: {0} image_contras_acc: {1}'\
                    .format(np.mean(text_contras_acc_list) if len(text_contras_acc_list) > 0 else 0,
                            np.mean(img_contras_acc_list)))

    def save_model(self, name):
        torch.save({
                    "state_dict": self.model.state_dict(),
                    "logit_scale": self.logit_scale.state_dict(),#module.logit_scale,
                    "text_proj": self.text_proj.state_dict() if self.config.training.use_text_proj else None,
                    "image_proj": self.image_proj.state_dict() if self.config.training.use_image_proj else None,
                    "optimizer": self.optimizer.state_dict(),
                    "scheduler": self.scheduler.state_dict() if self.config.training.use_openclip_optimizer_scheduler == False else None,
                    "epoch": self.epoch,
                    "step": self.step,
                    "best_img_contras_acc": self.best_img_contras_acc,
                    "best_text_contras_acc": self.best_text_contras_acc,
                    "best_modelnet40_overall_acc": self.best_modelnet40_overall_acc,
                    "best_modelnet40_class_acc": self.best_modelnet40_class_acc,
                    "best_lvis_acc": self.best_lvis_acc,
                }, os.path.join(self.config.ckpt_dir, '{}.pt'.format(name)))

    def accuracy(self, output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.reshape(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res, correct
    
    def train(self):
        for epoch in range(self.epoch, self.config.training.max_epoch):
            self.epoch = epoch
            if self.rank == 0:
                logging.info("Epoch: {}".format(self.epoch))
            self.train_one_epoch()
            if self.rank == 0:
                self.save_model('latest')
                self.test_modelnet40()
                self.test_objaverse_lvis()
            if self.rank == 0 and self.epoch % self.config.training.save_freq == 0:
                self.save_model('epoch_{}'.format(self.epoch))
                        
    def test_modelnet40(self):
        self.model.eval()
        if self.config.training.use_text_proj:
            self.text_proj.eval()
        clip_text_feat = torch.from_numpy(self.modelnet40_loader.dataset.clip_cat_feat).to(self.config.device)
        if self.config.training.use_text_proj:
            clip_text_feat = self.text_proj(clip_text_feat)
        per_cat_correct = torch.zeros(40).to(self.config.device)
        per_cat_count = torch.zeros(40).to(self.config.device)
        category2idx = self.modelnet40_loader.dataset.category2idx
        idx2category = {v: k for k, v in category2idx.items()}
        
        logits_all = []
        labels_all = []
        with torch.no_grad():
            for data in self.modelnet40_loader:
                if not self.config.model.get("use_dense", False):
                    pred_feat = self.model(data['xyz'], data['features'], \
                                            device = self.config.device, \
                                            quantization_size = self.config.model.voxel_size)
                else:
                    pred_feat = self.model(data['xyz_dense'], data['features_dense'])
                logits = F.normalize(pred_feat, dim=1) @ F.normalize(clip_text_feat, dim=1).T
                labels = data['category'].to(self.config.device)
                logits_all.append(logits.detach())
                labels_all.append(labels)

                for i in range(40):
                    idx = (labels == i)
                    if idx.sum() > 0:
                        per_cat_correct[i] += (logits[idx].argmax(dim=1) == labels[idx]).float().sum()
                        per_cat_count[i] += idx.sum()
        topk_acc, correct = self.accuracy(torch.cat(logits_all), torch.cat(labels_all), topk=(1,3,5,))

        overall_acc = per_cat_correct.sum() / per_cat_count.sum()
        per_cat_acc = per_cat_correct / per_cat_count

        if overall_acc > self.best_modelnet40_overall_acc:
            self.best_modelnet40_overall_acc = overall_acc
            self.save_model('best_modelnet40_overall')
        if per_cat_acc.mean() > self.best_modelnet40_class_acc:
            self.best_modelnet40_class_acc = per_cat_acc.mean()
            self.save_model('best_modelnet40_class')

        logging.info('Test ModelNet40: overall acc: {0}({1}) class_acc: {2}({3})'.format(overall_acc, self.best_modelnet40_overall_acc, per_cat_acc.mean(), self.best_modelnet40_class_acc))
        logging.info('Test ModelNet40: top1_acc: {0} top3_acc: {1} top5_acc: {2}'.format(topk_acc[0].item(), topk_acc[1].item(), topk_acc[2].item()))

    def test_objaverse_lvis(self):
        self.model.eval()
        if self.config.training.use_text_proj:
            self.text_proj.eval()
        clip_text_feat = torch.from_numpy(self.objaverse_lvis_loader.dataset.clip_cat_feat).to(self.config.device)
        if self.config.training.use_text_proj:
            clip_text_feat = self.text_proj(clip_text_feat)
        per_cat_correct = torch.zeros(1156).to(self.config.device)
        per_cat_count = torch.zeros(1156).to(self.config.device)
        category2idx = self.objaverse_lvis_loader.dataset.category2idx
        idx2category = {v: k for k, v in category2idx.items()}
        
        logits_all = []
        labels_all = []
        with torch.no_grad():
            for data in tqdm(self.objaverse_lvis_loader):
                if not self.config.model.get("use_dense", False):
                    pred_feat = self.model(data['xyz'], data['features'], \
                                            device = self.config.device, \
                                            quantization_size = self.config.model.voxel_size)
                else:
                    pred_feat = self.model(data['xyz_dense'], data['features_dense'])
                logits = F.normalize(pred_feat, dim=1) @ F.normalize(clip_text_feat, dim=1).T
                labels = data['category'].to(self.config.device)
                logits_all.append(logits.detach())
                labels_all.append(labels)
                # calculate per class accuracy
                for i in torch.unique(labels):
                    idx = (labels == i)
                    if idx.sum() > 0:
                        per_cat_correct[i] += (logits[idx].argmax(dim=1) == labels[idx]).float().sum()
                        per_cat_count[i] += idx.sum()
        topk_acc, correct = self.accuracy(torch.cat(logits_all), torch.cat(labels_all), topk=(1,3,5,))

        overall_acc = per_cat_correct.sum() / per_cat_count.sum()
        per_cat_acc = per_cat_correct / per_cat_count

        if overall_acc > self.best_lvis_acc:
            self.best_lvis_acc = overall_acc
            self.save_model('best_lvis')

        logging.info('Test ObjaverseLVIS: overall acc: {0} class_acc: {1}'.format(overall_acc, per_cat_acc.mean()))
        logging.info('Test ObjaverseLVIS: top1_acc: {0} top3_acc: {1} top5_acc: {2}'.format(topk_acc[0].item(), topk_acc[1].item(), topk_acc[2].item()))
        
    def test_scanobjectnn(self):
        self.model.eval()
        if self.config.training.use_text_proj:
            self.text_proj.eval()
        clip_text_feat = torch.from_numpy(self.scanobjectnn_loader.dataset.clip_cat_feat).to(self.config.device)
        if self.config.training.use_text_proj:
            clip_text_feat = self.text_proj(clip_text_feat)
        per_cat_correct = torch.zeros(15).to(self.config.device)
        per_cat_count = torch.zeros(15).to(self.config.device)
        category2idx = self.scanobjectnn_loader.dataset.category2idx
        idx2category = {v: k for k, v in category2idx.items()}
        
        logits_all = []
        labels_all = []
        with torch.no_grad():
            for data in self.scanobjectnn_loader:
                if not self.config.model.get("use_dense", False):
                    pred_feat = self.model(data['xyz'], data['features'], \
                                            device = self.config.device, \
                                            quantization_size = self.config.model.voxel_size)
                else:
                    pred_feat = self.model(data['xyz_dense'], data['features_dense'])
                logits = F.normalize(pred_feat, dim=1) @ F.normalize(clip_text_feat, dim=1).T
                labels = data['category'].to(self.config.device)
                logits_all.append(logits.detach())
                labels_all.append(labels)
                # calculate per class accuracy
                for i in range(15):
                    idx = (labels == i)
                    if idx.sum() > 0:
                        per_cat_correct[i] += (logits[idx].argmax(dim=1) == labels[idx]).float().sum()
                        per_cat_count[i] += idx.sum()

        topk_acc, correct = self.accuracy(torch.cat(logits_all), torch.cat(labels_all), topk=(1,3,5,))

        overall_acc = per_cat_correct.sum() / per_cat_count.sum()
        per_cat_acc = per_cat_correct / per_cat_count

        logging.info('Test ScanObjectNN: overall acc: {0} class_acc: {1}'.format(overall_acc, per_cat_acc.mean()))
        logging.info('Test ScanObjectNN: top1_acc: {0} top3_acc: {1} top5_acc: {2}'.format(topk_acc[0].item(), topk_acc[1].item(), topk_acc[2].item()))
