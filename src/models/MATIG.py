import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from .MATIG_point_encoder import PointcloudEncoder
from timm.models.layers import DropPath, trunc_normal_
import logging

## Transformers
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, context = None):
        B, N, C = x.shape
        if context is None:
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
        else:
            q_weight, k_weight, v_weight = self.qkv.weight.split(self.dim, dim=0)
            q_bias, k_bias, v_bias = self.qkv.bias.split(self.dim, dim=0)

            # 手动计算 query, key 和 value
            query = F.linear(x, q_weight, q_bias)
            key = F.linear(context, k_weight, k_bias)
            value = F.linear(context, v_weight, v_bias)

            # 划分为多头
            M = context.shape[1]
            q = query.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
            k = key.view(B, M, self.num_heads, self.head_dim).transpose(1, 2)
            v = value.view(B, M, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim, eps = 1e-6)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim, eps = 1e-6)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        
    def forward(self, x, context = None):
        x = x + self.drop_path(self.attn(self.norm1(x), context))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim=384, depth=4, num_heads=6, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.point_flag_emb = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.img_flag_emb = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.text_flag_emb = nn.Parameter(torch.zeros(1, 1, embed_dim))

        trunc_normal_(self.point_flag_emb, std=.02)
        trunc_normal_(self.img_flag_emb, std=.02)
        trunc_normal_(self.text_flag_emb, std=.02)
        # self.head = nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, pos, context = None, img_feature_size = 3, text_feature_size = 3, attn_type = 'cross'):
        if attn_type == 'cross':
            for _, block in enumerate(self.blocks):
                x = block(x + pos, context)
        elif attn_type == 'self':
            batch_size, len, _ = x.shape
            point_flag = self.point_flag_emb.expand(batch_size, len, -1)
            img_flag = self.img_flag_emb.expand(batch_size, img_feature_size, -1)
            text_flag = self.text_flag_emb.expand(batch_size, text_feature_size, -1)
            x = x + point_flag
            img = context[:, :img_feature_size, :] + img_flag
            text = context[:, img_feature_size:, :] + text_flag
            x = torch.concat([x, img, text], dim = 1)
            for _, block in enumerate(self.blocks):
                x[:, :len, :] = x[:, :len, :] + pos
                x = block(x)
            x = x[:, :len, :]

        x = self.norm(x)  # only return the mask tokens predict pixel
        return x

class MATIG_MODEL(nn.Module):
    def __init__(self, config):
        super().__init__()
        logging.info("MATIG")

        self.config = config
        self.mask_ratio = config.model.transformer_config.mask_ratio
        self.group_size = config.model.group_size
        self.num_group = config.model.num_group
        self.num_mask = int(self.mask_ratio * self.num_group)

        if config.model.transformer_config.encoder_type == 'base_lvis':
            point_transformer = timm.create_model(config.model.transformer_config.base_lvis.pc_model, drop_path_rate=config.model.transformer_config.drop_path_rate)
            checkpoint = torch.load(config.model.transformer_config.base_lvis.ckpt_path, map_location='cpu')
            print('loaded checkpoint {}'.format(config.model.transformer_config.base_lvis.ckpt_path))
            self.pc_feat_dim = config.model.transformer_config.base_lvis.pc_feat_dim
        elif config.model.transformer_config.encoder_type == 'base_no_lvis':
            point_transformer = timm.create_model(config.model.transformer_config.base_no_lvis.pc_model, drop_path_rate=config.model.transformer_config.drop_path_rate)
            checkpoint = torch.load(config.model.transformer_config.base_no_lvis.ckpt_path, map_location='cpu')
            print('loaded checkpoint {}'.format(config.model.transformer_config.base_no_lvis.ckpt_path))
            self.pc_feat_dim = config.model.transformer_config.base_no_lvis.pc_feat_dim
        else:
            exit(0)

        # create whole point cloud encoder
        self.point_encoder = PointcloudEncoder(point_transformer, config)
        print(self.point_encoder)

        sd = checkpoint['module']
        if next(iter(sd.items()))[0].startswith('module'):
            sd = {k[len('module.'):]: v for k, v in sd.items()}
        # if next(iter(sd.items()))[0].startswith('point_encoder'):
        #     sd = {k[len('point_encoder.'):]: v for k, v in sd.items()}
        sd1 = {}
        for k, v in sd.items():
            if k.startswith('point_encoder'):
                sd1[k[len('point_encoder.'):]] = v

        self.point_encoder.load_state_dict(sd1)
        print("load done!!!{}".format(len(sd1)))
        print('\n'.join(sd1.keys()))

        self.trans_dim = self.pc_feat_dim
        self.context_dim = config.model.context_channel
        self.decoder_dim = config.model.transformer_config.decoder_dim

        if self.context_dim != self.decoder_dim:
            self.context_embed = nn.Linear(self.context_dim, self.decoder_dim, bias=True)
            self.init_linear(self.context_embed)
        self.decoder_embed = nn.Linear(self.trans_dim, self.decoder_dim, bias=True)
        self.init_linear(self.decoder_embed)
        if self.trans_dim != self.context_dim:
            self.cls_trans = nn.Linear(self.trans_dim, self.context_dim, bias=True)
            self.init_linear(self.cls_trans)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.decoder_dim))
        trunc_normal_(self.mask_token, std=.02)
        self.decoder_cls_pos = nn.Parameter(torch.randn(1, 1, self.decoder_dim))
        torch.nn.init.normal_(self.decoder_cls_pos, std=.02)

        self.decoder_pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.decoder_dim)
        )
        self.init_linear(self.decoder_pos_embed[0])
        self.init_linear(self.decoder_pos_embed[2])

        self.decoder_depth = config.model.transformer_config.decoder_depth
        self.decoder_num_heads = config.model.transformer_config.decoder_num_heads
        self.drop_path_rate = config.model.transformer_config.drop_path_rate
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.decoder_depth)]
        self.MAE_decoder = TransformerDecoder(
            embed_dim=self.decoder_dim,
            depth=self.decoder_depth,
            drop_path_rate=dpr,
            num_heads=self.decoder_num_heads,
            qkv_bias=True
        )

        logging.info("GuidanceUni3d divide point cloud into G{} x S{} points".format(self.num_group, self.group_size))

        # prediction head
        self.coord_pred = nn.Sequential(
            # nn.Conv1d(self.trans_dim, 1024, 1),
            # nn.BatchNorm1d(1024),
            # nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(self.decoder_dim, 3 * self.group_size, 1)
        )
        torch.nn.init.xavier_uniform_(self.coord_pred[0].weight, gain=1)  # 或者使用xavier_normal_
        torch.nn.init.constant_(self.coord_pred[0].bias, 0)  # 偏置初始化为0

    def init_linear(self, layer):
        # 例如，使用Xavier均匀分布初始化权重和偏置
        torch.nn.init.xavier_uniform_(layer.weight)
        if layer.bias is not None:
            torch.nn.init.zeros_(layer.bias)

    def load_mae_mode(self):

        from mmpretrain import get_model
        mae_model = get_model(model = self.config.model.mae_pretrained.model, pretrained=self.config.model.mae_pretrained.pretrained)
        # mae_model.to(self.config.device)

        param_mapping = [{
            f'backbone.layers.{i}.ln1.weight': f'MAE_encoder.blocks.blocks.{i}.norm1.weight',
            f'backbone.layers.{i}.ln1.bias': f'MAE_encoder.blocks.blocks.{i}.norm1.bias',
            f'backbone.layers.{i}.ln2.weight': f'MAE_encoder.blocks.blocks.{i}.norm2.weight',
            f'backbone.layers.{i}.ln2.bias': f'MAE_encoder.blocks.blocks.{i}.norm2.bias',
            f'backbone.layers.{i}.ffn.layers.0.0.weight': f'MAE_encoder.blocks.blocks.{i}.mlp.fc1.weight',
            f'backbone.layers.{i}.ffn.layers.0.0.bias': f'MAE_encoder.blocks.blocks.{i}.mlp.fc1.bias',
            f'backbone.layers.{i}.ffn.layers.1.weight': f'MAE_encoder.blocks.blocks.{i}.mlp.fc2.weight',
            f'backbone.layers.{i}.ffn.layers.1.bias': f'MAE_encoder.blocks.blocks.{i}.mlp.fc2.bias',
            f'backbone.layers.{i}.attn.qkv.weight': f'MAE_encoder.blocks.blocks.{i}.attn.qkv.weight',
            f'backbone.layers.{i}.attn.qkv.bias': f'MAE_encoder.blocks.blocks.{i}.attn.qkv.bias',
            f'backbone.layers.{i}.attn.proj.weight': f'MAE_encoder.blocks.blocks.{i}.attn.proj.weight',
            f'backbone.layers.{i}.attn.proj.bias': f'MAE_encoder.blocks.blocks.{i}.attn.proj.bias',
        } for i in range(len(mae_model.backbone.layers))]

        param_mapping.extend([{
            f'neck.decoder_blocks.{i}.ln1.weight': f'MAE_decoder.blocks.{i}.norm1.weight',
            f'neck.decoder_blocks.{i}.ln1.bias': f'MAE_decoder.blocks.{i}.norm1.bias',
            f'neck.decoder_blocks.{i}.attn.qkv.weight': f'MAE_decoder.blocks.{i}.attn.qkv.weight',
            f'neck.decoder_blocks.{i}.attn.qkv.bias': f'MAE_decoder.blocks.{i}.attn.qkv.bias',
            f'neck.decoder_blocks.{i}.attn.proj.weight': f'MAE_decoder.blocks.{i}.attn.proj.weight',
            f'neck.decoder_blocks.{i}.attn.proj.bias': f'MAE_decoder.blocks.{i}.attn.proj.bias',
            f'neck.decoder_blocks.{i}.ln2.weight': f'MAE_decoder.blocks.{i}.norm2.weight',
            f'neck.decoder_blocks.{i}.ln2.bias': f'MAE_decoder.blocks.{i}.norm2.bias',
            f'neck.decoder_blocks.{i}.ffn.layers.0.0.weight': f'MAE_decoder.blocks.{i}.mlp.fc1.weight',
            f'neck.decoder_blocks.{i}.ffn.layers.0.0.bias': f'MAE_decoder.blocks.{i}.mlp.fc1.bias',
            f'neck.decoder_blocks.{i}.ffn.layers.1.weight': f'MAE_decoder.blocks.{i}.mlp.fc2.weight',
            f'neck.decoder_blocks.{i}.ffn.layers.1.bias': f'MAE_decoder.blocks.{i}.mlp.fc2.bias'
        } for i in range(len(mae_model.neck.decoder_blocks))])
        param_mapping = dict((k, v) for d in param_mapping for k, v in d.items())

        # print('\n'.join(["{:<45} --> {}".format(k, v) for k, v in param_mapping.items()]))
        # print(len(param_mapping))
        param_mapping.update({
            'backbone.ln1.weight': 'MAE_encoder.norm.weight',
            'backbone.ln1.bias'  : 'MAE_encoder.norm.bias',
            'neck.ln1.weight'    : 'MAE_decoder.norm.weight',
            'neck.ln1.bias'      : 'MAE_decoder.norm.bias',
            'backbone.cls_token' : 'MAE_encoder.encoder_cls_token',
            'neck.mask_token'    : 'mask_token'
        })
        print('len of param_mapping: ', len(param_mapping))
        logging.info("len of param_mapping: {}".format(len(param_mapping)))
        logging.info("param_mapping: {}".format(param_mapping))


        # 遍历映射关系并赋值
        src_md = mae_model.state_dict()
        tgt_md = self.state_dict()
        for src_key, tgt_key in param_mapping.items():
            assert src_key in src_md, "{} not in src_model".format(src_key)
            assert tgt_key in tgt_md, "{} not in tgt_model".format(tgt_key)
            tgt_md[tgt_key].copy_(src_md[src_key])
        self.load_state_dict(tgt_md)
        print('DONE!!!')

        res = []
        import re
        for src_key, tgt_key in param_mapping.items():
            res.append(torch.equal(
                eval(re.sub(r'\.(\d+)|\b(\d+)(?=\.)', lambda m: '[' + m.group(1) + ']' if m.group(1) else '',
                            'mae_model.' + src_key)),
                eval(re.sub(r'\.(\d+)|\b(\d+)(?=\.)', lambda m: '[' + m.group(1) + ']' if m.group(1) else '',
                            'self.' + tgt_key))
            ))

        logging.info("{} {}".format(len(res), sum(res)))
        assert sum(res) == len(param_mapping), "some pytorch variable not copy!!!"

        mae_model.cpu()
        del mae_model
        torch.cuda.empty_cache()

        logging.info("mae model parameters have been loaded, and free mae_model")

    def forward(self, pts, features, img_feature = None, text_feature = None, vis = False, **kwargs):
        # center is absolute coord, neighborhood is relative coord with centers
        neighborhood, center, x_vis, mask = self.point_encoder(pts, features)
        point_cls_emb = x_vis[:, 0, :]

        if self.trans_dim != self.context_dim:
            point_cls_emb = self.cls_trans(point_cls_emb)

        if not (img_feature is None and text_feature is None):
            context = torch.cat([img_feature, text_feature], dim = 1)
            context = self.context_embed(context)
        else:
            context = None

        if context is None:
            return point_cls_emb

        x_vis = self.decoder_embed(x_vis)
        B, N, C = x_vis.shape # B VIS CF

        decoder_cls_pos = self.decoder_cls_pos.expand(B, -1, -1)
        pos_emb_vis = self.decoder_pos_embed(center[~mask]).reshape(B, -1, C)
        pos_emb_mask = self.decoder_pos_embed(center[mask]).reshape(B, -1, C)
        pos_full = torch.cat([decoder_cls_pos, pos_emb_vis, pos_emb_mask], dim=1)
        # print("pos_emb_vis.shape: ", pos_emb_vis.shape, "pos_emb_mask.shape: ", pos_emb_mask.shape)

        mask_tokens = self.mask_token.expand(B, pos_emb_mask.shape[1], -1)
        x_full = torch.cat([x_vis, mask_tokens], dim=1)

        x_rec = self.MAE_decoder(x_full, pos_full, context, img_feature.shape[1], text_feature.shape[1], self.config.model.transformer_config.decoder_attn_type)
        x_rec_mask = x_rec[:, -self.num_mask:, :]

        rebuild_points = self.coord_pred(x_rec_mask.transpose(1, 2)).transpose(1, 2)
        rebuild_points = rebuild_points.reshape(B, self.num_mask, self.group_size, 3)
        gt_mask_points = neighborhood[mask].reshape(B, self.num_mask, self.group_size, 3)

        return point_cls_emb, rebuild_points, gt_mask_points

def make(cfg):
    return MATIG_MODEL(cfg)


