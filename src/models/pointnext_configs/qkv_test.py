# import torch
# import torch.nn as nn
#
#
# class Attention(nn.Module):
#     def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
#         super().__init__()
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
#         self.scale = qk_scale or head_dim ** -0.5
#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)
#
#     def forward(self, x, context = None):
#
#         if context is None:
#             B, N, C = x.shape
#             qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#             q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
#         else:
#             # self.qkv.weight
#             B, N, C = x.shape
#             qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#             q= qkv[0]   # make torchscript happy (cannot use tensor as tuple)
#             B, N, C = context.shape
#             qkv = self.qkv(context).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#             k, v = qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
#
#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)
#
#         x = (attn @ v).transpose(1, 2).reshape(B, N, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x
#
# attn = Attention(dim=256, num_heads=8, qkv_bias=True)
# x = torch.randn([2, 128, 256])
# context = torch.randn([2, 16, 256])
# y = attn(x, context)

import torch
import torch.nn as nn
import torch.nn.functional as F


class StandardCrossAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super(StandardCrossAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"

        # 使用独立的线性层生成 query, key 和 value
        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)

        # 最后的输出投影层
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x, context):
        # 生成 query, key 和 value
        query = self.query_proj(x)
        key = self.key_proj(context)
        value = self.value_proj(context)

        # 划分为多头
        batch_size, x_seq_len, _ = x.shape
        context_seq_len = context.shape[1]
        query = query.view(batch_size, x_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, context_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, context_seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力分数
        scores = (query @ key.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)

        # 加权求和
        context = attn @ value
        context = context.transpose(1, 2).contiguous().view(batch_size, x_seq_len, self.dim)

        # 输出投影
        output = self.out_proj(context)

        return output


# 优化后的实现
class OptimizedCrossAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super(OptimizedCrossAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # 单一线性层生成 qkv 映射
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x, context):
        # 提取权重和偏置进行手动分割
        q_weight, k_weight, v_weight = self.qkv.weight.split(self.dim, dim=0)
        q_bias, k_bias, v_bias = self.qkv.bias.split(self.dim, dim=0)

        # 手动计算 query, key 和 value
        query = F.linear(x, q_weight, q_bias)
        key = F.linear(context, k_weight, k_bias)
        value = F.linear(context, v_weight, v_bias)

        # 划分为多头
        batch_size, x_seq_len, _ = x.shape
        context_seq_len = context.shape[1]
        query = query.view(batch_size, x_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, context_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, context_seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力分数
        scores = (query @ key.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)

        # 加权求和
        context = attn @ value
        context = context.transpose(1, 2).contiguous().view(batch_size, x_seq_len, self.dim)

        # 输出投影
        output = self.out_proj(context)

        return output


# 验证步骤
dim = 6  # 输入特征维度
num_heads = 2  # 注意力头数
batch_size = 1
x_seq_len = 3
context_seq_len = 5

# 设置随机种子以确保初始化相同
# torch.manual_seed(42)

# 初始化两个模型
standard_model = StandardCrossAttention(dim, num_heads)
optimized_model = OptimizedCrossAttention(dim, num_heads)

total_params = sum(p.numel() for p in standard_model.parameters())
print("模型总参数量:", total_params)

total_params = sum(p.numel() for p in optimized_model.parameters())
print("模型总参数量:", total_params)

print(standard_model)
print('\n\n\noptimized_model')
print(optimized_model)

# 将标准模型的权重复制到优化模型中
optimized_model.qkv.weight.data[:dim] = standard_model.query_proj.weight.data
optimized_model.qkv.weight.data[dim:2 * dim] = standard_model.key_proj.weight.data
optimized_model.qkv.weight.data[2 * dim:] = standard_model.value_proj.weight.data
optimized_model.qkv.bias.data[:dim] = standard_model.query_proj.bias.data
optimized_model.qkv.bias.data[dim:2 * dim] = standard_model.key_proj.bias.data
optimized_model.qkv.bias.data[2 * dim:] = standard_model.value_proj.bias.data
optimized_model.out_proj.weight.data = standard_model.out_proj.weight.data
optimized_model.out_proj.bias.data = standard_model.out_proj.bias.data

# 生成相同的输入
x = torch.randn(batch_size, x_seq_len, dim)
context = torch.randn(batch_size, context_seq_len, dim)

# 运行两个模型
standard_output = standard_model(x, context)
optimized_output = optimized_model(x, context)
print(standard_output)
print(optimized_output)

# 比较结果
print("差异：", torch.allclose(standard_output, optimized_output, atol=1e-6))


import torch.nn as nn

# 定义一个简单模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(128, 64)

model = SimpleModel()

# 获取模型中特定权重的参数量
num_params = model.fc.weight.numel()
print("参数量:", num_params)

total_params = sum(p.numel() for p in model.parameters())
print("模型总参数量:", total_params)

print('==' * 36)

import torch.nn as nn

# 定义一个简单模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)

        return x

class ComplexModel(nn.Module):
    def __init__(self):
        super(ComplexModel, self).__init__()
        self.fc0 = SimpleModel()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)

    def forward(self, x):
        x = self.fc0(x)
        x = self.fc1(x)
        x = self.fc2(x)

        return x


model = ComplexModel()

# 获取模型的 state_dict
state_dict = model.state_dict()

# 输出 state_dict 的内容
for param_tensor in state_dict:
    print(param_tensor, "\t", state_dict[param_tensor].size())

for name, param in state_dict.items():
    print(name, "\t", param.size())

