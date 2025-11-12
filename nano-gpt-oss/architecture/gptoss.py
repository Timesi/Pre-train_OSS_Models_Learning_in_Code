import json
import math
import os

import torch
from typing import Optional, Union
from dataclasses import dataclass
import torch.distributed as dist

@dataclass
class ModelConfig:
    num_hidden_layers: int = 24         # Transformer层数
    num_experts: int = 32               # MoE专家数量
    experts_per_token: int = 4          # 每个token激活的专家
    vocab_size: int = 201088            # 词表大小
    hidden_size: int = 2880             # 隐藏层维度（每个token的维度）
    intermediate_size: int = 2880       # 中间层维度（FFN）
    swiglu_limit: float = 7.0           # 设置SwiGLU激活函数的限制值，用于控制激活函数的输入范围
    head_dim: int = 64                  # 每个注意力头维度
    num_attention_heads: int = 64       # 矩阵注意力头数
    num_key_value_heads: int = 8        # kv矩阵头数，分组查询注意力（和分组数量相同）
    sliding_window: int = 128           # 滑动窗口大小
    initial_context_length: int = 4096  # 初始内容长度
    rope_theta: float = 150000.0        # RoPE基础频率参数
    rope_scaling_factor: float = 32.0   # 位置编码缩放因子
    rope_ntk_alpha: float = 1.0         # NTK缩放alpha参数
    rope_ntk_beta: float = 32.0         # NTK缩放beta参数


class RMSNorm(torch.nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-05, device: Optional[torch.device] = None):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        # 缩放参数
        self.scale = torch.nn.Parameter(torch.ones(num_features), device=device, dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.num_features
        t, dtype = x.float(), x.dtype
        t = t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.scale * t).to(dtype)

# 对向量旋转位置进行编码
# [new_x] = [cos(θ)  -sin(θ)] [x]
# [new_y] = [sin(θ)   cos(θ)] [y]
#
# new_x = x·cos(θ) - y·sin(θ)
# new_y = x·sin(θ) + y·cos(θ)
def _apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    cos = cos.unsqueeze(-2).to(x.dtype)     # 对齐输入的维度
    sin = sin.unsqueeze(-2).to(x.dtype)
    x1, x2 = torch.chunk(x, 2, dim=-1)  # 将输入向量拆分为两部分，各自表示嵌入的一半维度
    # 对两半向量应用旋转公式
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    return torch.cat((o1, o2), dim=-1)  # 拼接两个向量

class RotaryEmbedding(torch.nn.Module):
    def __init__(
            self,
            head_dim: int,      # 注意力头的维度
            base: int,          # 频率基底参数，150000
            dtype: torch.dtype, # 数据类型
            initial_context_length: int = 4096, # 初始内容的长度
            scaling_factor: float = 1.0,        # 缩放因子，大于1.0时启用YaRN扩展
            ntk_alpha: float = 1.0,             # NTK缩放的alpha参数
            ntk_beta: float = 32.0,             # NTK缩放的beta参数
            device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.base = base
        self.dtype = dtype
        self.initial_context_length = initial_context_length
        self.scaling_factor = scaling_factor
        self.ntk_alpha = ntk_alpha
        self.ntk_beta = ntk_beta
        self.device =  device

    def _compute_concentration_and_inv_freq(self) -> torch.Tensor:
        # 生成两两分组之后的频率
        freq = self.base**(-torch.arange(0, self.head_dim, 2, device=self.device, dtype=self.dtype) / self.head_dim)
        # concentration是集中度参数，用于控制频率分布的集中程度，scaling_factor越大，concentration越小，频率分布越分散
        if self.scaling_factor > 1.0:
            # 计算集中度参数，YaRN论文中的公式
            concentration = (
                0.1 * math.log(self.scaling_factor) + 1.0
            )  # YaRN concentration

            # 计算一半的维度数
            # 例如：head_dim=8时，d_half = 4.0
            d_half = self.head_dim / 2
            # NTK by parts
            low = (
                d_half
                * math.log(self.initial_context_length / (self.ntk_beta * 2 * math.pi))
                / math.log(self.base)
            )
            high = (
                d_half
                * math.log(self.initial_context_length / (self.ntk_alpha * 2 * math.pi))
                / math.log(self.base)
            )
            assert 0 < low < high < d_half - 1

            interpolation = 1.0 / (self.scaling_factor * freq)
            extrapolation = 1.0 / freq

            ramp = (
                torch.arange(d_half, dtype=torch.float32, device=freq.device) - low
            ) / (high - low)
            mask = 1 - ramp.clamp(0, 1)

            inv_freq = interpolation * (1 - mask) + extrapolation * mask
        else:
            concentration = 1.0
            inv_freq = 1.0 / freq

        return concentration, inv_freq

    def _compute_cos_sin(self, num_tokens: int):
        # 调用上面的方法获取集中度和逆频率
        concentration, inv_freq = self._compute_concentration_and_inv_freq()
        # 生成位置索引向量，例如num_tokens=5时，t=[0,1,2,3,4]
        t = torch.arange(num_tokens, dtype=torch.float32, device=self.device)
        # 使用einsum计算每个位置×每个频率的组合，freqs表示每个位置在每个维度上的旋转相位角theta
        # freqs[i][j] = t[i] × inv_freq[j]，freqs[i][j] 表示第i个token在第j个维度上的旋转相位角
        # t[i] = 第i个token的位置索引 (0, 1, 2, 3, 4...)，inv_freq[j] = 第j个维度的逆频率 = 1/(base^(2j/head_dim))
        # 对应公式：theta = w_i * p = 1/(10000^(2i/d)) * p
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        # 根据旋转相位角计算cos和sin值，并乘以集中度参数进行调整
        cos = freqs.cos() * concentration
        sin = freqs.sin() * concentration
        # 返回形状为(num_tokens, head_dim//2)的cos和sin矩阵
        return cos, sin

    def forward(
        self,
        query: torch.Tensor,    # query矩阵
        key: torch.Tensor,      # key矩阵
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_tokens = query.shape[0]                         # 获取输入序列的长度
        cos, sin = self._compute_cos_sin(num_tokens)        # 计算该序列长度的cos和sin位置编码

        query_shape = query.shape                           # 保存原始query形状以便后续恢复
        query = query.view(num_tokens, -1, self.head_dim)   # 改变query的形状以适应RoPE应用：将最后一个维度作为head_dim
        query = _apply_rotary_emb(query, cos, sin)          # 应用旋转位置编码到query
        query = query.reshape(query_shape)                  # 恢复query到原始形状

        # 同上
        key_shape = key.shape
        key = key.view(num_tokens, -1, self.head_dim)
        key = _apply_rotary_emb(key, cos, sin)
        key = key.reshape(key_shape)
        return query, key                                   # 返回旋转之后的query和key矩阵


def sdpa(Q, K, V, S, sm_scale, sliding_window=0):
    n_tokens, n_heads, q_mult, d_head = Q.shape
    assert K.shape == (n_tokens, n_heads, d_head)
    assert V.shape == (n_tokens, n_heads, d_head)
    # 将K、V扩展到与Q相同的维度，让其与Q相匹配，用于分组注意力计算
    K = K[:, :, None, :].expand(-1, -1, q_mult, -1)
    V = V[:, :, None, :].expand(-1, -1, q_mult, -1)
    # 创建汇聚层，汇聚位置允许模型关注特殊的全局信息，类似于为每个注意力头提供全局的特殊token
    # 准备sink logits，因为每个头共享一个值，所以先扩展
    S = S.reshape(n_heads, q_mult, 1, 1).expand(-1, -1, n_tokens, -1)
    # 创建一个上三角矩阵，上三角部分为负无穷
    mask = torch.triu(Q.new_full((n_tokens, n_tokens), -float("inf")), diagonal=1)
    if sliding_window > 0:
        # 创建一个下三角矩阵，对角线以上部分为0，以下部分为负无穷
        # diagonal=-sliding_window表示将对角线向下偏移 sliding_window 位置
        # 然后将上三角矩阵mask和下三角矩阵相加得到最终的掩码矩阵
        mask += torch.tril(mask.new_full((n_tokens, n_tokens), -float("inf")), diagonal=-sliding_window)
    QK = torch.einsum("qhmd,khmd->hmqk", Q, K)      # Q*K^T
    QK *= sm_scale
    QK += mask[None, None, :, :]
    QK = torch.cat([QK, S], dim=-1)     # 将汇聚层拼接到最后面
    W = torch.softmax(QK, dim=-1)
    W = W[..., :-1]     # 去掉汇聚层
    attn = torch.einsum("hmqk,khmd->qhmd", W, V)
    # atten的形状为 (n_tokens, n_heads, q_mult, d_head)
    # 通过reshape转换为(n_tokens, -1)，即(序列长度, 展平的特征维度)与后续out层的期望输入保持一致
    return attn.reshape(n_tokens, -1)


class AttentionBlock(torch.nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int = 0, device: Optional[torch.device] = None):
        super().__init__()
        self.head_dim = config.head_dim                         # 注意力头维度
        self.num_attention_heads = config.num_attention_heads   # 注意力头数量
        self.num_key_value_heads = config.num_key_value_heads   # KV头的数量（未复制前）

        # 只对偶数层应用滑动窗口
        self.silding_window = config.silding_window if layer_idx % 2 == 0 else 0
        # 创建注意力汇聚层，每个头中的值相同
        self.sinks = torch.nn.Parameter(torch.empty(config.num_attention_heads, device=device, dtype=torch.bfloat16))
        self.norm = RMSNorm(config.hidden_size, device=device)
        # 分组查询注意力，多个Q头共享一组K/V头，因此头数为config.num_attention_heads + 2 * config.num_key_value_heads
        qkv_dim = config.head_dim * (config.num_attention_heads + 2 * config.num_key_value_heads)
        self.qkv = torch.nn.Linear(config.hidden_size, qkv_dim, device=device, dtype=torch.bfloat16)    # 为了得到QKV矩阵
        self.out = torch.nn.Linear(config.head_dim * config.num_attention_heads, config.hidden_size, device=device, dtype=torch.bfloat16)
        self.sm_scale = 1 / math.sqrt(self.head_dim)    # 缩放因子
        self.rope = RotaryEmbedding(
            config.head_dim,
            config.rope_theta,
            torch.float32,
            initial_context_length=config.initial_context_length,
            scaling_factor=config.rope_scaling_factor,
            ntk_alpha=config.ntk_alpha,
            ntk_beta=config.ntk_beta,
            device=device,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = self.norm(x)
        qkv = self.qkv(t)
        # 从qkv中分别取出Q、K、V矩阵
        q = qkv[:, : self.num_attention_heads * self.head_dim].contigous()
        k = qkv[
            :,
            self.num_attention_heads
            * self.head_dim: (self.num_attention_heads + self.num_key_value_heads)
            * self.head_dim
        ].contiguous()
        v = qkv[
            :,
            (self.num_attention_heads + self.num_key_value_heads)
            * self.head_dim: (self.num_attention_heads + 2 * self.num_key_value_heads)
            * self.head_dim
        ].contiguous()

        # 重新组织 Query 张量的维度结构，以支持分组查询注意力
        # Q先将num_key_value_heads作为head（这里的head可以理解为分组）
        # 接着将head再分解为（self.num_attention_heads // self.num_key_value_heads, self.head_dim），相当于按照维度来划分每个分组中头的数量
        # 举例：
        # 假设 q 张量如下（已展开为2D）形状: (2, 12)  其中 12 = 4 * 3 (num_attention_heads * head_dim)：
        # q = [
        #     [q111, q112, q113, q121, q122, q123, q131, q132, q133, q141, q142, q143],  # 第1个token
        #     [q211, q212, q213, q221, q222, q223, q231, q232, q233, q241, q242, q243]   # 第2个token
        # ]
        #
        # 重塑后张量示意：
        # 第1个维度(-1): 序列长度 = 2
        # 第2个维度(num_key_value_heads): 分组数 = 2
        # 第3个维度(num_attention_heads // num_key_value_heads): 每组头数 = 2
        # 第4个维度(head_dim): 头维度 = 3
        # [
        #   # 第1个token
        #   [
        #     # 第1组 (对应第1个KV头)
        #     [
        #       [q111, q112, q113],  # 第1组中的第1个Q头
        #       [q121, q122, q123]   # 第1组中的第2个Q头
        #     ],
        #     # 第2组 (对应第2个KV头)
        #     [
        #       [q131, q132, q133],  # 第2组中的第1个Q头
        #       [q141, q142, q143]   # 第2组中的第2个Q头
        #     ]
        #   ],
        #   # 第2个token
        #   [
        #     # 第1组 (对应第1个KV头)
        #     [
        #       [q211, q212, q213],  # 第1组中的第1个Q头
        #       [q221, q222, q223]   # 第1组中的第2个Q头
        #     ],
        #     # 第2组 (对应第2个KV头)
        #     [
        #       [q231, q232, q233],  # 第2组中的第1个Q头
        #       [q241, q242, q243]   # 第2组中的第2个Q头
        #     ]
        #   ]
        # ]
        # 假设 k 张量如下（已展开为2D）形状: (2, 6)  其中 6 = 2 * 3 (num_key_value_heads * head_dim)：
        # k = [
        #     [k111, k112, k113, k121, k122, k123],  # 第1个token的所有KV头数据
        #     [k211, k212, k213, k221, k222, k223]  # 第2个token的所有KV头数据
        # ]
        #
        # 重塑后张量示意：
        # 第1个维度(-1): 序列长度 = 2
        # 第2个维度(num_key_value_heads): KV头数 = 2
        # 第3个维度(head_dim): 头维度 = 3
        #
        # [
        #   # 第1个token
        #   [
        #     [k111, k112, k113],  # 第1个KV头
        #     [k121, k122, k123]   # 第2个KV头
        #   ],
        #   # 第2个token
        #   [
        #     [k211, k212, k213],  # 第1个KV头
        #     [k221, k222, k223]   # 第2个KV头
        #   ]
        # ]
        q = q.view(-1, self.num_key_value_heads, self.num_attention_heads // self.num_key_value_heads, self.head_dim)
        # K中num_key_value_heads表示head，head_dim表示head的维度
        # 在后续操作中还需要复制self.num_attention_heads // self.num_key_value_heads份以对应上Q
        k = k.view(-1, self.num_key_value_heads, self.head_dim)
        v = v.view(-1, self.num_key_value_heads, self.head_dim)
        q, k = self.rope(q, k)
        t = sdpa(q, k, v, self.sinks, self.sm_scale, self.silding_window)
        t = self.out(t)
        return t + x


def swiglu(x, alpha: float = 1.702, limit: float = 7.0):
    # 将输入张量在最后一个维度上分成两半
    # x_glu: 偶数索引位置的元素，用作门控信号
    # x_linear: 奇数索引位置的元素，用作线性变换部分
    x_glu, x_linear = x[..., ::2], x[..., 1::2]
    # 对两部分输入进行裁剪，防止数值过大
    # x_glu: 限制在(-∞, limit] 范围内
    # x_linear 限制在 [-limit, limit] 范围内
    x_glu = x_glu.clamp(min=None, max=limit)
    x_linear = x_linear.clamp(min=-limit, max=limit)
    # Swish 激活函数的实现: x * sigmoid(alpha * x),用作门控信号，决定多少信息可以通过
    out_glu = x_glu * torch.sigmoid(alpha * x_glu)
    # 将门控信号与线性部分相乘,需要注意这里对x_linear添加了偏置值1
    return out_glu * (x_linear + 1)


class MLPBlock(torch.nn.Module):
    def __init__(self, config: ModelConfig, device: Optional[torch.device] = None):
        super().__init__()
        self.num_experts = config.num_experts   # 所有专家数量
        self.experts_per_token = config.experts_per_token   # 每个token激活的的专家数量
        self.swiglu_limit = config.swiglu_limit     # 设置SwiGLU激活函数的限制值，用于控制激活函数的输入范围
        # 获取分布式训练中的进程数量，如果未启用分布式训练则默认为1
        # 用于在分布式环境中正确分配计算资源
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.norm = RMSNorm(config.hidden_size, device=device)
        # 创建门控网络，用于决定每个token应该分配给哪些专家
        # 输入维度为隐藏层大小，输出维度为专家数量
        self.gate = torch.nn.Linear(config.hidden_size, config.num_experts, device=device, dtype=torch.bfloat16)
        assert config.intermediate_size % self.world_size == 0

        # 创建专家网络集合
        self.experts = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(
                    config.hidden_size,     # 输入维度2880
                    # 乘2是因为后面要做SwiGLU的“门控”激活,即一半当“门”，一半当“值”
                    config.intermediate_size * 2 // self.world_size,
                    device=device,
                    dtype=torch.bfloat16,
                ),
                torch.nn.Linear(
                    config.intermediate_size // self.world_size,
                    config.hidden_size,
                    device=device,
                    dtype=torch.bfloat16,
                )
            ) for _ in range(config.num_experts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len, hidden_size = x.shape
        t = self.norm(x)        # 输入x的shape为(seq_len, hidden_size)
        g = self.gate(t)

        # 假设输入 x 形状: (2, 4)
        # 序列长度 seq_len = 2（2个token）
        # 隐藏层维度 hidden_size = 4
        # 专家总数 num_experts = 6
        # 每个token激活专家数 experts_per_token = 3
        # x = [
        #     [0.1, 0.2, 0.3, 0.4],  # 第1个token
        #     [0.5, 0.6, 0.7, 0.8]   # 第2个token
        # ]
        #
        # 经过 self.norm(x) 得到 t
        # t = [
        #     [0.09, 0.18, 0.27, 0.36],  # 第1个token
        #     [0.45, 0.54, 0.63, 0.72]  # 第2个token
        # ]
        #
        # 经过门控网络 self.gate(t) 得到 g (每个token对6个专家的得分)
        # g = [
        #     [2.1, 0.5, 3.2, 1.8, 2.5, 0.9],  # 第1个token对6个专家的得分
        #     [1.2, 2.8, 0.9, 3.1, 1.5, 2.0]  # 第2个token对6个专家的得分
        # ]

        # token选择得分最高的前k个专家
        experts = torch.topk(g, k=self.experts_per_token, dim=-1, sorted=True)      # 获取每个列维度前k个专家的索引
        expert_weights = torch.nn.functional.softmax(experts.values, dim=-1)        # 将专家选择的原始得分转换为归一化的权重
        expert_indices = experts.indices        # 获取实际处理该token的专家的索引

        # experts = torch.topk(g, k=3, dim=-1, sorted=True)
        # 第1个token选择的前3个专家索引: [2, 4, 0] (对应得分: [3.2, 2.5, 2.1])
        # 第2个token选择的前3个专家索引: [3, 1, 5] (对应得分: [3.1, 2.8, 2.0])
        #
        # expert_indices = [
        #     [2, 4, 0],  # 第1个token选择的专家索引
        #     [3, 1, 5]   # 第2个token选择的专家索引
        # ]
        #
        # experts.values = [
        #     [3.2, 2.5, 2.1],  # 第1个token选择的专家得分
        #     [3.1, 2.8, 2.0]   # 第2个token选择的专家得分
        # ]
        #
        # expert_weights = softmax(experts.values, dim=-1)
        # expert_weights = [
        #     [0.52, 0.26, 0.22],  # 第1个token的专家权重
        #     [0.45, 0.32, 0.23]  # 第2个token的专家权重
        # ]

        # 展平
        t_flat = t.view(-1, hidden_size)        # (seq_len, hidden_size)
        expert_indices_flat = expert_indices.view(-1, self.experts_per_token)   # (seq_len, experts_per_token)
        expert_weights_flat = expert_weights.view(-1, self.experts_per_token)   # (seq_len, experts_per_token)
        output = torch.zeros_like(t_flat)

        # t_flat = t.view(-1, hidden_size)  # 形状不变，仍为(2, 4)
        # t_flat = [
        #     [0.09, 0.18, 0.27, 0.36],  # 第1个token
        #     [0.45, 0.54, 0.63, 0.72]   # 第2个token
        # ]
        #
        # expert_indices_flat = expert_indices.view(-1, experts_per_token)  # 形状(2, 3)
        # expert_indices_flat = [
        #     [2, 4, 0],  # 第1个token选择的专家索引
        #     [3, 1, 5]   # 第2个token选择的专家索引
        # ]
        #
        # expert_weights_flat = expert_weights.view(-1, experts_per_token)  # 形状(2, 3)
        # expert_weights_flat = [
        #     [0.52, 0.26, 0.22],  # 第1个token的专家权重
        #     [0.45, 0.32, 0.23]   # 第2个token的专家权重
        # ]
        #
        # output = torch.zeros_like(t_flat)  # 形状(2, 4)的零矩阵
        # output = [
        #     [0.0, 0.0, 0.0, 0.0],  # 第1个token的输出
        #     [0.0, 0.0, 0.0, 0.0]  # 第2个token的输出
        # ]

        # 遍历每个专家
        for expert_idx in range(self.num_experts):
            # 判断每个token是否激活该专家
            mask = (expert_indices_flat == expert_idx).any(dim=-1)
            if not mask.any():
                continue

            # 当 expert_idx = 0 时:
            # mask = (expert_indices_flat == 0).any(dim=-1)
            # expert_indices_flat = [[2, 4, 0], [3, 1, 5]]
            # 比较结果: [[False, False, True], [False, False, False]]
            # mask = [True, False]  # 第1个token激活了专家0，第2个token没有激活

            # 取出激活专家的行号
            token_indices = torch.where(mask)[0]

            # token_indices = torch.where(mask)[0]
            # mask = [True, False]
            # token_indices = [0]  # 只有第1个token激活了专家0

            # 取出对应专家的列号
            expert_pos = (expert_indices_flat[token_indices] == expert_idx).nonzero(as_tuple=True)[1]

            # expert_pos = (expert_indices_flat[token_indices] == expert_idx).nonzero(as_tuple=True)[1]
            # expert_indices_flat[0] = [2, 4, 0]
            # 比较 [2, 4, 0] == 0 得到 [False, False, True]
            # nonzero得到位置索引: 2
            # expert_pos = [2]  # 专家0在第1个token的专家列表中位于第3个位置(索引2)

            # 获取对应专家的输入
            expert_input = t_flat[token_indices]

            # expert_input = t_flat[token_indices]
            # t_flat[0] = [0.09, 0.18, 0.27, 0.36]
            # expert_input = [[0.09, 0.18, 0.27, 0.36]]  # 第1个token的输入

            # 取出对应位置的权重
            weights = expert_weights_flat[token_indices, expert_pos]

            # weights = expert_weights_flat[token_indices, expert_pos]
            # expert_weights_flat[0, 2] = 0.22
            # weights = [0.22]  # 专家0对第1个token的权重

            # 前向传播通过专家
            expert_out = expert_input
            expert_out = self.experts[expert_idx][0](expert_out)    # 通过第一个线性层
            expert_out = swiglu(expert_out, limit=self.swiglu_limit)    # 通过激活层
            expert_out = self.experts[expert_idx][1](expert_out)    # 通过第二个线性层

            # expert_out = expert_input  # [[0.09, 0.18, 0.27, 0.36]]
            # expert_out = self.experts[0][0](expert_out)    # 通过第一个线性层
            # expert_out = swiglu(expert_out, limit=self.swiglu_limit)    # 通过激活层
            # expert_out = self.experts[0][1](expert_out)    # 通过第二个线性层
            # 假设最终输出为: [[0.15, 0.25, 0.35, 0.45]]

            output[token_indices] += expert_out * weights.unsqueeze(-1)

            # output[token_indices] += expert_out * weights.unsqueeze(-1)
            # output[0] += [0.15, 0.25, 0.35, 0.45] * 0.22
            # output[0] += [0.033, 0.055, 0.077, 0.099]
            # 经过所有专家处理后:
            # output = [
            #     [0.033, 0.055, 0.077, 0.099],  # 第1个token的加权输出
            #     [0.067, 0.089, 0.111, 0.133]   # 第2个token的加权输出
            # ]

        if self.world_size > 1:
            dist.all_reduce(output, op=dist.ReduceOp.SUM)

        output = output.view(seq_len, hidden_size)
        # 最终返回 x + output 实现残差连接
        return x + output


class TransformerBlock(torch.nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int, device: Optional[torch.device] = None):
        super().__init__()
        self.layer_idx = layer_idx
        self.atten = AttentionBlock(config, layer_idx, device=device)
        self.mlp = MLPBlock(config, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.atten(x)
        x = self.mlp(x)
        return x


class Transformer(torch.nn.Module):
    def __init__(self, config: ModelConfig, device: Optional[torch.device] = None):
        super().__init__()
        # 创建词嵌入层
        self.embedding = torch.nn.Embedding(config.vocab_size, config.hidden_size, device=device, dtype=torch.bfloat16)
        # 创建transformer block列表
        self.block = torch.nn.ModuleList(
            [
                TransformerBlock(config, layer_idx, device=device)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        # 创建归一化层
        self.norm = RMSNorm(config.hidden_size, device=device)
        # 创建unembedding层将模型输出映射回vocab_size维向量
        self.unembedding = torch.nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            device=device,
            dtype=torch.bfloat16,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        for block in self.block:
            x = block(x)
        x = self.norm(x)
        x = self.unembedding(x)
        return x

    # 用于从检查点加载预训练模型的静态方法
    @staticmethod
    def from_checkpoint(
        path: str, device: Union[str, torch.device]
    ) -> "Transformer":
        if not isinstance(device, torch.device):
            device = torch.device(device)

        # 加载配置文件
        config_path = os.path.join(path, "config.json")
        with open(config_path, "r") as f:
            json_config = json.load(f)
            config = ModelConfig(**json_config)

        # 创建模型实例
        model = Transformer(
            config=config,
            device=device,
        )
        # 设置为评估模式
        model.eval()
