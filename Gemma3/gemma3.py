import torch
from datasets import load_dataset
import tiktoken
import os
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm.auto import tqdm
from contextlib import nullcontext
from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR
import matplotlib.pyplot as plt


ds = load_dataset("roneneldan/TinyStories")
enc = tiktoken.get_encoding("gpt2")     # 使用gpt2的分词方法


def compute_rope_params(head_dim, theta_base=10000, context_length=4096, dtype=torch.float32):
    assert head_dim % 2 == 0    # 必须为偶数，下面要两两旋转

    # 计算逆频率
    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2, dtype=dtype)[: head_dim // 2].float() / head_dim))

    # 生成位置索引
    positions = torch.arange(context_length, dtype=dtype)

    # 计算角度,即每个位置要旋转多少度
    angles = positions[:, None] * inv_freq[None, :]     # Shape: (context_length, head_dim // 2)

    # 扩展角度匹配head_dim
    angles = torch.cat([angles, angles], dim=1)     # Shape: (context_length, head_dim)

    # 计算sine和cosine
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos, sin


def apply_rope(x, cos, sin):
    # x: (batch_size, num_heads, seq_len, head_dim)
    batch_size, num_heads, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "Head dimension must be even"

    # 在头数维度将x拆为两部分
    x1 = x[..., : head_dim // 2]
    x2 = x[..., head_dim // 2:]

    # 调整sin和cos的shape
    cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0)    # # Shape: (1, 1, seq_len, head_dim)
    sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0)

    # 应用旋转
    # 相当于将输入x拆成[x1,x2]，然后交换位置变成[-x2, x1]
    # 将他拆成两部分是因为RoPE的基本旋转单元是二维复数平面，让两列向量应用同一个旋转角度
    # x2变为负数并交换位置是为了满足旋转公式
    rotated = torch.cat((-x2, x1), dim=-1)
    x_rotated = (x * cos) + (rotated * sin)

    return x_rotated.to(dtype=x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-6, bias=False):
        super().__init__()
        self.eps = eps
        # Gemma3存储以零为中心的权重，并在forward期间使用(1+权重)
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim)) if bias else None

    def forward(self, x):
        # 计算float32的范数，然后缩放（1+权重）
        input_dtype = x.dtype
        x_f = x.float()
        var = x_f.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x_f * torch.rsqrt(var + self.eps)
        out = x_norm * (1.0 + self.scale.float())

        if self.shift is not None:
            out = out + self.shift.float()

        return out.to(input_dtype)


class GroupedAttention(nn.Module):
    def __init__(self, d_in, num_heads, num_kv_groups, head_dim=None, qk_norm=False, query_pre_attn_scalar=None, dtype=None,):
        super().__init__()
        assert num_heads % num_kv_groups == 0, "num_heads must be divisible by num_kv_groups"

        self.num_heads = num_heads      # 注意力头数
        self.num_kv_groups = num_kv_groups      # KV分组数
        self.group_size = num_heads // num_kv_groups    # 每个KV组中的注意力头数

        if head_dim is None:
            assert d_in % num_heads == 0, "`d_in` must be divisible by `num_heads` if `head_dim` is not set"
            head_dim = d_in // num_heads

        self.head_dim = head_dim    # 注意力头的维度
        self.d_out = head_dim * num_heads

        # 创建QKV的线性变换层
        self.W_query = nn.Linear(d_in, self.d_out, bias=False, dtype=dtype)
        self.W_key = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)
        self.W_value = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)

        # 输出投影层，将多头注意力结果映射回原始维度
        self.out_proj = nn.Linear(self.d_out, d_in, bias=False, dtype=dtype)

        # 可选的归一化操作，是否对QK进行归一化
        if qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=1e-6)
            self.k_norm = RMSNorm(head_dim, eps=1e-6)
        else:
            self.q_norm = self.k_norm = None

        # 设置注意力计算中的缩放因子，用于防止点积值过大导致的梯度问题
        if query_pre_attn_scalar is not None:
            self.scaling = (query_pre_attn_scalar) ** -0.5
        else:
            self.scaling = (head_dim) ** -0.5

    def forward(self, x, mask, cos, sin):
        b, num_tokens, _ = x.shape

        # 应用投影
        queries = self.W_query(x)       # (b, num_tokens, num_heads * head_dim)
        keys = self.W_key(x)            # (b, num_tokens, num_kv_groups * head_dim)
        values = self.W_value(x)        # (b, num_tokens, num_kv_groups * head_dim)

        # reshape
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
        values = values.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)

        # 归一化可选项，是否对QK进行归一化
        if self.q_norm:
            queries = self.q_norm(queries)
        if self.k_norm:
            keys = self.k_norm(keys)

        # Rope
        queries = apply_rope(queries, cos, sin)
        keys = apply_rope(keys, cos, sin)

        # 扩展K和V以适配每组的头数
        keys = keys.repeat_interleave(self.group_size, dim=1)
        values = values.repeat_interleave(self.group_size, dim=1)

        # 缩放
        queries = queries * self.scaling

        # 计算注意力分数
        attn_scores = queries @ keys.transpose(2, 3)
        attn_scores = attn_scores.masked_fill(mask, -torch.inf)
        attn_weights = torch.softmax(attn_scores, dim=-1)

        context = (attn_weights @ values).transpose(1, 2).reshape(b, num_tokens, self.d_out)
        return self.out_proj(context)


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc2 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc3 = nn.Linear(cfg["hidden_dim"], cfg["emb_dim"], dtype=cfg["dtype"], bias=False)

    def forward(self, x):
        # 不同于传统的"线性->激活->线性"结构，而是使用了门控机制（通过两个并行的线性层和逐元素乘积实现），这种设计有助于提高模型的表达能力
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = nn.functional.gelu(x_fc1, approximate="tanh") * x_fc2
        return self.fc3(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: dict, attn_type: dict):
        super().__init__()
        self.attn_type = attn_type      # 保存注意力类型

        # 创建分组注意力
        self.att = GroupedAttention(
            d_in=cfg["emb_dim"],
            num_heads=cfg["n_heads"],
            num_kv_groups=cfg["n_kv_groups"],
            head_dim=cfg["head_dim"],
            qk_norm=cfg["qk_norm"],
            query_pre_attn_scalar=cfg["query_pre_attn_scalar"],
            dtype=cfg["dtype"],
        )

        self.ff = FeedForward(cfg)      # 创建FeedForward
        # 创建四个不同阶段的归一化层
        self.input_layernorm = RMSNorm(cfg["emb_dim"], eps=1e-6)
        self.post_attention_layernorm = RMSNorm(cfg["emb_dim"], eps=1e-6)
        self.pre_feedforward_layernorm = RMSNorm(cfg["emb_dim"], eps=1e-6)
        self.post_feedforward_layernorm = RMSNorm(cfg["emb_dim"], eps=1e-6)

    def forward(self, x, mask_global, mask_local, cos_global, sin_global, cos_local, sin_local):
        shortcut = x
        x = self.input_layernorm(x)
        if self.attn_type == "sliding_attention":       # 判断是否使用滑动窗口注意力
            attn_mask = mask_local
            cos = cos_local
            sin = sin_local
        else:
            attn_mask = mask_global
            cos = cos_global
            sin = sin_global

        x_attn = self.att(x, attn_mask, cos, sin)       # 计算注意力模块输出
        x_attn = self.post_attention_layernorm(x_attn)      # 对输出进行归一化
        x = shortcut + x_attn                           # shortcut

        # shortcut连接
        shortcut = x
        x_ffn = self.pre_feedforward_layernorm(x)
        x_ffn = self.ff(x_ffn)
        x_ffn = self.post_feedforward_layernorm(x_ffn)
        x = shortcut + x_ffn
        return x


class Gemma3Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg["layer_types"] is not None and len(cfg["layer_types"]) == cfg["n_layers"]        # 验证配置中的层数类型与层数量是否匹配

        # 模型主要参数
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])      # 词嵌入层
        self.blocks = nn.ModuleList([       # Transformer块列表
            TransformerBlock(cfg, attn_type) for attn_type in cfg["layer_types"]
        ])
        self.final_norm = RMSNorm(cfg["emb_dim"], eps=1e-6) # 最终的归一化层
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"])    # 输出层
        self.cfg = cfg      # 保存模型配置

        # 为什么滑动窗口注意力和全局注意力需要不一样的theta_base？
        # RoPE使用不同的频率基底 (theta_base) 来控制位置编码的旋转角度：
        # theta_base 越大，位置编码的变化越缓慢
        # theta_base 越小，位置编码的变化越快速
        # 滑动窗口注意力用于处理局部上下文，需要更精细的位置区分能力，较小的theta_base提供更快速的旋转变化，能更好地区分近距离位置
        # 全局注意力用于处理长距离依赖，需要更平滑的位置编码以避免远距离位置的混淆，较大的theta_base提供更缓慢的旋转变化，适合处理长序
        # 这种设计在局部注意力中精确区分相邻位置，在全局注意力中保持长序列的位置稳定性，可以平衡计算效率和位置编码效果

        # 滑动窗口注意力RoPE参数
        cos_local, sin_local = compute_rope_params(
            head_dim=cfg["head_dim"],
            theta_base=cfg["rope_local_base"],
            context_length=cfg["context_length"],
            dtype=torch.float32,
        )

        # 全局注意力RoPE参数
        cos_global, sin_global = compute_rope_params(
            head_dim=cfg["head_dim"],
            theta_base=cfg["rope_base"],
            context_length=cfg["context_length"],
            dtype=torch.float32,
        )

        # 注册为模型缓冲区，会随模型一起移动到指定设备但不会被优化器更新
        self.register_buffer("cos_local", cos_local, persistent=False)
        self.register_buffer("sin_local", sin_local, persistent=False)
        self.register_buffer("cos_global", cos_global, persistent=False)
        self.register_buffer("sin_global", sin_global, persistent=False)

    def _create_masks(self, seq_len, device):
        ones = torch.ones((seq_len, seq_len), dtype=torch.bool, device=device)

        # mask_global (future is masked: j > i)
        #     j:  0 1 2 3 4 5 6 7
        #  i
        #  0:     0 1 1 1 1 1 1 1
        #  1:     0 0 1 1 1 1 1 1
        #  2:     0 0 0 1 1 1 1 1
        #  3:     0 0 0 0 1 1 1 1
        #  4:     0 0 0 0 0 1 1 1
        #  5:     0 0 0 0 0 0 1 1
        #  6:     0 0 0 0 0 0 0 1
        #  7:     0 0 0 0 0 0 0 0
        mask_global = torch.triu(ones, diagonal=1)      # 全局注意力掩码

        # far_past (too far back is masked: i - j >= sliding_window)
        # where sliding_window = 4
        #     j:  0 1 2 3 4 5 6 7
        #  i
        #  0:     0 0 0 0 0 0 0 0
        #  1:     0 0 0 0 0 0 0 0
        #  2:     0 0 0 0 0 0 0 0
        #  3:     0 0 0 0 0 0 0 0
        #  4:     1 0 0 0 0 0 0 0
        #  5:     1 1 0 0 0 0 0 0
        #  6:     1 1 1 0 0 0 0 0
        #  7:     1 1 1 1 0 0 0 0
        far_past = torch.triu(ones, diagonal=self.cfg["sliding_window"]).T

        # Local (sliding_window) = future OR far-past
        # mask_local
        #     j:  0 1 2 3 4 5 6 7
        # i
        # 0:      0 1 1 1 1 1 1 1
        # 1:      0 0 1 1 1 1 1 1
        # 2:      0 0 0 1 1 1 1 1
        # 3:      0 0 0 0 1 1 1 1
        # 4:      1 0 0 0 0 1 1 1
        # 5:      1 1 0 0 0 0 1 1
        # 6:      1 1 1 0 0 0 0 1
        # 7:      1 1 1 1 0 0 0 0
        mask_local = mask_global | far_past     # 滑动窗口注意力掩码
        return mask_global, mask_local

    def forward(self, input_ids, targets=None):
        b, seq_len = input_ids.shape        # 获取输入的批次大小和序列长度
        x = self.tok_emb(input_ids) * (self.cfg["emb_dim"] ** 0.5)      # 通过词嵌入层，并乘以嵌入维度的平方根进行缩放
        mask_global, mask_local = self._create_masks(seq_len, x.device)     # 创建全局和局部掩码

        for block in self.blocks:       # 逐层通过Transformer块，传入相应的掩码和位置编码参数
            x = block(
                x,
                mask_global=mask_global,
                mask_local=mask_local,
                cos_global=self.cos_global,
                sin_global=self.sin_global,
                cos_local=self.cos_local,
                sin_local=self.sin_local,
            )

        x = self.final_norm(x)      # 应用最终归一化
        logits = self.out_head(x.to(self.cfg["dtype"]))     # 通过输出头获得logits
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))        # 计算交叉熵损失
        return logits, loss

    @torch.no_grad()        # 禁用梯度计算，提高推理效率
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
      for _ in range(max_new_tokens):       # 循环生成制定数量的新token
        ctx_len = self.cfg["context_length"]        # 获取模型配置中的上下文长度限制
        idx_cond = idx if idx.size(1) <= ctx_len else idx[:, -ctx_len:]         # 如果输入序列idx的长度小于上下文长度限制，则使用完整的输入序列，否则只使用最后ctx_len个token
        logits, _ = self(idx_cond)  # 通过模型前向传播获得logits输出，因为是生成所以targets为None
        logits = logits[:, -1, :] / temperature     # 取最后一个位置的logits并应用温度缩放，控制生成的随机性
        if top_k is not None:       # 判断是否启用top-k采样，如果启用则只保留概率最大的top-k个预测，将小于top-k的概率设置为负无穷大，从而实现top-k采样
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")
        probs = F.softmax(logits, dim=-1)       # 对logits应用softmax函数得到概率分布
        idx_next = torch.multinomial(probs, num_samples=1)      # 根据概率分布进行采样，得到下一个token，相比于每次都选择概率最高的词，这种方法可以产生更加自然的文本
        idx = torch.cat((idx, idx_next), dim=1)     # 将新生成的token拼接到原有序列末尾
      return idx


def process(example):
    ids = enc.encode_ordinary(example['text'])  # 将普通文本转换成不带任何特殊标记的token ID序列
    out = {'ids': ids, 'len': len(ids)}
    return out


# 就从磁盘上的 .bin 文件里,随机选择batch_size段长度为block_size的连续token序列，做成(x, y)对
def get_batch(split):
    if split == 'train':
        data = np.memmap('train.bin', dtype=np.uint16, mode='r')
    else:
        data = np.memmap('validation.bin', dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])      # 输入
    y = torch.stack([torch.from_numpy(data[i+1:i+block_size+1].astype(np.int64)) for i in ix])      # target
    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


# 计算损失
def estimate_loss(model):
    out = {}
    model.eval()        # 设置为评估模式
    with torch.inference_mode():    # 使用torch.inference_mode()上下文管理器，禁用梯度计算以提高推理效率并节省内存
        for split in ['train', 'val']:      # 遍历训练集和验证集
            losses = torch.zeros(eval_iters)        # 创建一个大小为eval_iters的零张量，用于存储多次评估的损失值
            for k in range(eval_iters):
                X, Y = get_batch(split)     # 加载数据
                with ctx:       # 在自动混合精度上下文ctx中进行前向传播，将每次计算得到的损失值存储在losses数组中
                    logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()      # 计算数据集上多次评估的平均损失，并存储在结果字典中
    model.train()
    return out


GEMMA3_CONFIG_270M = {
    # 基础模型参数
    "vocab_size": 50257,        # 词汇表大小
    "context_length": 32768,    # 模型支持的最大上下文长度
    "emb_dim": 640,             # 嵌入维度
    # 注意力机制参数
    "n_heads": 4,               # 注意力头的数量
    "n_layers": 18,             # Transformer层数量
    "hidden_dim": 2048,         # 前馈网络的隐藏层维度
    "head_dim": 256,            # 每个注意力头的维度
    "qk_norm": True,            # 是否对QK进行归一化
    "n_kv_groups": 1,           # 键值对分组数量，用于分组查询注意
    # 位置编码参数
    "rope_local_base": 10_000.0,        # 滑动窗口注意力使用的RoPE基础频率
    "rope_base": 1_000_000.0,           # 全局注意力使用的RoPE基础频率
    "sliding_window": 512,              # 滑动窗口大小
    # 注意力层配置
    "layer_types": [
        "sliding_attention",    # 滑动窗口注意力机制
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",       # 全局注意力机制
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention"
    ],
    # 训练参数
    "dtype": torch.bfloat16,    # 使用bfloat16数据类型进行训练
    "query_pre_attn_scalar": 256,   # 注意力计算中的查询缩放因子
}

torch.manual_seed(123)
model = Gemma3Model(GEMMA3_CONFIG_270M)

# 超参数
learning_rate = 1e-4    # more stable training, earlier 1e-4
max_iters = 150000      # increase from 25000
warmup_steps = 1000     # smoother initial train, earlier 100
min_lr = 5e-4           # lower rate, earlier 5e-4
eval_iters = 500        # increased from 100
batch_size = 32         # changed from 16, better gradient estimate
block_size = 128        #changed from 64, capture longer range dependencies

gradient_accumulation_steps = 32    # reduced from 50

device = "cuda" if torch.cuda.is_available() else "cpu"
device_type = 'cuda' if 'cuda' in device else 'cpu'     # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler

# How to use autocast https://wandb.ai/wandb_fc/tips/reports/How-To-Use-Autocast-in-PyTorch--VmlldzoyMTk4NTky
# dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'   # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]

ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

torch.set_default_device(device)
torch.manual_seed(42)


if __name__ == '__main__':
    if not os.path.exists("train.bin"):
        tokenized = ds.map(
            process,        # 自定义处理函数
            remove_columns=['text'],    # 删除原始列，只保留返回的新列
            desc="tokenizing the splits",   # 进度条显示
            num_proc=8,     # 使用8个进程并行处理
            )
        # 按split把token ID拼成一条长流，通过内存映射以16-bit整数高效写入.bin文件，供后续训练
        for split, dset in tokenized.items():
            arr_len = np.sum(dset['len'], dtype=np.uint64)  # 计算总token量
            filename = f'{split}.bin'   # 保存的文件名
            dtype = np.uint16   # GPT-2 词表最大索引 50256 < 65536，故用 16 bit 无符号整型即可节省空间
            arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))     # 创建内存映射数组
            total_batches = 1024        # 把数据拆成 1024 个连续子块逐步写入，兼顾速度与内存

            idx = 0
            for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
                # Batch together samples for faster write
                batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
                arr_batch = np.concatenate(batch['ids'])        # 把样本的token ID列表拼成一条长一维数组
                # Write into mmap
                arr[idx: idx + len(arr_batch)] = arr_batch
                idx += len(arr_batch)
            arr.flush()

    # 创建AdamW优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.1,
                                  eps=1e-9)  # weight decay for regularization

    # 创建线性学习率预热调度器，在训练开始时逐步将学习率从0线性增加到初始学习率
    scheduler_warmup = LinearLR(optimizer, total_iters=warmup_steps)
    # 创建余弦退火学习率衰减调度器，在预热结束后使用余弦函数逐渐降低学习率
    scheduler_decay = CosineAnnealingLR(optimizer, T_max=max_iters - warmup_steps, eta_min=min_lr)
    # 创建顺序学习率调度器，组合预热和衰减两个阶段，按顺序使用两个调度器
    scheduler = SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_decay],
                             milestones=[warmup_steps])

    # https://stackoverflow.com/questions/72534859/is-gradscaler-necessary-with-mixed-precision-training-with-pytorch
    # 创建梯度缩放器用于混合精度训练，仅在使用float16数据类型时启用，通过缩放梯度来避免在低精度训练中的梯度下溢问题
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

    # 预训练
    best_val_loss = float('inf')        # 初始化损失为无穷大
    best_model_params_path = "best_model_params.pt"     # 设置保存路径
    train_loss_list, validation_loss_list = [], []      # 创建空列表用于记录训练和验证损失历史

    # 确保模型被移动到正确的计算设备
    model = model.to(device)

    # 循环训练
    for epoch in tqdm(range(max_iters)):
        if epoch % eval_iters == 0 and epoch != 0:      # 每隔eval_iters次迭代进行一次模型评估
            losses = estimate_loss(model)
            print(f"Epoch {epoch}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            print(f"The current learning rate: {optimizer.param_groups[0]['lr']:.5f}")
            train_loss_list += [losses['train']]        # 将损失值添加到历史记录列表中
            validation_loss_list += [losses['val']]

            # 如果当前验证损失优于最佳验证损失，则保存模型参数
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                torch.save(model.state_dict(), best_model_params_path)

        # 从训练数据中获取一个批次的数据
        X, y = get_batch("train")
        X, y = X.to(device), y.to(device)

        with ctx:   # 自动混合精度上下文
            logits, loss = model(X, y)      # 在自动混合精度上下文 ctx 中进行前向传播
            # 由于显存限制，无法使用较大的 batch_size，通过梯度累积模拟大批次训练效果
            loss = loss / gradient_accumulation_steps       # 计算损失并进行梯度累积（将损失除以累积步数）
            scaler.scale(loss).backward()       # 使用梯度缩放器缩放损失并执行反向传播

        if ((epoch + 1) % gradient_accumulation_steps == 0) or (epoch + 1 == max_iters):    # 当达到梯度累积步数或训练结束时，执行参数更新
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)    # 进行梯度裁剪以防止梯度爆炸
            scaler.step(optimizer)      # 使用梯度缩放器更新优化器参数，然后调用优化器的step()方法更新模型参数
            scaler.update()     # 根据训练过程中的梯度情况动态调整缩放因子，以确保在后续训练中能够正确地进行梯度缩放
            optimizer.zero_grad(set_to_none=True)       # 清空梯度并更新缩放器状态
        scheduler.step()        # 更新学习率调度器，调整当前学习率

    # 画出loss图
    train_loss_list_converted = [i.cpu().detach() for i in train_loss_list]
    validation_loss_list_converted = [i.cpu().detach() for i in validation_loss_list]

    plt.plot(train_loss_list_converted, 'g', label='train_loss')
    plt.plot(validation_loss_list_converted, 'r', label='validation_loss')
    plt.xlabel("Steps - Every 100 epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    # 推理
    model = Gemma3Model(GEMMA3_CONFIG_270M)  # 使用配置创建模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    best_model_params_path = "best_model_params.pt"
    model.load_state_dict(torch.load(best_model_params_path, map_location=torch.device(device)))  # 加载最好的模型参数
    sentence = "Once upon a time there was a pumpkin."
    context = (torch.tensor(enc.encode_ordinary(sentence)).unsqueeze(dim=0))
    y = model.generate(context, 200)
    print(enc.decode(y.squeeze().tolist()))
