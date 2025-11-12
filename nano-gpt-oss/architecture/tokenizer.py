import tiktoken     # tiktoken是一个用于创建和使用tokenizer的库

def get_tokenizer():
    o200k_base = tiktoken.get_encoding("o200k_base")    # 获取名为"o200k_base"的基础编码器
    tokenizer = tiktoken.Encoding(          # 创建新编码器
        name="o200k_harmony",               # 新编码器的名称
        pat_str=o200k_base._pat_str,        # 使用基础编码器的正则表达式模式
        mergeable_ranks=o200k_base._mergeable_ranks,    # 使用基础编码器的可合并token等级
        special_tokens={
            **o200k_base._special_tokens,           # 继承基础编码器的所有特殊token
            # 自定义特殊token
            "<|startoftext|>": 199998,
            "<|endoftext|>": 199999,
            "<|reserved_200000|>": 200000,
            "<|reserved_200001|>": 200001,
            "<|return|>": 200002,
            "<|constrain|>": 200003,
            "<|reserved_200004|>": 200004,
            "<|channel|>": 200005,
            "<|start|>": 200006,
            "<|end|>": 200007,
            "<|message|>": 200008,
            "<|reserved_200009|>": 200009,
            "<|reserved_200010|>": 200010,
            "<|reserved_200011|>": 200011,
            "<|call|>": 200012,
        } | {
            f"<|reserved_{i}|>": i for i in range(200013, 201088)
        },
    )
    return tokenizer
