import torch
from inference import generate_text
import time,os,gc
import wandb
from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR

from tqdm.notebook import tqdm


def clear_gpu_memory():
    if torch.cuda_is_available():
        torch.cuda.empty_cache()    # 清空PyTorch CUDA缓存分配器中的未占用内存
        torch.cuda.synchronize()    # 等待所有在默认CUDA设备上的流完成任务
    gc.collect()


# 计算模型再给定输入batch和目标batch上的平均交叉熵损失
def calcc(input_batch, target_batch, model, device):
    total_loss = 0
    for i in range(len(input_batch)):
        inp = input_batch[i].to(device, non_blocking=True)
        logits = model(inp)
        del inp
        tgt = target_batch[i].to(device, non_blocking=True)
        loss = torch.nn.functional.cross_entropy(logits, tgt)
        total_loss += loss
        del tgt, logits, loss

    clear_gpu_memory()
    return total_loss / len(input_batch)


# 按照batch计算损失
def calc_loss_batch(input_batch, target_batch, model, device):
    loss = calcc(input_batch,target_batch,model,device)
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.     # 初始化总损失为0，用于累加所有批次的损失值
    if len(data_loader) == 0:       # 判断数据加载器是否为空
        return float("nan")
    elif num_batches is None:       # 如果未指定num_batches参数，则将num_batches设置为数据加载器中的总批次数
        num_batches = len(data_loader)
    else:       # 如果指定了num_batches，则取指定值和数据加载器总批次数中的较小值，避免超出数据范围
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        # 检查当前批次索引是否小于需要计算的批次数
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
            del loss
        else:
            break

    clear_gpu_memory()
    return total_loss / num_batches


# 评估模型
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def train_model(model, train_loader, val_loader, optimizer, scheduler, device, num_epochs, eval_freq, eval_iter, start_context):
    # train_losses存储训练损失值的列表
    # val_losses存储验证损失值的列表
    # track_tokens_seen存储已处理的token数量的列表
    train_losses, val_losses, track_tokens_seen = [], [], []
    # tokens_seen是已处理的token数量
    # global_step是全局训练步数计数器，每处理一个batch数据时都会增加
    tokens_seen, global_step = 0, -1
    # 最佳验证损失值，初始化为正无穷大
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in tqdm(train_loader):
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            wandb.log({"train/step_loss": loss.item(),
                       "tokens_seen": tokens_seen,
                       "epoch": epoch,
                       "lr": optimizer.param_groups[0]["lr"]
                       }, step=global_step)

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch + 1} (Step {global_step:06d}): "f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

            wandb.log({"train/loss": train_loss, "val/loss": val_loss}, step=global_step)

            # 保存结果最好的模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), 'model/gptoss_best.pt')
                # 创建一个名为gptoss-model的Weights & Biases(Artifact)对象，类型为model
                artifact = wandb.Artifact("gptoss-model", type="model")

                # 通过 artifact.add_file() 添加模型文件
                # 并通过 wandb.log_artifact() 将模型上传到Weights & Biases平台进行版本控制和共享
                artifact.add_file("model/goptoss_best.pt")
                wandb.log_artifact(artifact)
                print(f"✅ Saved new best model with val_loss={val_loss:.3f}")

        torch.save(model.state_dict(), "model/gotoss.pt")
        artifact = wandb.Artifact("gptoss-model", type="model")
        artifact.add_file("model/gotoss.pt")
        wandb.log_artifact(artifact, aliases=["latest"])

        torch.save([train_losses, val_losses, tokens_seen], "model/losses.pt")

        txt = generate_text(model, start_context)
        print(txt)
        wandb.log({"generated_text": wandb.Html(txt)}, step=global_step)
        clear_gpu_memory()

    return train_losses, val_losses, track_tokens_seen


def trainer(model, train_loader, val_loader, device):
    learning_rate = 3e-4    # 学习率
    max_iters = 5
    warmup_steps = 100      # 预热100步，在这个阶段学习率会逐步增加
    min_lr = 3e-5           # 设置最小的学习率
    eval_iters = 5           # 设置评估时使用的迭代次数
    eval_freq = 150         # 设置验证频率

    torch.manual_seed(123)      # 固定随机种子
    print(f"Using {device}")

    # 检查是否存在与训练模型文件，如果存在则加载模型权重到当前模型
    if os.path.exists('model/gptpss.pt'):
        model.load_state_dict(torch.load('model/gptpss.pt'))

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1)       # 创建AdamW优化器，用于更新模型参数
    scheduler_warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)          # 创建线性学习率预热调度器，从较小值逐步提升到初始学习率
    schedular_decay = CosineAnnealingLR(optimizer, T_max=max_iters - warmup_steps, eta_min=min_lr)      # 创建余弦退火学习率衰减调度器，预热结束后，学习率按余弦函数形状从峰值衰减到最小值
    scheduler = SequentialLR(optimizer, schedulers=[scheduler_warmup, schedular_decay], milestones=[warmup_steps])      # 创建顺序学习率调度器，将预热和衰减调度器按顺序串联，形成两阶段调度策略
    num_epochs = max_iters      # 设置训练轮数

    wandb.init(
        project="gptoss-VS-gpt2",
        name="gptoss-model",
        group="model-comparison",
        config={
            "learning_rate": learning_rate,
            "max_iters": max_iters,
            "warmup_steps": warmup_steps,
            "min_lr": min_lr,
            "eval_iters": eval_iters,
            "eval_freq": eval_freq,
            "device": device,
            "model_type": "gpt-oss"
        }
    )

    start_time = time.time()
    train_losses, val_losses, tokens_seen = train_model(
        model, train_loader, val_loader, optimizer, scheduler, device,
        num_epochs=num_epochs, eval_freq=eval_freq, eval_iter=eval_iters,
        start_context="a fast driver named Tim went for",
    )

    torch.save(model.state_dict(),"model/gotoss.pt")
    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    wandb.log({"training_time_min": execution_time_minutes})
    wandb.finish()
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")
    return train_losses, val_losses, tokens_seen
