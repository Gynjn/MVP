import importlib
import functools
import os
import time
import wandb
import torch
from rich import print
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from setup import init_config, init_distributed, init_wandb_and_backup
from training_utils import create_optimizer, create_lr_scheduler, auto_resume_job, print_rank0
from loss import LossComputer
import math
from copy import deepcopy
from collections import OrderedDict

@torch.no_grad()
def update_ema(ema_model, model, decay=0.999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

def remove_module_prefix(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        key = key.replace("_checkpoint_wrapped_module.", "")
        key = key.replace("_orig_mod.", "")
        while key.startswith("module."):
            key = key[len("module."):]
        new_state_dict[key] = value
    return new_state_dict


# Load config and read(override) arguments from CLI
config = init_config()

os.environ["OMP_NUM_THREADS"] = str(config.training.get("num_threads", 1))

# Set up DDP for training/inference and Fix random seed
ddp_info = init_distributed(seed=777)
dist.barrier()

# Set up wandb and backup source code
if ddp_info.is_main_process:
    init_wandb_and_backup(config)
dist.barrier()


# Set up tf32
torch.backends.cuda.matmul.allow_tf32 = config.training.use_tf32
torch.backends.cudnn.allow_tf32 = config.training.use_tf32
amp_dtype_mapping = {
    "fp16": torch.float16, 
    "bf16": torch.bfloat16, 
    "fp32": torch.float32, 
    'tf32': torch.float32
}

# Load dataset
dataset_name = config.training.get("dataset_name", "data.dataset.Dataset")
module, class_name = dataset_name.rsplit(".", 1)
Dataset = importlib.import_module(module).__dict__[class_name]
dataset = Dataset(config)
batch_size_per_gpu = config.training.batch_size_per_gpu

datasampler = DistributedSampler(dataset)
dataloader = DataLoader(
    dataset,
    batch_size=batch_size_per_gpu,
    shuffle=False,
    num_workers=config.training.num_workers,
    persistent_workers=True,
    pin_memory=False,
    drop_last=True,
    prefetch_factor=config.training.prefetch_factor,
    sampler=datasampler,
)
dataloader_iter = iter(dataloader)

total_train_steps = config.training.train_steps
grad_accum_steps = config.training.grad_accum_steps
total_param_update_steps = total_train_steps
total_train_steps = total_train_steps * grad_accum_steps # real train steps when using gradient accumulation
total_batch_size = batch_size_per_gpu * ddp_info.world_size * grad_accum_steps
total_num_epochs = int(total_param_update_steps * total_batch_size / len(dataset))


module, class_name = config.model.class_name.rsplit(".", 1)
MVP = importlib.import_module(module).__dict__[class_name]
model = MVP(config).to(ddp_info.device)
ema = deepcopy(model).to(ddp_info.device)
# ema99 = deepcopy(model).to(ddp_info.device)
requires_grad(ema, False)
# requires_grad(ema99, False)
model = DDP(model, device_ids=[ddp_info.local_rank])


optimizer, optimized_param_dict, all_param_dict = create_optimizer(
    model,
    config.training.weight_decay,
    config.training.lr,
    (config.training.beta1, config.training.beta2),
)
optim_param_list = list(optimized_param_dict.values())


scheduler_type = config.training.get("scheduler_type", "cosine")
lr_scheduler = create_lr_scheduler(
    optimizer,
    total_param_update_steps,
    config.training.warmup,
    scheduler_type=scheduler_type,
)


if config.training.get("resume_ckpt", "") != "":
    ckpt_load_path = config.training.resume_ckpt
else:
    ckpt_load_path = config.training.checkpoint_dir
reset_training_state = config.training.get("reset_training_state", False)
optimizer, lr_scheduler, cur_train_step, cur_param_update_step = auto_resume_job(
    ckpt_load_path,
    model,
    optimizer,
    lr_scheduler,
    reset_training_state,
)


dist.barrier()

start_train_step = cur_train_step
update_ema(ema, model.module, decay=0)
model.train()
ema.eval()
# ema99.eval()

while cur_train_step <= total_train_steps:
    tic = time.time()
    cur_epoch = int(cur_train_step * (total_batch_size / grad_accum_steps) // len(dataset) )
    try:
        data = next(dataloader_iter)
    except StopIteration:
        print(f"Current Rank {ddp_info.local_rank} Ran out of data. Resetting dataloader epoch to {cur_epoch}; might take a while...")
        datasampler.set_epoch(cur_epoch)
        dataloader_iter = iter(dataloader)
        data = next(dataloader_iter)

    batch = {k: v.to(ddp_info.device) for k, v in data.items() if type(v) == torch.Tensor}
    input_data_dict = {key: value[:, :config.data.num_input_frames] for key, value in batch.items()}
    target_data_dict = {key: value[:, -config.data.num_target_frames:] for key, value in batch.items()}

    with torch.autocast(
        enabled=config.training.use_amp,
        device_type="cuda",
        dtype=amp_dtype_mapping[config.training.amp_dtype],
    ):
        ret_dict = model(input_data_dict, target_data_dict)
    update_grads = (cur_train_step + 1) % grad_accum_steps == 0 or cur_train_step == total_train_steps        

    if update_grads:
        with model.no_sync():  # no sync grads for efficiency
            (ret_dict.loss_metrics.loss / grad_accum_steps).backward()
    else:
        (ret_dict.loss_metrics.loss / grad_accum_steps).backward()
    cur_train_step += 1

    skip_optimizer_step = False
    
    if torch.isnan(ret_dict.loss_metrics.loss) or torch.isinf(ret_dict.loss_metrics.loss):
        print(f"NaN or Inf loss detected, skip this iteration")
        skip_optimizer_step = True
        ret_dict.loss_metrics.loss.data = torch.zeros_like(ret_dict.loss_metrics.loss)

    total_grad_norm = 0

    if update_grads:
        # if not skip_optimizer_step:
        #     total_grad_norm = torch.nn.utils.clip_grad_norm_(optim_param_list, max_norm=config.training.grad_clip_norm).item()
            # if not math.isfinite(total_grad_norm):
            #     print(f"WARNING: step {cur_train_step} got non-finite grad norm {total_grad_norm}, skipping optimizer step")
            #     skip_optimizer_step = True
            # elif total_grad_norm > config.training.grad_clip_norm * 5.0:
            #     print(f"WARNING: step {cur_train_step} grad norm too large {total_grad_norm} > {config.training.grad_clip_norm * 5.0}, skipping optimizer step")
            #     skip_optimizer_step = True
        
        if not skip_optimizer_step:
            optimizer.step()
            cur_param_update_step += 1
        optimizer.zero_grad(set_to_none=True)
        lr_scheduler.step()
        if not skip_optimizer_step:
            update_ema(ema, model.module)
            # update_ema(ema99, model.module, 0.99)

    # log and save checkpoint
    if ddp_info.is_main_process:
        loss_dict = {k: float(f"{v.item():.6f}") for k, v in ret_dict.loss_metrics.items()}
        # print in console
        if (cur_train_step % config.training.print_every == 0) or (cur_train_step < 100 + start_train_step):
            print_str = f"[Epoch {int(cur_epoch):>3d}] | Forwad step: {int(cur_train_step):>6d} (Param update step: {int(cur_param_update_step):>6d})"
            print_str += f" | Iter Time: {time.time() - tic:.2f}s | LR: {optimizer.param_groups[0]['lr']:.6f}\n"
            # Add loss values
            for k, v in loss_dict.items():
                print_str += f"{k}: {v:.6f} | "
            print(print_str)

        # log in wandb
        if (cur_train_step % config.training.wandb_log_every == 0) or (
            cur_train_step < 200 + start_train_step
        ):
            log_dict = {
                "iter": cur_train_step, 
                "forward_pass_step": cur_train_step,
                "param_update_step": cur_param_update_step,
                "lr": optimizer.param_groups[0]["lr"],
                "iter_time": time.time() - tic,
                "grad_norm": total_grad_norm,
                "epoch": cur_epoch,
            }
            log_dict.update({"train/" + k: v for k, v in loss_dict.items()})
            wandb.log(
                log_dict,
                step=cur_train_step,
            )

        # save checkpoint
        if (cur_train_step % config.training.checkpoint_every == 0) or (cur_train_step == total_train_steps):
            checkpoint = {
                "model": remove_module_prefix(model.state_dict()),
                "ema": ema.state_dict(),
                # "ema99": ema99.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "fwdbwd_pass_step": cur_train_step,
                "param_update_step": cur_param_update_step,
            }
            os.makedirs(config.training.checkpoint_dir, exist_ok=True)
            ckpt_path = os.path.join(config.training.checkpoint_dir, f"ckpt_{cur_train_step:016}.pt")
            torch.save(checkpoint, ckpt_path)
            print(f"Saved checkpoint at step {cur_train_step} to {os.path.abspath(ckpt_path)}")
        
        

dist.barrier()
dist.destroy_process_group()
