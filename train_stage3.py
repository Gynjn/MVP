import importlib
import os
import time
import wandb
import torch
from rich import print
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from setup import init_config, init_distributed, init_wandb_and_backup
from training_utils import create_optimizer, create_lr_scheduler, auto_resume_job, print_rank0
from copy import deepcopy
from collections import OrderedDict
from data import get_train_data_loader


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

data_loader = get_train_data_loader(
    config,
    num_workers=config.training.num_workers,
    shuffle=True,
    drop_last=True,
    pin_mem=True,
)

total_train_steps = config.training.train_steps
grad_accum_steps = config.training.grad_accum_steps
total_param_update_steps = total_train_steps
total_train_steps = total_train_steps * grad_accum_steps # real train steps when using gradient accumulation


module, class_name = config.model.class_name.rsplit(".", 1)
MVP = importlib.import_module(module).__dict__[class_name]
model = MVP(config).to(ddp_info.device)
ema = deepcopy(model).to(ddp_info.device)
requires_grad(ema, False)
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
remain_steps = total_train_steps - cur_train_step

for epoch in range(1000000):
    if hasattr(data_loader, "dataset") and hasattr(data_loader.dataset, "set_epoch"):
        data_loader.dataset.set_epoch(epoch)
    if hasattr(data_loader, "sampler") and hasattr(data_loader.sampler, "set_epoch"):
        data_loader.sampler.set_epoch(epoch)
    if hasattr(data_loader, "batch_sampler") and hasattr(
        data_loader.batch_sampler, "set_epoch"
    ):
        data_loader.batch_sampler.set_epoch(epoch)

    for data_dict in data_loader:
        if cur_train_step == total_train_steps:
            break
        tic = time.time()
        batch = {k: v.to(ddp_info.device) for k, v in data_dict.items() if type(v) == torch.Tensor}
        input_data_dict = {
            "image": batch["input_image"],
            "fxfycxcy": batch["input_fxfycxcy"],
            "c2w": batch["input_c2w"],
            "image_indices": batch["input_indices"],
        }
        target_data_dict = {
            "image": batch["target_image"],
            "fxfycxcy": batch["target_fxfycxcy"],
            "c2w": batch["target_c2w"],
            "image_indices": batch["target_indices"],
        }
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
            if not skip_optimizer_step:
                optimizer.step()
                cur_param_update_step += 1
            optimizer.zero_grad(set_to_none=True)
            lr_scheduler.step()
            if not skip_optimizer_step:
                update_ema(ema, model.module)

        # log and save checkpoint
        if ddp_info.is_main_process:
            loss_dict = {k: float(f"{v.item():.6f}") for k, v in ret_dict.loss_metrics.items()}
            # print in console
            if (cur_train_step % config.training.print_every == 0) or (cur_train_step < 100 + start_train_step):
                print_str = f"[Epoch {int(epoch):>3d}] | Forwad step: {int(cur_train_step):>6d} (Param update step: {int(cur_param_update_step):>6d})"
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
