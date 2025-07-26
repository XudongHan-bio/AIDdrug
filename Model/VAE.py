import torch.nn as nn
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
#from diffusers.utils import is_wandb_available
from accelerate import Accelerator
import argparse
from tqdm import tqdm
import math
from PIL import Image, ImageDraw
import torch.nn.functional as F
from audtorch.metrics.functional import pearsonr
from transformers import get_scheduler
import pandas as pd
import random
import torch.distributed as dist
from util import Encoder, Decoder, VAE, load_and_preprocess, compute_rowwise_pearsonr_torch,vae_loss_function
import wandb

os.environ["WANDB_MODE"] = "offline"


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--exp_dir",
        type=str,
        default='D:/Multi_omics_LDM/MOSTA/python/imagedata',
        help=("image_dir."),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help=("batch_size."),
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help=("learning_rate."),
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help=("gradient_accumulation_steps."),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help=("num_workers."),
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1000,
        help=("num_train_epochs."),
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=200,
        help=("Number of steps between validations."),
    )
    parser.add_argument(
        "--project_n",
        type=str,
        default=None,
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--project_n2",
        type=str,
        default=None,
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--svae_path",
        type=str,
        default=None,
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )

    args, unknown = parser.parse_known_args()
    return args

def main():
    args = parse_args()

    # 添加 resume=True 和 id 参数确保同一个 Run
    # 定义accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        rng_types=[] 
        # log_with=args.report_to
    )

    if accelerator.is_main_process:
        wandb.init(project=args.project_n,
                   name=args.project_n2,
                   config=vars(args))
    # 数据
    dataset = load_and_preprocess(args.exp_dir).data
    #dataset2 = load_and_preprocess(args.exp_dir).data
    
    #dataset = torch.log1p(dataset)
    #torch_gen = torch.Generator()
    #torch_gen.manual_seed(42)
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    dim = dataset.shape[1]
    model = VAE(input_dim=dim)
    # model = torch.load('model.pth', map_location=torch.device('cpu'),weights_only=False)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    steps_per_epoch = math.ceil(len(dataset) / (args.batch_size * accelerator.num_processes))
    max_train_steps = steps_per_epoch * args.num_train_epochs
    dataset = dataset.to(accelerator.device)

    num_warmup_steps_for_scheduler = int(0.1 * max_train_steps)
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps_for_scheduler,
        num_training_steps=max_train_steps,
    )
    accelerator.print(f"总训练步数: {max_train_steps}, Warmup步数: {num_warmup_steps_for_scheduler}")

    model, optimizer, dataloader, lr_scheduler = accelerator.prepare(model, optimizer, dataloader, lr_scheduler)

    global_step = 1
    first_epoch = 0

    progress_bar = tqdm(
        range(0, max_train_steps),
        initial=global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )
    save_model_path = os.path.join(args.svae_path, "VAEmodel.pth")
    min_loss_val = float('inf')
    min_ps_val = 0
  
    save_step = 0
    
    for epoch in range(first_epoch, args.num_train_epochs):
        # KL退火策略 - 前20%的epoch逐渐增加KL权重
        #anneal_factor = min(0.01, epoch / (0.2 * args.num_train_epochs))
        #model.beta = anneal_factor * model.beta
        #base_beta = int(40.0 / dim)
        #kl_weight = 0.01 * (1 + math.cos(math.pi * epoch / args.num_train_epochs))
        model.train()
        #beta = min(1.0, epoch / kl_anneal_epochs)
        for batch_idx, images in enumerate(dataloader):
            with accelerator.accumulate(model):
                x_recon, mu, logvar = model(images)
                loss, recon_loss, kl_loss = vae_loss_function(x_recon, images, mu, logvar, beta=0.001)
                #x_recon = zr.sample()
                #loss, recon_loss, kl_loss = vae_loss_function(zr,images, mu, logvar)
                #loss = F.mse_loss(zr,images, reduction='sum')

                if accelerator.sync_gradients:
                    optimizer.zero_grad()
                    accelerator.backward(loss)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                    optimizer.step()
                    lr_scheduler.step()
                    global_step += 1
                    progress_bar.update(1)

            if accelerator.sync_gradients and accelerator.is_main_process:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
            
                logs = {
                    "loss": loss.item(),
                    'recon_loss': recon_loss.item(),
                    'kl_loss': kl_loss.item(),
                    'grad_norm': grad_norm.item(),
                }
                wandb.log(logs, step=global_step)

                # if loss < min_loss_val:
                    # min_loss_val = loss.item()
                    # models = accelerator.unwrap_model(model)
                    # torch.save(models, 'model.pth')

                progress_bar.set_postfix({
                    'loss': loss.item(),
                    'recon_loss': recon_loss.item(),
                    'kl_loss': kl_loss.item(),
                    "lr": lr_scheduler.get_last_lr()[0]

                })

                # Validation and logging
                if global_step % args.validation_steps == 0:
                    with torch.no_grad():
                        all_recon, mu, logvar = model(dataset)
                        #zc = all_recon.sample()
                        #val_loss = criterion(all_recon, dataset)
                        val_loss = F.mse_loss(all_recon, dataset, reduction='mean')
                        corr_vector = compute_rowwise_pearsonr_torch(all_recon, dataset)
                        min_ps = corr_vector.min().item()
                        max_ps = corr_vector.max().item()
                        mean_ps = corr_vector.mean().item()

                        if mean_ps > 0.85 and mean_ps > min_ps_val and loss < min_loss_val:
                            min_ps_val = mean_ps
                            min_loss_val = loss.item()
                            models = accelerator.unwrap_model(model)
                            torch.save(models, save_model_path)
                            save_step = global_step
                            

                        wandb.log({
                            "min_ps": min_ps,
                            "max_ps": max_ps,
                            "mean_ps": mean_ps,
                            "validation_loss": val_loss,
                            "save_step": save_step,
                            # "validation/images": log_image
                        }, step=global_step)
    accelerator.end_training()
# 使用示例
if __name__ == "__main__":
    main()
    
