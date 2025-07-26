import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
from accelerate import Accelerator
from tqdm import tqdm
#from PIL import Image, ImageDraw
from transformers import get_scheduler
import sys
import torch.nn as nn
from audtorch.metrics.functional import pearsonr
import math
import os
from util import SelfAttention, Encoder, Decoder, VAE, CondTransformer, CombinedDataset2, load_and_preprocess, compute_rowwise_pearsonr_torch
import wandb
os.environ["WANDB_MODE"] = "offline"

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--project_name",
        type=str,
        default="G2st",
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="p1",
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--VAE_model",
        type=str,
        default="/home4/hnaxudong/Project1/MOSTA/p1/image_AE/model.pth",
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--image_data_dir",
        type=str,
        default='D:/Multi_omics_LDM/MOSTA/python/imagedata',
        help=("image_dir."),
    )
    parser.add_argument(
        "--gene_data_dir",
        type=str,
        default='gene_gcn_f_dj.pt',
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--exp_dir",
        type=str,
        default='D:/Multi_omics_LDM/MOSTA/python/imagedata',
        help=("image_dir."),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=200,
        help=("Number of steps between validations."),
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=64,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
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

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with="wandb"
    )

    if accelerator.is_main_process:
        wandb.init(project=args.project_name,
                   name=args.run_name,
                   config=vars(args))

    device = accelerator.device

    #数据准备
    vae = torch.load(args.VAE_model, map_location=torch.device('cpu'),
                     weights_only=False)
    vae.requires_grad_(False)
    vae = vae.to(device)
    
    original_img = load_and_preprocess(args.exp_dir).data
    #original_img = original_img.to(device)
    dataset = torch.load(args.image_data_dir,map_location=torch.device('cpu')).detach()
    gene_data = torch.load(args.gene_data_dir,map_location=torch.device('cpu')).detach()
    combined_dataset = CombinedDataset2(dataset,gene_data,original_img)

    train_dataloader = torch.utils.data.DataLoader(
        combined_dataset,
        shuffle=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers
    )
    
    

    #模型
    Tmodel = CondTransformer(d_model=gene_data.shape[1],i_dim=dataset.shape[1],)
    criterion = nn.MSELoss()

    # 优化器
    optimizer = torch.optim.AdamW(Tmodel.parameters(), lr=args.learning_rate)
    #学习率
    steps_per_epoch = math.ceil(len(train_dataloader) / (args.train_batch_size * accelerator.num_processes))
    max_train_steps = steps_per_epoch * args.num_train_epochs

    num_warmup_steps_for_scheduler = int(0.1 * max_train_steps)
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps_for_scheduler,
        num_training_steps=max_train_steps,
    )
    Tmodel, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(Tmodel, optimizer, train_dataloader, lr_scheduler)

    #开始循环
    global_step = 1
    first_epoch = 0

    progress_bar = tqdm(
        range(0, max_train_steps),
        initial=global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )
    min_loss_val = float('inf')
    min_ps_val = 0
    save_step = 0
    save_model_path = os.path.join(args.svae_path, "TransModel.pth")

    for epoch in range(first_epoch, args.num_train_epochs):
        Tmodel.train()
        for batch_idx, batch in enumerate(train_dataloader):
            with accelerator.accumulate(Tmodel):
                gene = batch['gene_feature']
                image = batch['pixel_values']
                rigdata = batch['ori_data']

                outputs = Tmodel(gene)
               
                loss1 = criterion(outputs, image)
                rdata = vae.decoder(outputs)
                loss2 = criterion(rdata, rigdata)
                loss = 0.5*loss1 + 0.5*loss2

                if accelerator.sync_gradients:
                    optimizer.zero_grad()
                    accelerator.backward(loss)
                    torch.nn.utils.clip_grad_norm_(Tmodel.parameters(), max_norm=10)
                    optimizer.step()
                    lr_scheduler.step()
                    global_step += 1
                    progress_bar.update(1)

            if accelerator.sync_gradients and accelerator.is_main_process:
                logs = {
                    "loss": loss.item()
                }
                wandb.log(logs, step=global_step)

                #if loss < min_loss_val:
                #    min_loss_val = loss.item()
                #    models = accelerator.unwrap_model(Tmodel)
                #    torch.save(models, 'Tmodel.pth')

                progress_bar.set_postfix({
                    'loss': loss.item(),
                    "lr": lr_scheduler.get_last_lr()[0]

                })

                if global_step % args.validation_steps == 0:
                    #models = accelerator.unwrap_model(Tmodel)
                    #models.eval()
                    with torch.no_grad():
                        #gene_f = gene[0].unsqueeze(0)#torch.Size([1, 8192])
                        #original_img = image[0] #torch.Size([4608])
                        #original_img = original_img.view(-1, dataset.shape[1]) #torch.Size([1, 4096])
                        generated = Tmodel(gene_data.to(device)) #torch.Size([1, 4096])
                        reconstructed_img = vae.decoder(generated) #torch.Size([65303])
                        #original_img = vae.decoder(original_img)  # torch.Size([65303])

                        val_loss = criterion(reconstructed_img, original_img.to(device))
                    corr_vector = compute_rowwise_pearsonr_torch(reconstructed_img, original_img.to(device))
                    min_ps = corr_vector.min().item()
                    max_ps = corr_vector.max().item()
                    mean_ps = corr_vector.mean().item()
                        
                    if mean_ps > 0.85 and mean_ps > min_ps_val and loss < min_loss_val:
                        min_ps_val = mean_ps
                        min_loss_val = loss.item()
                        modelsave = accelerator.unwrap_model(Tmodel)
                        torch.save(modelsave, save_model_path)
                        save_step = global_step

                        # original_img = (original_img + 1) / 2
                        # reconstructed_img = (reconstructed_img + 1) / 2
                        # target_num_elements = 35 * 35
                        # pad_size = target_num_elements - original_img.size(0)
                        # original_img = torch.nn.functional.pad(original_img, (0, pad_size))
                        # original_img = original_img.view(35, 35)
                        #
                        # reconstructed_img = torch.nn.functional.pad(reconstructed_img, (0, pad_size))
                        # reconstructed_img = reconstructed_img.view(35, 35)

                        #log_image = plot_images(original_img, reconstructed_img, global_step)

                    wandb.log({
                        "min_ps": min_ps,
                        "max_ps": max_ps,
                        "mean_ps": mean_ps,
                        "validation_loss": val_loss,
                        "save_step": save_step,
                        #"validation/images": log_image
                    }, step=global_step)
    accelerator.end_training()
# 使用示例
if __name__ == "__main__":
    main()