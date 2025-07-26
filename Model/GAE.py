from torch.utils.data import Dataset
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
import time
from tqdm import tqdm
import pandas as pd
import os
import numpy as np
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import wandb
from util import SelfAttentionBlock, ProjectionMLP_g, nt_xent_loss, CombinedDataset

os.environ["WANDB_MODE"] = "offline"

###########
accelerator = Accelerator(
    gradient_accumulation_steps=1,
    log_with="wandb"
)

if accelerator.is_main_process:
    wandb.init(project="Mouse_hippocampus",
               name="clip")  # ,
    # config=vars(args))
device = accelerator.device

dataset2 = torch.load("outdata/image_data.pt", map_location=torch.device('cpu'))
gene_data = torch.tensor(
    [list(map(float, line.strip().split("\t"))) for line in open(r"data/final_gene_data_scale.txt")],
    dtype=torch.float
)  # torch.Size([3000, 8806])
combined_dataset = CombinedDataset(dataset2, gene_data)
train_batch_size = 667
train_dataloader = torch.utils.data.DataLoader(
    combined_dataset,
    shuffle=True,
    batch_size=train_batch_size
)

proj2 = ProjectionMLP_g(gene_data.shape[1], out_dim=dataset2.shape[1])

optimizer = torch.optim.Adam(proj2.parameters(), lr=1e-3)

num_epochs = 10000
steps_per_epoch = math.ceil(len(dataset2) / (train_batch_size * accelerator.num_processes))
max_train_steps = steps_per_epoch * num_epochs

proj2, optimizer, train_dataloader = accelerator.prepare(proj2, optimizer, train_dataloader)

start_epoch = 0
min_loss_val = float('inf')
# train_loss = []
global_step = 1
progress_bar = tqdm(
    range(0, max_train_steps),
    initial=global_step,
    desc="Steps",
    disable=not accelerator.is_local_main_process,
)

for epoch in range(start_epoch, num_epochs):
    progress_bar.set_description(f"Epoch {epoch}")

    proj2.train()

    avg_loss = []
    #avg_loss_clip = []
    #avg_loss_cg = []
    for step, data in enumerate(train_dataloader):
        # dataset2直接用，无编码
        z2,reconstructed = proj2(data['gene_feature'])
        #z2 = proj2(data['gene_feature'])
        loss1 = nt_xent_loss(data["pixel_values"], z2)
        loss2 = F.mse_loss(reconstructed, data['gene_feature'],reduction='mean')
        loss = 0.5*loss1 + 0.5*loss2
        #loss = nt_xent_loss(data["pixel_values"], z2)

        if accelerator.sync_gradients:
            optimizer.zero_grad()
            accelerator.backward(loss)
            torch.nn.utils.clip_grad_norm_(proj2.parameters(), max_norm=1.0)
            optimizer.step()
            avg_loss.append(loss.item())
            #avg_loss_clip.append(loss1.item())
            #avg_loss_cg.append(loss2.item())
            global_step += 1
            progress_bar.update(1)

        if accelerator.is_main_process and global_step % 100 == 0:
            #logs = {"loss": loss.detach().item()}
            progress_bar.set_postfix({
                    'loss': loss.item(),
                    'loss_db': loss1.item(),
                    'loss_cg': loss2.item(),
                    #"lr": lr_scheduler.get_last_lr()[0]

                })
            if loss < min_loss_val:
                min_loss_val = loss
                unwrapped_model = accelerator.unwrap_model(proj2)
                accelerator.save(unwrapped_model, "outdata/best_geneEncoder.pth")
                print(f"Epoch {epoch}: New best model saved with loss {min_loss_val:.4f}")

            wandb.log({
                "loss": loss.item(),
                'loss_db': loss1.item(),
                'loss_cg': loss2.item(),
                "min_loss_val": min_loss_val,
            }, step=global_step)

    # train_loss.append(loss.detach().item())
    # pd.DataFrame(train_loss).to_csv("all_loss.csv", index=False, header=0)

accelerator.end_training()
