from torch.utils.data import Dataset
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
#from utils import *
import time
from tqdm import tqdm
import pandas as pd
import os
import numpy as np
import sys
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from util import SelfAttentionBlock, ProjectionMLP_g, nt_xent_loss, CombinedDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#vqgan = VQModel.from_pretrained(r"D:\Multi_omics_LDM\project20241030\ladf3\diffusers-main\examples\vqgan\vqgan-output\vqmodel")
loaded_tensor = torch.tensor(
    [list(map(float, line.strip().split("\t"))) for line in
     open(r"data/final_gene_data_scale.txt")],
    dtype=torch.float
)#torch.Size([3000, 8806])
#geneEncoder = TextEncoder(256, vqgan.config.latent_channels*vqgan.config.block_out_channels[0]*vqgan.config.block_out_channels[0]).to(device)
gene_encoder = torch.load(r"outdata/best_geneEncoder.pth",weights_only=False).to(device)
gene_encoder.eval()
with torch.no_grad():
    gene,recons =gene_encoder(loaded_tensor.to(device))
#pd.DataFrame(gene.detach().cpu()).to_csv("gene_f.csv", index=False, header=0)
torch.save(gene,"outdata/gene_f.pt")
#torch.save(gene_data,"gene_data.pt")