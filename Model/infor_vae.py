import torch.nn as nn
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from util import SelfAttention, Encoder, Decoder, VAE, load_and_preprocess

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = load_and_preprocess(r"data/hvg_matrix.txt").data

vae = torch.load(r"outdata/VAEmodel.pth", map_location=torch.device('cpu'),
                     weights_only=False)
vae.requires_grad_(False)
vae = vae.to(device)
mu, logvar = vae.encoder(dataset.to(device))
#mu, logvar = vae.encoder(dataset.to(device))
image_data = vae.reparameterize(mu, logvar)
torch.save(image_data,"outdata/image_data.pt")

# zr, mu, logvar = vae(dataset.to(device))
# df = pd.DataFrame(zr.detach().cpu().numpy())
# df.to_csv('output.csv', index=False, header=False)