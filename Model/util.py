import torch.nn as nn
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from audtorch.metrics.functional import pearsonr

#VAE################
class SelfAttention(nn.Module):
    def __init__(self, embed_dim, heads=4):
        super(SelfAttention, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        out, _ = self.attn(x, x, x)
        return self.norm(x + out)

class Encoder(nn.Module):
    def __init__(self, input_dim=300):
        super(Encoder, self).__init__()
        
        #定义不同 input_dim 范围对应的 d1 和 latent_dim
        dim_mapping = [
            (1024, (256, 512)),
            (2048, (1024, 512)),
            (4096, (2048, 512)),
            (float('inf'), (4096, 512))  # input_dim > 4096 的情况
        ]
        
        #查找合适的 d1 和 latent_dim
        for threshold, (d1_val, latent_val) in dim_mapping:
            if input_dim < threshold:
                d1 = d1_val
                #latent_dim = latent_val
                break
        
        
        self.fc1 = nn.Linear(input_dim, d1)
        self.bn1 = nn.BatchNorm1d(d1)  # 添加 BatchNorm
        self.attn = SelfAttention(d1)
        self.fc_mu = nn.Linear(d1, 512)
        self.fc_logvar = nn.Linear(d1, 512)

    def forward(self, x):
        h = F.gelu(self.bn1(self.fc1(x)))       # [N, 128]
        h = self.attn(h.unsqueeze(1)).squeeze(1)
        mu = self.fc_mu(h)               # [N, latent_dim]
        logvar = self.fc_logvar(h)
        logvar = torch.clamp(logvar, min=-10, max=10)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, output_dim=300):
        super(Decoder, self).__init__()
        
        dim_mapping = [
            (1024, (256, 512)),
            (2048, (1024, 512)),
            (4096, (2048, 512)),
            (float('inf'), (4096, 512))  # input_dim > 4096 的情况
        ]
        
        #查找合适的 d1 和 latent_dim
        for threshold, (d1_val, latent_val) in dim_mapping:
            if output_dim <= threshold:
                d1 = d1_val
                #latent_dim = latent_val
                break
        
        self.fc1 = nn.Linear(512, d1)
        self.bn1 = nn.BatchNorm1d(d1)
        self.attn = SelfAttention(d1)
        self.fc2 = nn.Linear(d1, output_dim)

    def forward(self, z):
        h = F.gelu(self.bn1(self.fc1(z)))
        h = self.attn(h.unsqueeze(1)).squeeze(1)
        return self.fc2(h)  # 输出可为任意实数，用于重构 scale.data

class VAE(nn.Module):
    def __init__(self, input_dim=300):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim)
        self.decoder = Decoder(input_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar


class load_and_preprocess(Dataset):
    def __init__(self, file_path):
        # Load and preprocess data
        data = np.loadtxt(file_path, delimiter='\t')  # Shape (3000, 65303)
        #data = data * 2 - 1
        self.data = torch.FloatTensor(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]  # Returns a tensor directly

def compute_rowwise_pearsonr_torch(all_recon: torch.Tensor, dataset: torch.Tensor):
    # all_recon, dataset: [N, D]
    correlations = []
    for i in range(all_recon.size(0)):
        # pearsonr expects 1D torch tensors
        corr = pearsonr(all_recon[i], dataset[i])  # returns scalar tensor
        correlations.append(corr)

    return torch.stack(correlations)

def vae_loss_function(recon_x, x, mu, logvar, beta=0.01, kl_clip=1e6):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')/ x.shape[0]
    kl_raw = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kl_loss = torch.sum(kl_raw)/ x.shape[0]
    
    # clip KL loss
    kl_loss =  torch.clamp(kl_loss, max=kl_clip)
    
    pearson_loss = 1 - pearsonr(recon_x, x).mean()

    total_loss = recon_loss + pearson_loss*beta + kl_loss*beta
    return total_loss, recon_loss, kl_loss*beta
 
######################

#clip####################

class SelfAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=8):
        super(SelfAttentionBlock, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x shape: (batch_size, seq_len, embed_dim)
        attn_output, _ = self.attn(x, x, x)  # Self-attention
        x = self.norm1(x + attn_output)
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        return x

class ProjectionMLP_g(nn.Module):
    def __init__(self, in_dim, out_dim=512):
        super(ProjectionMLP_g, self).__init__()
        # 编码器部分
        self.fc1 = nn.Linear(in_dim, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.attn_block = SelfAttentionBlock(embed_dim=1024, num_heads=8)
        self.fc3 = nn.Linear(1024, out_dim)

        # 解码器部分（重构原始 gene_feature）
        self.decoder = nn.Sequential(
            nn.Linear(out_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, 2048),
            nn.GELU(),
            nn.Linear(2048, in_dim)
        )

    def forward(self, x):
        # 编码
        x = F.gelu(self.fc1(x))
        x = F.gelu(self.fc2(x))
        x = x.unsqueeze(1)
        x = self.attn_block(x)
        x = x.squeeze(1)
        z = self.fc3(x)
        recons = self.decoder(z)

        return z,recons  # 返回的是编码向量，decoder在主循环中调用

# 对比损失函数（NT-Xent，SimCLR 风格）
def nt_xent_loss(z1, z2, temperature=0.7):
    batch_size = z1.size(0)
    #z1 = F.normalize(z1, dim=1)
    #z2 = F.normalize(z2, dim=1)

    representations = torch.cat([z1, z2], dim=0)  # [2*batch, dim]
    similarity_matrix = torch.matmul(representations, representations.T)  # [2B, 2B]

    labels = torch.arange(batch_size, device=z1.device)
    labels = torch.cat([labels, labels], dim=0)
    labels = labels.unsqueeze(0) == labels.unsqueeze(1)  # [2B, 2B]
    labels = labels.float()

    # 去除对角元素（self-similarity）
    mask = torch.eye(labels.shape[0], dtype=torch.bool, device=z1.device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)

    labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)

    logits = logits / temperature
    loss = F.cross_entropy(logits, labels)
    return loss

class CombinedDataset(Dataset):
    def __init__(self, data_tensor, gene_data):
        assert len(data_tensor) == len(gene_data), "两个输入张量的样本数必须相同"
        self.data_tensor = data_tensor
        self.gene_data = gene_data

    def __len__(self):
        return len(self.data_tensor)

    def __getitem__(self, idx):
        return {
            "pixel_values": self.data_tensor[idx],   # 512维，直接用，不编码
            "gene_feature": self.gene_data[idx]
        }
#############################

#tran############
class CombinedDataset2(Dataset):
    def __init__(self, data_tensor, gene_data, ori_data):
        assert len(data_tensor) == len(gene_data), "两个输入张量的样本数必须相同"
        self.data_tensor = data_tensor
        self.gene_data = gene_data
        self.ori_data = ori_data

    def __len__(self):
        return len(self.data_tensor)

    def __getitem__(self, idx):
        return {
            "pixel_values": self.data_tensor[idx],   # 512维，直接用，不编码
            "gene_feature": self.gene_data[idx],
            "ori_data": self.ori_data[idx]
        }
        
class CondTransformer(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0,
        #g_dim: int = 2048,
        i_dim: int = 512,
    ):
        super().__init__()
        # 1) 将 G 映射到 d_model 维度的 token
        #self.g_proj = nn.Linear(g_dim, d_model)
        # 2) 用一个可学习的“输出起始 token”初始化 decoder 输入
        #self.start_token = nn.Parameter(torch.randn(1, 1, d_model))
        # 3) Transformer 模块
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # 方便使用 (B, S, D) 输入
        )
        # 4) 最终把 decoder 的输出投影回 I 的维度
        self.i_proj = nn.Linear(d_model, i_dim)

        # 位置编码（可选）
        #self.pos_enc = nn.Parameter(torch.randn(1, 1, d_model))
        #self.output_activation = nn.Softplus()

    def forward(self, G: torch.Tensor):
        """
        G: (B, 4608)  B 行、4608 维的基因特征
        返回: (B, 8192) 对应生成的表达特征
        """
        B = G.size(0)
        # —— Encoder 部分 ——
        # 1) 投影并加上位置编码
        #enc_in = self.g_proj(G).unsqueeze(1)  # (B, 1, d_model)
        enc_in = G.unsqueeze(1)
        #enc_in = enc_in + self.pos_enc        # (B, 1, d_model)

        # —— Decoder 部分 ——
        # decoder 输入是重复的 start_token 序列（这里只用 1 个 token，可扩展为 N 个）
        #dec_in = self.start_token.repeat(B, 1, 1)  # (B, 1, d_model)
        #dec_in = dec_in + self.pos_enc             # (B, 1, d_model)
        dec_in = enc_in

        # Transformer 前向
        # 注意：memory 是 encoder 输出，decoder 对它做 cross-attn
        dec_out = self.transformer(
            src=enc_in,
            tgt=dec_in,
            src_key_padding_mask=None,
            tgt_key_padding_mask=None
        )  # 输出 (B, 1, d_model)

        # 最后投影回 8192 维
        I_pred = self.i_proj(dec_out).squeeze(1)   # (B, 8192)
        #I_pred = self.output_activation(I_pred)
        return I_pred
#############################


# def plot_images(original, reconstructed, step):
#     """Plot original and reconstructed images and return as wandb Image"""
#     # images = []
#     # image2 = original[0, :, :]
#     original = (original + 1) / 2  # 反归一化
#     image2 = (original * 255).to(torch.uint8).cpu().numpy()
#     image2 = Image.fromarray(image2, mode='L')
#     # images.append(image2)
#
#     # image1 = reconstructed[0, :, :]
#     reconstructed = (reconstructed + 1) / 2  # 反归一化
#     image1 = (reconstructed * 255).to(torch.uint8).cpu().numpy()
#     image1 = Image.fromarray(image1, mode='L')
#     # images.append(image1)
#
#     # 创建对比图（可选：横向拼接）
#     width, height = image2.size
#     comparison_img = Image.new('L', (width * 2 + 1, height))
#     comparison_img.paste(image2, (0, 0))
#     comparison_img.paste(image1, (width, 0))
#     draw = ImageDraw.Draw(comparison_img)
#     draw.line([(width, 0), (width, height)], fill=255, width=1)
#
#     return wandb.Image(comparison_img, caption=f"Step {step}: Left-Original, Right-Reconstructed")
