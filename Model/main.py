import torch
import torch.nn as nn
import math
from torch.utils.data import Dataset

class GenePositionEncoding(nn.Module):
    def __init__(self, num_genes, d_model):
        super().__init__()
        self.encoding = self.get_positional_encoding(num_genes, d_model)

    def get_positional_encoding(self, num_positions, d_model):
        pe = torch.zeros(num_positions, d_model)
        position = torch.arange(0, num_positions, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # shape: [1, num_positions, d_model]

    def forward(self, x):
        # x: [B, num_positions, d_model]
        return x + self.encoding.to(x.device)

class GeneTransformerEncoder(nn.Module):
    def __init__(self, input_dim=1, d_model=512, num_genes=9262, nhead=8, num_layers=4, dropout=0.1):
        super().__init__()
        self.input_fc = nn.Linear(input_dim, d_model)  # [9262, 1] -> [9262, 512]
        self.pos_encoder = GenePositionEncoding(num_genes, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        """
        x: [B, 9262, 1]
        """
        x = self.input_fc(x)  # [B, 9262, 512]
        x = self.pos_encoder(x)
        out = self.transformer(x)  # [B, 9262, 512]
        return out

class SelfAttentionFusion(nn.Module):
    def __init__(self, embed_dim=1024, num_heads=8, dropout=0.1):
        super(SelfAttentionFusion, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, x):
        # x: [batch_size, 9262, 1024]
        attn_output, _ = self.attn(x, x, x)
        x = self.norm(x + attn_output)
        x = self.norm(x + self.ffn(x))  # Feedforward + Residual + Norm
        return x

class LabeledDrugDiseaseDataset(Dataset):
    def __init__(self, drug_tensor, dis_tensor, label_tensor):
        """
        drug_tensor: torch.FloatTensor [9262, 196]
        dis_tensor: torch.FloatTensor [9262, 196]
        label_tensor: torch.FloatTensor [196], 每对的标签，0或1
        """
        assert drug_tensor.shape == dis_tensor.shape, "drug 和 dis 尺寸必须一致"
        assert drug_tensor.shape[1] == label_tensor.shape[0], "标签数量必须等于样本数"

        self.drug = drug_tensor       # [9262, 196]
        self.dis = dis_tensor         # [9262, 196]
        self.labels = label_tensor    # [196]

    def __len__(self):
        return self.drug.shape[1]  # 一共有196对样本

    def __getitem__(self, idx):
        drug_sample = self.drug[:, idx].unsqueeze(1)      # [9262, 1]
        dis_sample = self.dis[:, idx].unsqueeze(1)        # [9262, 1]
        label = self.labels[idx]                          # scalar

        return {
            'drug': drug_sample,       # [9262, 1]
            'disease': dis_sample,     # [9262, 1]
            'label': label,            # float, 0. or 1.
            'index': idx
        }



def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 你原来的加载方式
    drug = torch.tensor(
        [list(map(float, line.strip().split("\t"))) for line in open(r"G:\Tnet_p\R_proj\r1\outdata\drug_matrix.txt")],
        dtype=torch.float
    ).to(device)

    dis = torch.tensor(
        [list(map(float, line.strip().split("\t"))) for line in open(r"G:\Tnet_p\R_proj\r1\outdata\dis_matrix.txt")],
        dtype=torch.float
    ).to(device)

    label = torch.tensor(
        [float(line.strip()) for line in open(r"G:\Tnet_p\R_proj\r1\outdata\label_vector.txt")],
        dtype=torch.float
    ).to(device)

    # 创建 Dataset & DataLoader
    from torch.utils.data import DataLoader

    dataset = LabeledDrugDiseaseDataset(drug, dis, label)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # 测试取一个 batch
    for batch in dataloader:
        drug_input = batch['drug']  # [8, 9262, 1]
        dis_input = batch['disease']  # [8, 9262, 1]
        labels = batch['label']  # [8]
        print(drug_input.shape, labels)
        break

    fused_input = torch.cat([drug_feat, dis_feat], dim=-1)



if __name__ == "__main__":
    main()