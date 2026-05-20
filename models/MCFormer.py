import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------
#  Patches
# ------------------------------
class Patches(nn.Module):
    def __init__(self, patch_size: int):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, x):
        # x: [B, C, T, 1]
        B, C, T, _ = x.shape
        num_patches = T // self.patch_size
        x = x[:, :, :num_patches*self.patch_size, :]  # trim
        x = x.reshape(B, num_patches, C*self.patch_size)  # flatten patch
        return x

# ------------------------------
#  Patch Encoder
# ------------------------------
class PatchEncoder(nn.Module):
    def __init__(self, num_patches: int, projection_dim: int, in_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, projection_dim)
        self.pos_embed = nn.Embedding(num_patches, projection_dim)
        nn.init.normal_(self.pos_embed.weight, std=0.02)

    def forward(self, x):
        B, num_patches, _ = x.shape
        pos = torch.arange(num_patches, device=x.device)
        pos = self.pos_embed(pos)[None, :, :]  # [1, num_patches, projection_dim]
        return self.proj(x) + pos

# ------------------------------
# Transformer Block
# ------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_units, attn_dropout=0.05, mlp_dropout=0.05):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=attn_dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_units[0]),
            nn.GELU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(mlp_units[0], mlp_units[1]),
            nn.Dropout(mlp_dropout)
        )
        # Init
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x1 = self.norm1(x)
        attn_out, _ = self.attn(x1, x1, x1)
        x = x + attn_out
        x2 = self.norm2(x)
        x = x + self.mlp(x2)
        return x

# ------------------------------
# ViT Classifier
# ------------------------------
class ViTClassifier_FE(nn.Module):
    def __init__(self, seq_length=128, num_class=11, channels=2, patch_size=16,
                 projection_dim=64, num_heads=4, transformer_layers=6, transformer_units=(128,64),
                 mlp_head_units=(128,), attn_dropout=0.05, mlp_dropout=0.05, head_dropout=0.1):
        super().__init__()
        self.num_patches = seq_length // patch_size
        self.patches = Patches(patch_size)
        self.patch_encoder = PatchEncoder(self.num_patches, projection_dim, in_dim=channels*patch_size)

        self.transformer_layers = nn.ModuleList([
            TransformerBlock(projection_dim, num_heads, transformer_units, attn_dropout, mlp_dropout)
            for _ in range(transformer_layers)
        ])
        self.norm = nn.LayerNorm(projection_dim)
        self.flatten = nn.Flatten()
        self.head_dropout = nn.Dropout(head_dropout)

        # MLP head
        mlp_layers = []
        in_dim = self.num_patches * projection_dim
        for u in mlp_head_units:
            mlp_layers.append(nn.Linear(in_dim, u))
            mlp_layers.append(nn.GELU())
            mlp_layers.append(nn.Dropout(head_dropout))
            in_dim = u
        self.mlp_head = nn.Sequential(*mlp_layers)
        # self.classifier = nn.Linear(in_dim, num_class)

        # Linear init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = x.squeeze(1)
        x = x.unsqueeze(-1)
        # x: [B,C,T,1]
        x = self.patches(x)
        x = self.patch_encoder(x)
        for blk in self.transformer_layers:
            x = blk(x)
        x = self.norm(x)
        x = self.flatten(x)
        x = self.head_dropout(x)
        fea = self.mlp_head(x)
        # logits = self.classifier(x)
        return fea
    
class ViTClassifier(nn.Module):
    def __init__(self, seq_length=128, num_class=11, channels=2, patch_size=16,
                 projection_dim=64, num_heads=4, transformer_layers=6, transformer_units=(128,64),
                 mlp_head_units=(128,), attn_dropout=0.05, mlp_dropout=0.05, head_dropout=0.1):
        super().__init__()
        self.encoder = ViTClassifier_FE(seq_length=seq_length,
                                        num_class=num_class,
                                        channels=channels,
                                        patch_size=patch_size,
                                        projection_dim=projection_dim,
                                        num_heads=num_heads,
                                        transformer_layers=transformer_layers,
                                        transformer_units=transformer_units,
                                        mlp_head_units=mlp_head_units,
                                        attn_dropout=attn_dropout,
                                        mlp_dropout=mlp_dropout,
                                        head_dropout=head_dropout)
        self.classifier = nn.Linear(128, num_class)

    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x

# ------------------------------
# Test
# ------------------------------
if __name__ == '__main__':
    x = torch.randn(1, 1 , 2, 128)
    model = ViTClassifier(seq_length=128, num_class=11)
    y = model(x)
    print(y.shape)  # should be [1,11]