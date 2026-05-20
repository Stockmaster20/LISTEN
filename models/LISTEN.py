import torch
import torch.nn as nn
import torch.nn.functional as F

class SAM(nn.Module):
    """
    Steady
    """
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels // 2, kernel_size=7, padding=3),
            nn.BatchNorm1d(channels // 2),
            nn.GELU(),
            nn.Conv1d(channels // 2, channels, kernel_size=7, padding=3)
        )

    def forward(self, x_minus_s):
        return self.net(x_minus_s)

class TEM(nn.Module):
    """
    Transient
    """
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.threshold = nn.Parameter(torch.full((1, channels, 1), 0.1))

    def forward(self, x_minus_l):
        features = self.conv(x_minus_l)
        sparse_out = torch.sign(features) * F.relu(torch.abs(features) - self.threshold)
        return sparse_out

class SPCPStage(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.l_update = SAM(channels)
        self.s_update = TEM(channels)

    def forward(self, X, L_prev, S_prev):
        # 1. Update L_{k+1} 
        L_k = self.l_update(X - S_prev)
        # 2. Update S_{k+1}
        S_k = self.s_update(X - L_k)
        
        return L_k, S_k

class LISTEN(nn.Module):
    def __init__(self, in_channels=2, num_stages=3, num_classes=11, feature_dim=32):
        super().__init__()
        self.num_stages = num_stages
        
        self.init_conv = nn.Sequential(
            nn.Conv1d(in_channels, feature_dim, kernel_size=7, padding=3),
            nn.BatchNorm1d(feature_dim),
            nn.GELU()
        )
        
        self.stages = nn.ModuleList([SPCPStage(feature_dim) for _ in range(num_stages)])
        self.adaptive_downsample = nn.AdaptiveAvgPool1d(128)
        
        fusion_dim = feature_dim * 3
        
        self.arm = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(fusion_dim, fusion_dim // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(fusion_dim // 4, fusion_dim, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.classifier = nn.Sequential(
            nn.Conv1d(fusion_dim, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):

        X_feat = self.init_conv(x)
        X_feat = self.adaptive_downsample(X_feat)

        L_k = torch.zeros_like(X_feat)
        S_k = torch.zeros_like(X_feat)

        for stage in self.stages:
            L_k, S_k = stage(X_feat, L_k, S_k)

        fused_features = torch.cat([X_feat, L_k, S_k], dim=1)
        attn_weights = self.arm(fused_features)
        fused_features = fused_features * attn_weights

        logits = self.classifier(fused_features)

        return logits, S_k, L_k, X_feat

class SPCPLoss(nn.Module):
    def __init__(self, 
                 gamma=0.3,     # reconstruction weight
                 lambda_s=0.005, # sparsity weight
                 beta_l=0.05    # low-rank weight
                 ):
        super().__init__()

        self.gamma = gamma
        self.lambda_s = lambda_s
        self.beta_l = beta_l

        self.criterion_cls = nn.CrossEntropyLoss()

    def forward(self, logits, targets, S_k, L_k, X_feat):
        # 1. Classification Loss
        loss_cls = self.criterion_cls(logits, targets)

        # 2. Robust Reconstruction
        recon = L_k + S_k
        loss_recon = F.smooth_l1_loss(recon, X_feat)

        # 3. Sparsity
        loss_sparse = torch.mean(torch.abs(S_k))

        # 4. Low-rank proxy
        diff_L = L_k[:, :, 1:] - L_k[:, :, :-1]
        loss_lowrank = torch.mean(diff_L ** 2)

        loss_total = (
            loss_cls
            + self.gamma * loss_recon
            + self.lambda_s * loss_sparse
            + self.beta_l * loss_lowrank
        )

        return loss_total, loss_cls, loss_recon

if __name__ == "__main__":
    import time
    import numpy as np
    
    batch_size = 4
    seq_length = 128
    num_classes = 11
    
    dummy_input = torch.randn(batch_size, 2, seq_length)
    model = LISTEN(in_channels=2, num_stages=3, num_classes=num_classes, feature_dim=32)
    
    logits, S_k, L_k, X_feat = model(dummy_input)
    print("Logits shape:", logits.shape)
    print("S_k shape:", S_k.shape)
    print("L_k shape:", L_k.shape)