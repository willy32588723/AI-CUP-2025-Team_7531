import torch
import torch.nn as nn
from tsai.models.InceptionTimePlus import InceptionBlockPlus

class MultiBranchInceptionModel(nn.Module):
    def __init__(self, input_channels=6, stat_feature_dim=30, num_modes=10, mode_emb_dim=64):
        super().__init__()
        self.backbone = InceptionBlockPlus(ni=input_channels, depth=6, nf=64)
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.stat_fc = nn.Sequential(
            nn.Linear(stat_feature_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU())
        self.mode_emb = nn.Embedding(num_modes, mode_emb_dim)
        self.mode_proj = nn.Linear(mode_emb_dim, 64)
        self.fc_shared = nn.Sequential(
            nn.Linear(384, 128), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(128, 64), nn.ReLU())
        self.gender_fc = nn.Linear(64, 1)
        self.hand_fc = nn.Linear(64, 1)
        self.players_fc = nn.Linear(64, 3)
        self.level_fc = nn.Linear(64, 4)

    def forward(self, x_ts, x_stat, mode):
        x_ts = self.pooling(self.backbone(x_ts)).squeeze(-1)
        x_stat = self.stat_fc(x_stat)
        mode_vec = self.mode_proj(self.mode_emb(mode))
        feat = torch.cat([x_ts, x_stat, mode_vec], dim=1)
        x = self.fc_shared(feat)
        return {
            "gender": self.gender_fc(x).squeeze(1),
            "hand": self.hand_fc(x).squeeze(1),
            "players": self.players_fc(x),
            "level": self.level_fc(x),
        }

def smooth_focal_loss(pred, target, alpha=None, gamma=2.0, smoothing=0.1):
    with torch.no_grad():
        num_classes = pred.size(1)
        true_dist = torch.zeros_like(pred)
        true_dist.fill_(smoothing / (num_classes - 1))
        true_dist.scatter_(1, target.data.unsqueeze(1), 1.0 - smoothing)

    log_prob = torch.nn.functional.log_softmax(pred, dim=1)
    ce_loss = -(true_dist * log_prob).sum(dim=1)
    pt = torch.exp(-ce_loss)
    loss = (1 - pt) ** gamma * ce_loss

    if alpha is not None:
        alpha_t = alpha[target]
        loss = loss * alpha_t
        
    return loss.mean()

