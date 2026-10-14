"""Exact Final MSF-PINN Architecture reconstructed from the project's earlier scientific report.

No Fourier features and no adaptive weighting. Each branch receives the same raw
coordinate vector and uses a distinct capacity. For Burgers (input_dim=2):
  Branch 1: 2 -> 32 -> 16
  Branch 2: 2 -> 64 -> 16
  Branch 3: 2 -> 96 -> 16
  Fusion: concat(48) -> 64 -> 1
This gives exactly 6,897 trainable parameters for the 2-D Burgers model.
"""
import torch
import torch.nn as nn


class MSFBranch(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, feature_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, feature_dim),
        )
    def forward(self, z):
        return self.net(z)


class MSFPINNV1(nn.Module):
    def __init__(self, input_dim=2, output_dim=1, branch_hidden_dims=(32,64,96), feature_dim=16, fusion_dim=64):
        super().__init__()
        if tuple(branch_hidden_dims) != (32,64,96):
            self.branch_hidden_dims = tuple(int(v) for v in branch_hidden_dims)
        else:
            self.branch_hidden_dims = (32,64,96)
        self.input_dim=input_dim; self.output_dim=output_dim; self.feature_dim=feature_dim; self.fusion_dim=fusion_dim
        self.branches=nn.ModuleList([MSFBranch(input_dim,w,feature_dim) for w in self.branch_hidden_dims])
        self.fusion=nn.Sequential(nn.Linear(3*feature_dim,fusion_dim),nn.Tanh())
        self.output_layer=nn.Linear(fusion_dim,output_dim)
    def forward(self,z):
        features=[b(z) for b in self.branches]
        fused=torch.cat(features,dim=-1)
        return self.output_layer(self.fusion(fused))
    def architecture_info(self):
        return {
            'model_type':'Final MSF-PINN Architecture (exact, no Fourier)',
            'input_dim':self.input_dim,
            'branches':[f'{self.input_dim}->{w}->{self.feature_dim}' for w in self.branch_hidden_dims],
            'fusion':f'{3*self.feature_dim}->{self.fusion_dim}->{self.output_dim}',
            'activation':'Tanh', 'fourier_features':False, 'adaptive_branch_weights':False,
        }
