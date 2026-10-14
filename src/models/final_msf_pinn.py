"""Final MSF-PINN architecture used by the active project.

Raw coordinates are processed by three complementary branches. No Fourier
features are used. The branches differ in width/depth and their feature vectors
are concatenated, then passed through a fusion MLP to produce one solution.
"""
import torch
import torch.nn as nn

class Branch(nn.Module):
    def __init__(self, input_dim, hidden_dim, feature_dim, hidden_layers):
        super().__init__()
        layers=[nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        for _ in range(max(0, hidden_layers-1)):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, feature_dim))
        self.net=nn.Sequential(*layers)
    def forward(self,x): return self.net(x)

class FinalMSFPINN(nn.Module):
    def __init__(self,input_dim=1,output_dim=1,feature_dim=8,fusion_dim=16,
                 branch_hidden_dims=(8,16,24),branch_hidden_layers=(1,2,3)):
        super().__init__()
        self.input_dim=input_dim; self.output_dim=output_dim
        self.feature_dim=feature_dim; self.fusion_dim=fusion_dim
        self.branch_hidden_dims=tuple(branch_hidden_dims)
        self.branch_hidden_layers=tuple(branch_hidden_layers)
        self.low_branch=Branch(input_dim,self.branch_hidden_dims[0],feature_dim,self.branch_hidden_layers[0])
        self.mid_branch=Branch(input_dim,self.branch_hidden_dims[1],feature_dim,self.branch_hidden_layers[1])
        self.high_branch=Branch(input_dim,self.branch_hidden_dims[2],feature_dim,self.branch_hidden_layers[2])
        self.fusion=nn.Sequential(nn.Linear(3*feature_dim,fusion_dim),nn.Tanh(),nn.Linear(fusion_dim,fusion_dim),nn.Tanh())
        self.output_layer=nn.Linear(fusion_dim,output_dim)
        self.model_type='Final MSF-PINN Architecture (No Fourier)'
    def forward(self,z):
        f1=self.low_branch(z); f2=self.mid_branch(z); f3=self.high_branch(z)
        return self.output_layer(self.fusion(torch.cat([f1,f2,f3],dim=-1)))
    def architecture_info(self):
        return {'model_type':self.model_type,'input_representation':'raw coordinates','scales':3,
                'branch_hidden_dims':self.branch_hidden_dims,'branch_hidden_layers':self.branch_hidden_layers,
                'feature_dim':self.feature_dim,'fusion_dim':self.fusion_dim,'activation':'Tanh',
                'fusion':'Concatenation + two-layer Tanh fusion MLP','fourier_features':False,
                'adaptive_scale_weights':False,'residual_fusion':False,'trainable_frequencies':False,
                'initial_frequencies':'Not used (raw-coordinate architecture)'}
    def get_frequencies(self): return torch.empty(0,device=next(self.parameters()).device)
    def get_scale_weights(self): return torch.full((3,),1/3,device=next(self.parameters()).device)
