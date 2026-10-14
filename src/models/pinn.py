"""Compatibility entry point for the active Final MSF-PINN Architecture."""
from .final_msf_pinn import FinalMSFPINN
class PINN(FinalMSFPINN):
    def __init__(self,input_dim=1,hidden_dim=32,hidden_layers=3,output_dim=1,**kwargs):
        super().__init__(input_dim=input_dim,output_dim=output_dim,
                         feature_dim=kwargs.get('feature_dim',8),fusion_dim=kwargs.get('fusion_dim',16),
                         branch_hidden_dims=kwargs.get('branch_hidden_dims',(8,16,24)),
                         branch_hidden_layers=kwargs.get('branch_hidden_layers',(1,2,3)))
