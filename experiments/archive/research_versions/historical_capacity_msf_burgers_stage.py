import os, sys, time, torch, numpy as np
PROJECT_ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))
if PROJECT_ROOT not in sys.path: sys.path.insert(0,PROJECT_ROOT)
from src.models.msf_pinn_v1 import MSFPINNV1
from src.equations.burgers import residual, DEFAULT_NU
from experiments.burgers.run_burgers import make_training_data, generate_reference
from src.evaluation.metrics import mse,mae,relative_l2_error,max_absolute_error

torch.set_num_threads(1)
SEED=42; LR=1e-3; STAGE=int(os.environ.get('MSF_STAGE','1000')); TARGET=5000
outdir=os.path.join(PROJECT_ROOT,'results','msf_v1'); os.makedirs(outdir,exist_ok=True)
statefile=os.path.join(outdir,'burgers_msf_v1_training_state.pt')

torch.manual_seed(SEED); np.random.seed(SEED)
model=MSFPINNV1(input_dim=2,branch_hidden_dims=(32,64,96),feature_dim=16,fusion_dim=64)
opt=torch.optim.Adam(model.parameters(),lr=LR)
start_epoch=0; history=[]
if os.path.exists(statefile):
    state=torch.load(statefile,map_location='cpu',weights_only=False)
    model.load_state_dict(state['model_state_dict']); opt.load_state_dict(state['optimizer_state_dict']); start_epoch=state['epoch']; history=state['history']
xt_ic,u_ic,xt_bc,u_bc,xt_pde=make_training_data(SEED,100,200,2000)
end=min(start_epoch+STAGE,TARGET); tic=time.perf_counter()
for ep in range(start_epoch+1,end+1):
    opt.zero_grad(); lic=((model(xt_ic)-u_ic)**2).mean(); lbc=((model(xt_bc)-u_bc)**2).mean(); r=residual(model,xt_pde,nu=DEFAULT_NU); lpde=(r**2).mean(); total=lic+lbc+lpde
    if not torch.isfinite(total): raise FloatingPointError(ep)
    total.backward(); opt.step(); history.append((total.item(),lic.item(),lbc.item(),lpde.item()))
print(f'stage {start_epoch+1}-{end} loss={history[-1][0]:.8e} sec={time.perf_counter()-tic:.3f}',flush=True)
torch.save({'epoch':end,'model_state_dict':model.state_dict(),'optimizer_state_dict':opt.state_dict(),'history':history,'seed':SEED,'lr':LR},statefile)
if end==TARGET:
    x,t,u=generate_reference(nx=256,nt=101,nu=DEFAULT_NU); X,T=np.meshgrid(x,t); xt=torch.tensor(np.column_stack([X.ravel(),T.ravel()]),dtype=torch.float32)
    with torch.no_grad(): pred=model(xt).reshape(len(t),len(x)).cpu(); target=torch.tensor(u,dtype=torch.float32)
    metrics={'mse':mse(pred,target),'mae':mae(pred,target),'relative_l2_error':relative_l2_error(pred,target),'max_absolute_error':max_absolute_error(pred,target)}
    torch.save({'model_state_dict':model.state_dict(),'architecture':model.architecture_info(),'seed':SEED,'epochs':TARGET,'nu':DEFAULT_NU,'metrics':metrics},os.path.join(outdir,'burgers_msf_v1_checkpoint.pt'))
    import csv
    with open(os.path.join(outdir,'burgers_msf_v1_loss.csv'),'w',newline='') as f:
        w=csv.writer(f); w.writerow(['epoch','total_loss','ic_loss','bc_loss','pde_loss']); [w.writerow([i,*v]) for i,v in enumerate(history,1)]
    with open(os.path.join(outdir,'burgers_msf_v1_metrics.csv'),'w',newline='') as f:
        w=csv.writer(f); w.writerow(['metric','value']); [w.writerow([k,v]) for k,v in metrics.items()]
    print('METRICS',metrics,flush=True)
