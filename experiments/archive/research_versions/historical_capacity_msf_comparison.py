import os, sys, csv, time, gc
import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.msf_pinn_v1 import MSFPINNV1
from src.equations.ode_decay import residual as decay_residual, exact_solution as decay_exact
from src.equations.ode_oscillator import residual as oscillator_residual, exact_solution as oscillator_exact
from src.equations.burgers import DEFAULT_NU, residual as burgers_residual
from src.evaluation.metrics import mse, mae, relative_l2_error, max_absolute_error
from experiments.burgers.run_burgers import make_training_data, generate_reference

torch.set_num_threads(1)
SEED=42
EPOCHS=int(os.environ.get("PINN_COMPARE_EPOCHS", "5000"))
LR=1e-3


def seed_all():
    torch.manual_seed(SEED); np.random.seed(SEED)

def make_model(input_dim):
    seed_all()
    return MSFPINNV1(input_dim=input_dim, output_dim=1, branch_hidden_dims=(32,64,96), feature_dim=16, fusion_dim=64)

def nparams(m): return sum(p.numel() for p in m.parameters())

def train_ode(model, x_col, x0, u0, residual_fn, du0=None):
    opt=torch.optim.Adam(model.parameters(), lr=LR); hist=[]; start=time.perf_counter()
    for ep in range(1,EPOCHS+1):
        opt.zero_grad(); r=residual_fn(model,x_col); physics=(r**2).mean(); ic=((model(x0)-u0)**2).mean()
        if du0 is not None:
            x0g=x0.clone().detach().requires_grad_(True); y0=model(x0g)
            dy0=torch.autograd.grad(y0,x0g,torch.ones_like(y0),create_graph=True)[0]
            dic=((dy0-du0)**2).mean()
        else: dic=torch.tensor(0.0)
        total=physics+ic+dic
        if not torch.isfinite(total): raise FloatingPointError(f"non-finite at {ep}")
        total.backward(); opt.step(); hist.append(total.item())
    return hist,time.perf_counter()-start

def run_ode(problem):
    model=make_model(1)
    if problem=='decay':
        xcol=torch.linspace(0,1,100).reshape(-1,1); x0=torch.tensor([[0.]],dtype=torch.float32); u0=torch.tensor([[1.]]); du0=None
        residual_fn=decay_residual; exact_fn=decay_exact; xt=torch.linspace(0,1,1000).reshape(-1,1)
    else:
        xcol=torch.linspace(0,2,100).reshape(-1,1); x0=torch.tensor([[0.]],dtype=torch.float32); u0=torch.tensor([[0.]]); du0=torch.tensor([[1.]])
        residual_fn=oscillator_residual; exact_fn=oscillator_exact; xt=torch.linspace(0,2,1000).reshape(-1,1)
    hist,sec=train_ode(model,xcol,x0,u0,residual_fn,du0)
    with torch.no_grad(): pred=model(xt); ref=exact_fn(xt)
    out=dict(problem=problem,model='MSF-PINN v1 (No Fourier)',epochs=EPOCHS,seed=SEED,parameters=nparams(model),training_time_seconds=sec,
             final_loss=hist[-1],mse=mse(pred,ref),mae=mae(pred,ref),relative_l2_error=relative_l2_error(pred,ref),max_absolute_error=max_absolute_error(pred,ref))
    return out,hist,model

def train_burgers(model,data):
    xt_ic,u_ic,xt_bc,u_bc,xt_pde=data; opt=torch.optim.Adam(model.parameters(),lr=LR); hist=[]; start=time.perf_counter()
    for ep in range(1,EPOCHS+1):
        opt.zero_grad(); lic=((model(xt_ic)-u_ic)**2).mean(); lbc=((model(xt_bc)-u_bc)**2).mean(); r=burgers_residual(model,xt_pde,nu=DEFAULT_NU); lpde=(r**2).mean(); total=lic+lbc+lpde
        if not torch.isfinite(total): raise FloatingPointError(f"non-finite loss {ep}")
        total.backward()
        for p in model.parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all(): raise FloatingPointError(f"non-finite grad {ep}")
        opt.step(); hist.append((total.item(),lic.item(),lbc.item(),lpde.item()))
        if ep==1 or ep%1000==0 or ep==EPOCHS: print(f"MSF-v1 Burgers {ep}: {total.item():.6e}",flush=True)
    return hist,time.perf_counter()-start

def run_burgers():
    data=make_training_data(SEED,100,200,2000); ref=generate_reference(nx=256,nt=101,nu=DEFAULT_NU); model=make_model(2)
    hist,sec=train_burgers(model,data)
    x,t,u=ref; X,T=np.meshgrid(x,t); xt=torch.tensor(np.column_stack([X.ravel(),T.ravel()]),dtype=torch.float32)
    with torch.no_grad(): pred=model(xt).reshape(len(t),len(x)).cpu(); target=torch.tensor(u,dtype=torch.float32)
    out=dict(problem='burgers',model='MSF-PINN v1 (No Fourier)',epochs=EPOCHS,seed=SEED,parameters=nparams(model),training_time_seconds=sec,
             final_loss=hist[-1][0],mse=mse(pred,target),mae=mae(pred,target),relative_l2_error=relative_l2_error(pred,target),max_absolute_error=max_absolute_error(pred,target))
    return out,[h[0] for h in hist],model

def main():
    outdir=os.path.join(PROJECT_ROOT,'results','msf_v1'); os.makedirs(outdir,exist_ok=True); rows=[]
    for p in ['decay','oscillator']:
        print('RUN',p,flush=True); row,hist,model=run_ode(p); rows.append(row)
        torch.save({'model_state_dict':model.state_dict(),'architecture':model.architecture_info(),'seed':SEED,'epochs':EPOCHS},os.path.join(outdir,f'{p}_msf_v1_checkpoint.pt'))
        with open(os.path.join(outdir,f'{p}_msf_v1_loss.csv'),'w',newline='') as f: w=csv.writer(f); w.writerow(['epoch','total_loss']); w.writerows(enumerate(hist,1))
        del model; gc.collect()
    print('RUN burgers',flush=True); row,hist,model=run_burgers(); rows.append(row)
    torch.save({'model_state_dict':model.state_dict(),'architecture':model.architecture_info(),'seed':SEED,'epochs':EPOCHS,'nu':DEFAULT_NU},os.path.join(outdir,'burgers_msf_v1_checkpoint.pt'))
    with open(os.path.join(outdir,'burgers_msf_v1_loss.csv'),'w',newline='') as f: w=csv.writer(f); w.writerow(['epoch','total_loss']); w.writerows(enumerate(hist,1))
    fields=list(rows[0].keys())
    with open(os.path.join(outdir,'msf_v1_results.csv'),'w',newline='') as f: w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    for r in rows: print(r,flush=True)

if __name__=='__main__': main()
