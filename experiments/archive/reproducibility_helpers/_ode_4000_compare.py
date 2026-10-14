import os,sys,time,csv,torch,numpy as np
ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__),'..'));sys.path.insert(0,ROOT);torch.set_num_threads(1)
from src.models.pinn import PINN as Final
from src.models.archive.baseline_pinn import PINN as Base
from src.equations.ode_decay import residual as dr, exact_solution as de
from src.equations.ode_oscillator import residual as orr, exact_solution as oe
from src.evaluation.metrics import mse,mae,relative_l2_error,max_absolute_error
SEED=42;E=4000;LR=1e-3

def run(problem,kind):
 torch.manual_seed(SEED);np.random.seed(SEED)
 model=(Base(input_dim=1,hidden_dim=32,hidden_layers=3,output_dim=1) if kind=='Baseline PINN' else Final(input_dim=1,feature_dim=8,fusion_dim=16,branch_hidden_dims=(8,16,24),branch_hidden_layers=(1,2,3)))
 opt=torch.optim.Adam(model.parameters(),lr=LR)
 if problem=='Decay':
  x=torch.linspace(0,1,100).reshape(-1,1);x0=torch.tensor([[0.]]);u0=torch.tensor([[1.]]);res=dr;exact=de;grid=torch.linspace(0,1,1000).reshape(-1,1);du0=None
 else:
  x=torch.linspace(0,2,100).reshape(-1,1);x0=torch.tensor([[0.]]);u0=torch.tensor([[0.]]);res=orr;exact=oe;grid=torch.linspace(0,2,1000).reshape(-1,1);du0=torch.tensor([[1.]])
 st=time.perf_counter();loss=None
 for ep in range(E):
  opt.zero_grad();lp=(res(model,x)**2).mean();lic=((model(x0)-u0)**2).mean();loss=lp+lic
  if du0 is not None:
   xx=x0.clone().detach().requires_grad_(True); yy=model(xx);dy=torch.autograd.grad(yy,xx,torch.ones_like(yy),create_graph=True)[0];loss=loss+((dy-du0)**2).mean()
  loss.backward();opt.step()
 sec=time.perf_counter()-st
 with torch.no_grad():pred=model(grid);ref=exact(grid)
 return {'problem':problem,'model':kind,'epochs':E,'seed':SEED,'parameters':sum(p.numel() for p in model.parameters()),'training_time_seconds':sec,'final_loss':loss.item(),'mse':mse(pred,ref),'mae':mae(pred,ref),'relative_l2_error':relative_l2_error(pred,ref),'max_absolute_error':max_absolute_error(pred,ref)}
rows=[]
for p in ['Decay','Oscillator']:
 for k in ['Baseline PINN','Final MSF-PINN Architecture']:
  r=run(p,k);print(r,flush=True);rows.append(r)
out=os.path.join(ROOT,'results','final_4000');os.makedirs(out,exist_ok=True)
with open(os.path.join(out,'ode_baseline_vs_final.csv'),'w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
