import os,sys,time,csv,math
import torch,numpy as np
ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__),'..')); sys.path.insert(0,ROOT)
torch.set_num_threads(1)
from src.models.pinn import PINN
from src.models.archive.baseline_pinn import PINN as Baseline
from src.equations.burgers import residual,DEFAULT_NU
from experiments.burgers.run_burgers import make_training_data,generate_reference
from src.evaluation.metrics import mse,mae,relative_l2_error,max_absolute_error
SEED=42; EPOCHS=4000; LR=1e-3

def train(model):
 torch.manual_seed(SEED); np.random.seed(SEED)
 data=make_training_data(SEED,100,200,2000); xt_ic,u_ic,xt_bc,u_bc,xt_pde=data
 opt=torch.optim.Adam(model.parameters(),lr=LR); hist=[]; st=time.perf_counter()
 for ep in range(1,EPOCHS+1):
  opt.zero_grad(); lic=((model(xt_ic)-u_ic)**2).mean(); lbc=(model(xt_bc)**2).mean(); lpde=(residual(model,xt_pde,DEFAULT_NU)**2).mean(); loss=lic+lbc+lpde
  loss.backward(); opt.step(); hist.append((ep,loss.item(),lic.item(),lbc.item(),lpde.item()))
  if ep%1000==0: print(ep,loss.item(),flush=True)
 sec=time.perf_counter()-st
 x,t,u=generate_reference(nx=256,nt=101,nu=DEFAULT_NU); X,T=np.meshgrid(x,t); xt=torch.tensor(np.column_stack([X.ravel(),T.ravel()]),dtype=torch.float32)
 with torch.no_grad(): pred=model(xt).reshape(len(t),len(x)); ref=torch.tensor(u,dtype=torch.float32)
 metrics={'mse':mse(pred,ref),'mae':mae(pred,ref),'relative_l2_error':relative_l2_error(pred,ref),'max_absolute_error':max_absolute_error(pred,ref)}
 return hist,sec,metrics,model

os.makedirs(os.path.join(ROOT,'results','final_4000'),exist_ok=True)
rows=[]
for name,model in [('Baseline PINN',Baseline(input_dim=2,hidden_dim=32,hidden_layers=3,output_dim=1)),('Final MSF-PINN Architecture',PINN(input_dim=2,feature_dim=8,fusion_dim=16,branch_hidden_dims=(8,16,24),branch_hidden_layers=(1,2,3)) )]:
 torch.manual_seed(SEED); np.random.seed(SEED); print('RUN',name,flush=True)
 hist,sec,met,model=train(model)
 n=sum(p.numel() for p in model.parameters())
 rows.append({'model':name,'epochs':EPOCHS,'seed':SEED,'parameters':n,'training_time_seconds':sec,'final_loss':hist[-1][1],**met})
 stem='baseline' if name.startswith('Baseline') else 'final_msf'
 torch.save({'model_state_dict':model.state_dict(),'epochs':EPOCHS,'seed':SEED,'architecture':getattr(model,'architecture_info',lambda:{'model_type':name})()},os.path.join(ROOT,'results','final_4000',stem+'_burgers_checkpoint.pt'))
 with open(os.path.join(ROOT,'results','final_4000',stem+'_burgers_loss.csv'),'w',newline='') as f: w=csv.writer(f);w.writerow(['epoch','total_loss','ic_loss','bc_loss','pde_loss']);w.writerows(hist)
with open(os.path.join(ROOT,'results','final_4000','burgers_baseline_vs_final.csv'),'w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(rows,flush=True)
