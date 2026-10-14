import os,sys,csv,torch,numpy as np
ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__),'..'));sys.path.insert(0,ROOT);torch.set_num_threads(1)
from src.models.archive.baseline_pinn import PINN
from src.equations.burgers import residual,DEFAULT_NU
from experiments.burgers.run_burgers import make_training_data,generate_reference
from src.evaluation.metrics import mse,mae,relative_l2_error,max_absolute_error
SEED=42;EPOCHS=4000
torch.manual_seed(SEED);np.random.seed(SEED);model=PINN(input_dim=2,hidden_dim=32,hidden_layers=3,output_dim=1);opt=torch.optim.Adam(model.parameters(),lr=1e-3)
xt_ic,u_ic,xt_bc,u_bc,xt_pde=make_training_data(SEED,100,200,2000);hist=[]
for ep in range(1,EPOCHS+1):
 opt.zero_grad();lic=((model(xt_ic)-u_ic)**2).mean();lbc=(model(xt_bc)**2).mean();lpde=(residual(model,xt_pde,DEFAULT_NU)**2).mean();loss=lic+lbc+lpde;loss.backward();opt.step();hist.append((ep,loss.item(),lic.item(),lbc.item(),lpde.item()))
 if ep%1000==0:print(ep,loss.item(),flush=True)
x,t,u=generate_reference(nx=256,nt=101,nu=DEFAULT_NU);X,T=np.meshgrid(x,t);xt=torch.tensor(np.column_stack([X.ravel(),T.ravel()]),dtype=torch.float32)
with torch.no_grad():pred=model(xt).reshape(len(t),len(x));ref=torch.tensor(u,dtype=torch.float32)
met={'mse':mse(pred,ref),'mae':mae(pred,ref),'relative_l2_error':relative_l2_error(pred,ref),'max_absolute_error':max_absolute_error(pred,ref)}
out=os.path.join(ROOT,'results','final_4000');os.makedirs(out,exist_ok=True)
with open(os.path.join(out,'baseline_burgers_loss.csv'),'w',newline='') as f:w=csv.writer(f);w.writerow(['epoch','total_loss','ic_loss','bc_loss','pde_loss']);w.writerows(hist)
with open(os.path.join(out,'baseline_burgers_metrics.csv'),'w',newline='') as f:w=csv.writer(f);w.writerow(['metric','value']);[w.writerow([k,v]) for k,v in met.items()]
torch.save({'model_state_dict':model.state_dict(),'epochs':EPOCHS,'seed':SEED,'metrics':met},os.path.join(out,'baseline_burgers_checkpoint.pt'))
print('METRICS',met,flush=True)
