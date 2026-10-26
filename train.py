import argparse, json, random, time
from pathlib import Path
import numpy as np, pandas as pd, torch, yaml
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from data import load_manifest, target_stats, make_transform, FrameDataset, DummyDataset
from models import build_model, freeze_backbone, unfreeze_late_layers, frozen_batchnorm_eval

TARGETS = ["calories","protein","carbohydrate","fat"]
PRED = ["pred_calories","pred_protein","pred_carbohydrate","pred_fat"]

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def metrics(df):
    out={}
    for t,p in zip(TARGETS,PRED):
        out[f"{t}_MAE"]=float(mean_absolute_error(df[t],df[p]))
        out[f"{t}_RMSE"]=float(np.sqrt(mean_squared_error(df[t],df[p])))
        out[f"{t}_R2"]=float(r2_score(df[t],df[p])) if len(df)>1 else float("nan")
    return out

def predict_epoch(model, loader, device, criterion, optimizer=None, scaler=None, amp=False, model_name="resnet18", stage=1):
    training = optimizer is not None
    model.train(training)
    if training: frozen_batchnorm_eval(model, model_name, stage)
    total_loss=0.; n=0; ys=[]; ps=[]; ids=[]; paths=[]
    with (torch.enable_grad() if training else torch.no_grad()):
        for x,y,dish,path in loader:
            x=x.to(device); y=y.to(device)
            if training: optimizer.zero_grad(set_to_none=True)
            enabled=bool(amp and device.type=="cuda")
            with torch.autocast(device_type=device.type, enabled=enabled):
                pred=model(x); loss=criterion(pred,y) # mean MSE over batch and four standardized targets
            if training:
                scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            total_loss += loss.item()*len(x); n += len(x)
            ys.append(y.detach().cpu().numpy()); ps.append(pred.detach().cpu().numpy())
            ids.extend(list(dish)); paths.extend(list(path))
    return total_loss/max(n,1), np.concatenate(ys), np.concatenate(ps), ids, paths

def make_prediction_tables(yz,pz,ids,paths,mean,std):
    yt=yz*std+mean; yp=pz*std+mean
    d={"dish_id":ids,"image_path":paths}
    for i,t in enumerate(TARGETS): d[t]=yt[:,i]; d[PRED[i]]=yp[:,i]
    frames=pd.DataFrame(d)
    agg={t:"first" for t in TARGETS}; agg.update({p:"mean" for p in PRED})
    dishes=frames.groupby("dish_id",as_index=False).agg(agg)
    return frames,dishes

def save_eval(yz,pz,ids,paths,mean,std,out,split):
    frames,dishes=make_prediction_tables(yz,pz,ids,paths,mean,std)
    frames.to_csv(out/f"predictions_{split}_frames.csv",index=False)
    dishes.to_csv(out/f"predictions_{split}_dishes.csv",index=False)
    return {"frame_level":metrics(frames),"dish_level":metrics(dishes)}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",required=True); ap.add_argument("--dummy",action="store_true")
    args=ap.parse_args()
    cfg=yaml.safe_load(open(args.config,encoding="utf-8"))
    seed_all(int(cfg.get("seed",42)))
    mc,dc,tc=cfg["model"],cfg["data"],cfg["training"]
    name=mc["name"].lower(); size=int(mc.get("image_size",224))
    out=Path(cfg["output_dir"]); out.mkdir(parents=True,exist_ok=True)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:",device,"| model:",name,"| dummy:",args.dummy)
    targets=dc.get("target_cols",TARGETS); dish_col=dc.get("dish_id_col","dish_id")
    mean=np.zeros(4,dtype=np.float32); std=np.ones(4,dtype=np.float32)
    if args.dummy:
        tr,va,te=DummyDataset(64,size,1),DummyDataset(24,size,2),DummyDataset(24,size,3)
    else:
        df=load_manifest(dc["manifest"],dc.get("image_root"),dc.get("image_path_col","image_path"),
                         dish_col,dc.get("split_col","split"),targets)
        train_df=df[df["__split"]=="train"].copy(); val_df=df[df["__split"]=="val"].copy(); test_df=df[df["__split"]=="test"].copy()
        if min(len(train_df),len(val_df),len(test_df))==0: raise ValueError("train/val/test must all contain images.")
        mean,std=target_stats(train_df,dish_col,targets)
        aug=cfg.get("augmentation",{})
        tr=FrameDataset(train_df,targets,make_transform(size,True,aug),mean,std,dish_col)
        va=FrameDataset(val_df,targets,make_transform(size,False,aug),mean,std,dish_col)
        te=FrameDataset(test_df,targets,make_transform(size,False,aug),mean,std,dish_col)
    bs=int(dc.get("batch_size",32)); nw=int(dc.get("num_workers",0)); pin=device.type=="cuda"
    tl=DataLoader(tr,batch_size=bs,shuffle=True,num_workers=nw,pin_memory=pin)
    vl=DataLoader(va,batch_size=bs,shuffle=False,num_workers=nw,pin_memory=pin)
    el=DataLoader(te,batch_size=bs,shuffle=False,num_workers=nw,pin_memory=pin)
    model=build_model(name,bool(mc.get("pretrained",True)),mc.get("weights")).to(device)
    criterion=nn.MSELoss(reduction="mean")
    amp=bool(tc.get("use_amp",True)); scaler=torch.amp.GradScaler("cuda",enabled=amp and device.type=="cuda")
    best=float("inf"); rows=[]; patience=int(tc.get("patience",7)); delta=float(tc.get("min_delta",0))
    for stage in (1,2):
        if stage==1:
            freeze_backbone(model,name)
            groups=[{"params":[p for p in model.parameters() if p.requires_grad],"lr":float(tc.get("stage1_head_lr",1e-3))}]
            max_epochs=int(tc.get("stage1_epochs",15))
        else:
            unfreeze_late_layers(model,name)
            if name=="resnet18": bp=list(model.layer4.parameters()); hp=list(model.fc.parameters())
            else: bp=list(model.features[-3:].parameters()); hp=list(model.classifier.parameters())
            groups=[{"params":bp,"lr":float(tc.get("stage2_backbone_lr",1e-5))},
                    {"params":hp,"lr":float(tc.get("stage2_head_lr",1e-4))}]
            max_epochs=int(tc.get("stage2_epochs",30))
        opt=torch.optim.AdamW(groups,weight_decay=float(tc.get("weight_decay",1e-4)))
        sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode="min",factor=.5,patience=2)
        no_improve=0
        for epoch in range(1,max_epochs+1):
            start=time.time()
            tloss,*_=predict_epoch(model,tl,device,criterion,opt,scaler,amp,name,stage)
            vloss,yz,pz,ids,paths=predict_epoch(model,vl,device,criterion,None,scaler,False,name,stage)
            vf,vd=make_prediction_tables(yz,pz,ids,paths,mean,std)
            vmae=float(mean_absolute_error(vd["calories"],vd["pred_calories"]))
            sched.step(vmae)
            row={"stage":stage,"epoch":epoch,"train_mse_standardized":tloss,"val_mse_standardized":vloss,
                 "val_dish_calories_MAE_kcal":vmae,"seconds":time.time()-start}
            row.update({f"val_frame_{k}":v for k,v in metrics(vf).items()})
            row.update({f"val_dish_{k}":v for k,v in metrics(vd).items()})
            rows.append(row)
            print(f"Stage {stage} epoch {epoch}/{max_epochs}: train MSE={tloss:.5f}; val dish calorie MAE={vmae:.3f} kcal")
            if vmae < best-delta:
                best=vmae; no_improve=0
                torch.save({"model_state_dict":model.state_dict(),"stage":stage,"epoch":epoch,
                            "val_dish_calories_MAE_kcal":vmae,"target_mean":mean.tolist(),
                            "target_std":std.tolist(),"config":cfg,"torch_version":str(torch.__version__),
                            "model_name":name},out/f"best_stage{stage}.pt")
            else: no_improve+=1
            if no_improve>=patience:
                print(f"Early stopping stage{stage}."); break
    candidates=[]
    for p in [out/"best_stage1.pt",out/"best_stage2.pt"]:
        if p.exists():
            # Checkpoints are generated locally by this script; metadata requires full loading on PyTorch 2.6+.
            candidates.append(torch.load(p,map_location=device,weights_only=False))
    if not candidates: raise RuntimeError("No checkpoint saved; check validation data.")
    best_ckpt=min(candidates,key=lambda c:c["val_dish_calories_MAE_kcal"])
    model.load_state_dict(best_ckpt["model_state_dict"]); torch.save(best_ckpt,out/"best_model.pt")
    pd.DataFrame(rows).to_csv(out/"metrics.csv",index=False)
    with open(out/"run_config.json","w",encoding="utf-8") as f:
        json.dump({"config":cfg,"target_mean":mean.tolist(),"target_std":std.tolist(),
                   "best_stage":best_ckpt["stage"],"best_epoch":best_ckpt["epoch"],
                   "best_val_dish_calories_MAE_kcal":best_ckpt["val_dish_calories_MAE_kcal"]},f,indent=2)
    _,yz,pz,ids,paths=predict_epoch(model,el,device,criterion,None,scaler,False,name,int(best_ckpt["stage"]))
    final=save_eval(yz,pz,ids,paths,mean,std,out,"test")
    final.update({"best_stage":best_ckpt["stage"],"best_epoch":best_ckpt["epoch"],
                  "selection_metric":"validation dish-level calorie MAE (kcal)"})
    json.dump(final,open(out/"final_metrics.json","w",encoding="utf-8"),indent=2)
    print("Best model:",out/"best_model.pt")
    print("Test metrics (dish-level):",final["dish_level"])
    print("Outputs:",out.resolve())

if __name__=="__main__": main()
