import argparse, csv, hashlib, json, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageOps
import torch
from torch import nn
from torchvision.models import resnet18, mobilenet_v2

from analysis_paths import ROOT,MANIFEST,IMAGE_ROOT,MODELS_DIR,PRED_DIR,OUT,ONNX_DIR
TARGETS=['calories','protein','carbs','fat']
MEAN=np.array([.485,.456,.406],np.float32)
STD=np.array([.229,.224,.225],np.float32)
SPECS={'CustomCNN':'cnn_best.pt','ResNet18_finetune':'Resnet_finetune.pt','MobileNetV2_finetune':'MobileNetV2_finetune.pt'}

import sys
sys.path.insert(0,str(ROOT/'src/models/nutrivision_cnn'))
from nutrivision.model import CustomCNN

def load_models(device,ort_backend=False):
    models={}
    for name,file in SPECS.items():
        ck=torch.load(MODELS_DIR/file,map_location='cpu',weights_only=True)
        if name=='CustomCNN':
            model=CustomCNN(ck['config']['dropout']); state=ck['model']; scaler=ck['label_scaler']; mu,sd=scaler['mean'],scaler['std']
        else:
            if name.startswith('ResNet'):
                model=resnet18(weights=None); model.fc=nn.Linear(model.fc.in_features,4)
            else:
                model=mobilenet_v2(weights=None); model.classifier[-1]=nn.Linear(model.classifier[-1].in_features,4)
            state=ck['model_state_dict']; mu,sd=ck['target_mean'],ck['target_std']
        model.load_state_dict(state,strict=True)
        if ort_backend:
            import onnxruntime as ort
            options=ort.SessionOptions();options.intra_op_num_threads=4;options.inter_op_num_threads=1
            model=ort.InferenceSession(str(ONNX_DIR/f'{name}.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
        else:model=model.to(device).eval()
        models[name]=(model,np.asarray(mu,np.float64),np.asarray(sd,np.float64))
    return models

def tensor(images):
    a=np.stack([np.asarray(im,np.float32)/255 for im in images])
    return torch.from_numpy(((a-MEAN)/STD).transpose(0,3,1,2).copy())

def read_image(path):
    with Image.open(IMAGE_ROOT/path) as im:
        return im.convert('RGB').resize((224,224),Image.Resampling.BILINEAR)

def predict(models,images,device):
    x=tensor(images).to(device)
    result={}
    for name,(model,mu,sd) in models.items():
        z=model.run(None,{'images':x.cpu().numpy()})[0] if hasattr(model,'run') else model(x).cpu().numpy()
        result[name]=z.astype(np.float64)*sd+mu
    return result

def metric_rows(df,model,condition):
    rows=[]
    for target in TARGETS:
        y=df['true_'+target]; p=df['pred_'+target]; e=p-y
        var=df.groupby('dish_id')['true_'+target].first().var(ddof=0)
        mae=e.abs().groupby(df.dish_id).mean().mean()
        mse=e.pow(2).groupby(df.dish_id).mean().mean()
        g=df.groupby('dish_id')[['true_'+target,'pred_'+target]].mean(); de=g['pred_'+target]-g['true_'+target]
        for level,ma,msevalue in [('single_frame_dish_balanced',mae,mse),('dish_level',de.abs().mean(),de.pow(2).mean())]:
            rows.append(dict(model=model,condition=condition,target=target,level=level,MAE=float(ma),RMSE=float(np.sqrt(msevalue)),R2=float(1-msevalue/var),n_dishes=df.dish_id.nunique(),n_frames=len(df)))
    return rows

def summarize():
    rows=[]
    for name in SPECS:
        orig=pd.read_csv(PRED_DIR/f'predictions_{name}.csv')
        rows+=metric_rows(orig,name,'Original')
        for cond in ['Grayscale','Saturation_0.3']:
            df=pd.read_csv(OUT/'predictions'/f'{name}_{cond}.csv')
            assert len(df)==20686 and df.dish_id.nunique()==676
            assert not df.duplicated(['dish_id','frame_path']).any()
            assert orig[['dish_id','frame_path']].equals(df[['dish_id','frame_path']])
            for t in TARGETS: assert np.allclose(orig['true_'+t],df['true_'+t],atol=1e-4)
            rows+=metric_rows(df,name,cond)
    m=pd.DataFrame(rows)
    orig=m[m.condition=='Original'].set_index(['model','target','level']).MAE
    m['delta_MAE']=[r.MAE-orig.loc[(r.model,r.target,r.level)] for r in m.itertuples()]
    m['delta_MAE_pct']=[100*r.delta_MAE/orig.loc[(r.model,r.target,r.level)] for r in m.itertuples()]
    m.to_csv(OUT/'color_metrics.csv',index=False)
    cal=m[(m.level=='single_frame_dish_balanced')&(m.target=='calories')]
    cal.to_csv(OUT/'color_calorie_summary.csv',index=False)
    print(cal[['model','condition','MAE','delta_MAE']].round(4).to_string(index=False),flush=True)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--smoke',action='store_true'); ap.add_argument('--batch',type=int,default=32); ap.add_argument('--summarize',action='store_true');ap.add_argument('--ort',action='store_true');ap.add_argument('--resume',action='store_true'); args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if args.summarize: summarize(); return
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    device='mps' if torch.backends.mps.is_available() else 'cpu'
    models=load_models(device,args.ort)
    manifest=pd.read_csv(MANIFEST).rename(columns={'carbohydrate':'carbs'})
    test=manifest[manifest.split=='test'].reset_index(drop=True)
    fields=['dish_id','frame_path']+[v for t in TARGETS for v in ['true_'+t,'pred_'+t]]
    checks=[]
    spot=test.sample(16,random_state=42)
    with torch.inference_mode():
        preds=predict(models,[read_image(p) for p in spot.image_path],device)
    for name,pred in preds.items():
        orig=pd.read_csv(PRED_DIR/f'predictions_{name}.csv').set_index('frame_path')
        expected=orig.loc[spot.image_path,['pred_'+t for t in TARGETS]].to_numpy()
        maxerr=np.max(np.abs(pred-expected),axis=0)
        assert np.allclose(pred,expected,rtol=2e-4,atol=.02),(name,maxerr)
        checks.append(dict(model=name,max_abs_difference=dict(zip(TARGETS,maxerr.tolist()))))
    (OUT/('original_spot_check_smoke.json' if args.smoke else 'original_spot_check.json')).write_text(json.dumps(checks,indent=2))
    print('16-image original check PASSED for all three models; device='+device,flush=True)
    conditions=['Grayscale','Saturation_0.3']
    if args.smoke: test=test.iloc[:64]
    pred_dir=OUT/('smoke_predictions' if args.smoke else 'predictions'); pred_dir.mkdir(exist_ok=True)
    handles={}; writers={};resume=0
    if not args.smoke and not args.resume and any((pred_dir/f'{n}_{c}.csv').exists() for n in SPECS for c in conditions):
        raise FileExistsError('Predictions already exist. Use --resume or set a fresh NV_ANALYSIS_OUT.')
    if args.resume:
        saved={}
        for name in SPECS:
            for cond in conditions:
                f=pred_dir/f'{name}_{cond}.csv'
                saved[name,cond]=pd.read_csv(f) if f.exists() else pd.DataFrame(columns=fields)
        resume=min(len(v) for v in saved.values())//args.batch*args.batch
        for (name,cond),df in saved.items():
            assert df.frame_path.iloc[:resume].tolist()==test.image_path.iloc[:resume].tolist()
            df.iloc[:resume].to_csv(pred_dir/f'{name}_{cond}.csv',index=False)
        print('Resuming from',resume,'frames with',('ONNX Runtime' if args.ort else 'PyTorch'),flush=True)
    for name in SPECS:
        for cond in conditions:
            f=(pred_dir/f'{name}_{cond}.csv').open('a' if args.resume else 'w',newline=''); w=csv.writer(f)
            if not args.resume:w.writerow(fields)
            handles[name,cond]=f; writers[name,cond]=w
    start=time.time()
    try:
        with ThreadPoolExecutor(max_workers=4) as pool, torch.inference_mode():
            for lo in range(resume,len(test),args.batch):
                chunk=test.iloc[lo:lo+args.batch]
                images=list(pool.map(read_image,chunk.image_path))
                for cond in conditions:
                    transformed=[ImageOps.grayscale(im).convert('RGB') if cond=='Grayscale' else ImageEnhance.Color(im).enhance(.3) for im in images]
                    preds=predict(models,transformed,device)
                    truth=chunk[TARGETS].to_numpy()
                    for name,pred in preds.items():
                        assert np.isfinite(pred).all()
                        writers[name,cond].writerows([[row.dish_id,row.image_path]+[v for j in range(4) for v in [truth[i,j],pred[i,j]]] for i,row in enumerate(chunk.itertuples())])
                n=min(lo+args.batch,len(test))
                if lo==resume or n%512==0 or n==len(test):
                    for f in handles.values():f.flush()
                    dt=time.time()-start
                    print(f'PROGRESS {n}/{len(test)} frames, six new predictions/frame, elapsed {dt:.1f}s, remaining {(len(test)-n)/(n-resume)*dt/60:.1f}min',flush=True)
    finally:
        for f in handles.values(): f.close()
    metadata=dict(device=device,backend='PyTorch FP32 then ONNX Runtime FP32' if args.ort and args.resume else ('ONNX Runtime FP32' if args.ort else 'PyTorch FP32'),threads=4,test_frames=len(test),conditions=conditions,saturation_factor=.3,grayscale='PIL ImageOps.grayscale then RGB',preprocessing='RGB resize 224x224 PIL bilinear, perturb, ImageNet normalization',original_source='D supplied predictions; only 16-image original validation rerun',checkpoint_sha256={n:hashlib.sha256((MODELS_DIR/f).read_bytes()).hexdigest() for n,f in SPECS.items()})
    (OUT/('smoke_metadata.json' if args.smoke else 'color_metadata.json')).write_text(json.dumps(metadata,indent=2))
    if not args.smoke: summarize()

if __name__=='__main__': main()
