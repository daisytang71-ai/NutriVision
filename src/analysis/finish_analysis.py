import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR',str(Path(os.environ.get('NV_ANALYSIS_OUT',Path(__file__).resolve().parents[2]/'results/color_shape'))/'.mpl_cache'))
import json,zipfile,shutil,hashlib
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr,rankdata,pearsonr
from run_color import OUT,SPECS,TARGETS
from analysis_paths import MANIFEST,IMAGE_ROOT,PRED_DIR

FEATURES={'area_fraction':'Food area fraction','aspect_ratio':'Aspect ratio','circularity':'Circularity'}
LABELS={'CustomCNN':'Custom CNN','ResNet18_finetune':'ResNet18 FT','MobileNetV2_finetune':'MobileNetV2 FT'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':200})

def shape():
    dest=OUT/'shape'; f=pd.read_csv(dest/('shape_features_pending_review.csv' if (dest/'shape_features_pending_review.csv').exists() else 'shape_features_reviewed.csv'))
    f['sample_index']=np.arange(1,len(f)+1)
    qa=pd.read_csv(Path(__file__).resolve().parent/'shape_qa_decisions.csv')
    for row in qa.itertuples():
        mask=dest/'masks'/f'{row.dish_id}.png'
        if mask.exists() and hashlib.sha256(mask.read_bytes()).hexdigest()!=row.mask_sha256:
            raise ValueError('Mask differs from reviewed version; review it again: '+row.dish_id)
    f=f.drop(columns=['qa_status','qa_note'],errors='ignore').merge(qa[['dish_id','frame_path','qa_status','qa_note']],on=['dish_id','frame_path'],how='left',validate='one_to_one')
    if f.qa_status.isna().any(): raise ValueError('New sample requires new visual QA decisions')
    f.to_csv(dest/'shape_features_reviewed.csv',index=False)
    joined=[];correlations=[]
    for name in SPECS:
        p=pd.read_csv(PRED_DIR/f'predictions_{name}.csv')
        m=f.merge(p,on=['dish_id','frame_path'],validate='one_to_one');m['model']=name
        for t in TARGETS:m['abs_error_'+t]=(m['pred_'+t]-m['true_'+t]).abs()
        joined.append(m)
        for subset,sub in [('all_automatic_masks',m),('visually_screened_masks',m[m.qa_status=='visually_screened'])]:
            for t in TARGETS:
                for feat in FEATURES:
                    x=sub[feat].to_numpy();y=sub['abs_error_'+t].to_numpy();z=sub['true_'+t].to_numpy()
                    rho,pvalue=spearmanr(x,y)
                    ranks=np.stack([rankdata(x),rankdata(y),rankdata(z)],axis=1)
                    cov=np.column_stack([np.ones(len(z)),ranks[:,2]])
                    xr=ranks[:,0]-cov@np.linalg.lstsq(cov,ranks[:,0],rcond=None)[0]
                    yr=ranks[:,1]-cov@np.linalg.lstsq(cov,ranks[:,1],rcond=None)[0]
                    partial=pearsonr(xr,yr).statistic
                    correlations.append(dict(subset=subset,model=name,target=t,feature=feat,n=len(sub),spearman_rho=rho,unadjusted_p=pvalue,partial_rank_rho_controlling_true_target=partial))
    alljoined=pd.concat(joined,ignore_index=True);alljoined.to_csv(dest/'shape_features_and_errors.csv',index=False)
    corr=pd.DataFrame(correlations)
    corr.to_csv(dest/'shape_error_correlations.csv',index=False)
    valid=alljoined[alljoined.qa_status=='visually_screened'];n=valid[valid.model=='CustomCNN'].dish_id.nunique()
    fig,axes=plt.subplots(3,3,figsize=(11,8.5),constrained_layout=True)
    for ri,name in enumerate(SPECS):
        sub=valid[valid.model==name]
        for ci,(feat,label) in enumerate(FEATURES.items()):
            ax=axes[ri,ci];ax.scatter(sub[feat],sub.abs_error_calories,s=18,alpha=.65,c='#275D87',edgecolors='none')
            r=corr[(corr.subset=='visually_screened_masks')&(corr.model==name)&(corr.target=='calories')&(corr.feature==feat)].iloc[0]
            ax.set_title(f'{LABELS[name]}   rho = {r.spearman_rho:+.2f}',fontsize=10)
            ax.set_xlabel(label)
            if ci==0:ax.set_ylabel('Absolute calorie error (kcal)')
            ax.grid(alpha=.16)
    fig.suptitle(f'Food shape and prediction error  |  {n} visually screened test images\nOne fixed frame per dish; observational associations; approximate automatic masks',fontsize=12)
    fig.savefig(dest/'shape_error_scatter.png');fig.savefig(dest/'shape_error_scatter.pdf');plt.close(fig)
    cal=corr[(corr.subset=='visually_screened_masks')&(corr.target=='calories')]
    print('SHAPE screened',n,'of',len(f),'images; rejected',len(f)-n,flush=True)
    print(cal[['model','feature','spearman_rho','partial_rank_rho_controlling_true_target']].round(3).to_string(index=False),flush=True)
    (dest/'shape_summary.json').write_text(json.dumps(dict(sampled=len(f),screened=n,rejected=len(f)-n,selection='Visual segmentation quality only, before inspecting prediction errors',ground_truth_masks=False,inference='Exploratory associations only; no causal inference, no claim of verified segmentation accuracy; food type, viewpoint, framing and nutrition may confound shape-error associations.'),indent=2))

def color():
    metrics=pd.read_csv(OUT/'color_metrics.csv');rng=np.random.default_rng(42);ci=[]
    for name in SPECS:
        orig=pd.read_csv(PRED_DIR/f'predictions_{name}.csv').set_index(['dish_id','frame_path'])
        for condition in ['Grayscale','Saturation_0.3']:
            df=pd.read_csv(OUT/'predictions'/f'{name}_{condition}.csv').set_index(['dish_id','frame_path'])
            assert set(orig.index)==set(df.index)
            df=df.loc[orig.index]
            for target in TARGETS:
                original_error=(orig['pred_'+target]-orig['true_'+target]).abs()
                perturbed_error=(df['pred_'+target]-df['true_'+target]).abs()
                per_dish=(perturbed_error-original_error).groupby(level=0).mean().to_numpy()
                boot=per_dish[rng.integers(len(per_dish),size=(5000,len(per_dish)))].mean(axis=1)
                low,high=np.quantile(boot,[.025,.975])
                ci.append(dict(model=name,condition=condition,target=target,delta_MAE=per_dish.mean(),ci_low=low,ci_high=high,n_dishes=len(per_dish),bootstrap_replicates=5000))
    pd.DataFrame(ci).to_csv(OUT/'color_delta_paired_bootstrap.csv',index=False)
    cal=metrics[(metrics.level=='single_frame_dish_balanced')&(metrics.target=='calories')]
    fig,ax=plt.subplots(figsize=(9,5),constrained_layout=True);x=np.arange(3);width=.24
    for j,(condition,label,c) in enumerate([('Original','Original','#334E68'),('Grayscale','Grayscale','#B95048'),('Saturation_0.3','Saturation 0.3','#D29B35')]):
        values=[cal[(cal.model==name)&(cal.condition==condition)].MAE.item() for name in SPECS]
        bars=ax.bar(x+(j-1)*width,values,width,label=label,color=c)
        ax.bar_label(bars,fmt='%.2f',padding=3,fontsize=9)
    ax.set_xticks(x,list(LABELS.values()));ax.set_ylim(0,cal.MAE.max()*1.20);ax.set_ylabel('Calorie MAE (kcal)');ax.set_title('Color sensitivity on all 676 test dishes\nSingle-frame dish-balanced evaluation');ax.legend(frameon=False,ncol=3,loc='upper right');ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.savefig(OUT/'color_calorie_MAE.png');fig.savefig(OUT/'color_calorie_MAE.pdf');plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(10,6),constrained_layout=True)
    for ax,t in zip(axes.flat,TARGETS):
        subset=metrics[(metrics.level=='single_frame_dish_balanced')&(metrics.target==t)&(metrics.condition!='Original')]
        vals=np.array([[subset[(subset.model==name)&(subset.condition==c)].delta_MAE.item() for c in ['Grayscale','Saturation_0.3']] for name in SPECS])
        lim=max(abs(vals).max(),.1);im=ax.imshow(vals,cmap='RdBu_r',vmin=-lim,vmax=lim,aspect='auto')
        ax.set_xticks([0,1],['Grayscale','Saturation 0.3']);ax.set_yticks(range(3),list(LABELS.values()));ax.set_title(t.capitalize()+(' (kcal)' if t=='calories' else ' (g)'))
        for (i,j),v in np.ndenumerate(vals):ax.text(j,i,f'{v:+.2f}',ha='center',va='center',color='white' if abs(v)>.6*lim else 'black')
        fig.colorbar(im,ax=ax,shrink=.85)
    fig.suptitle('Change in MAE relative to original images\nPositive values indicate higher error; each target uses its own color scale',fontsize=12)
    fig.savefig(OUT/'color_delta_MAE_heatmap.png');fig.savefig(OUT/'color_delta_MAE_heatmap.pdf');plt.close(fig)
    # One dish, common preprocessing, all three color conditions.
    test=pd.read_csv(MANIFEST);row=test[test.split=='test'].sort_values(['dish_id','image_path']).iloc[0]
    if not (IMAGE_ROOT/row.image_path).is_file():
        print('Original image unavailable; retaining any bundled input-example figure.',flush=True)
        return
    from PIL import Image,ImageOps,ImageEnhance,ImageDraw
    original=Image.open(IMAGE_ROOT/row.image_path).convert('RGB').resize((224,224),Image.Resampling.BILINEAR)
    canvas=Image.new('RGB',(672,254),'white');pen=ImageDraw.Draw(canvas)
    for j,(label,img) in enumerate([('Original',original),('Grayscale',ImageOps.grayscale(original).convert('RGB')),('Saturation 0.3',ImageEnhance.Color(original).enhance(.3))]):canvas.paste(img,(j*224,0));pen.text((j*224+8,233),label,fill='black')
    canvas.save(OUT/'color_input_examples.png')

def report():
    originals=OUT/'original_predictions';originals.mkdir(exist_ok=True)
    for name in SPECS:shutil.copy2(PRED_DIR/f'predictions_{name}.csv',originals/f'predictions_{name}.csv')
    m=pd.read_csv(OUT/'color_metrics.csv');cal=m[(m.level=='single_frame_dish_balanced')&(m.target=='calories')]
    shape_info=json.loads((OUT/'shape'/'shape_summary.json').read_text())
    c=pd.read_csv(OUT/'shape'/'shape_error_correlations.csv');c=c[(c.subset=='visually_screened_masks')&(c.target=='calories')]
    text='NutriVision Color and Food Shape Analysis\n\nCOLOR\nAll 676 test dishes / 20,686 frames. Three fixed checkpoints; no retraining. Original predictions are reused from D. A 16-image check reproduced each model before new inference.\n\n'+cal[['model','condition','MAE','delta_MAE','delta_MAE_pct']].round(4).to_string(index=False)+'\n\nPrimary evaluation: mean of per-dish mean frame absolute errors. Secondary evaluation: average frame predictions per dish before computing error. Calories use kcal; protein/carbs/fat use grams. Four-target, two-level results are in color_metrics.csv. Paired 95% bootstrap intervals resample whole dishes, 5,000 replicates with seed 42; these quantify dish-sampling uncertainty for these fixed models, not training variability.\n\nGrayscale: Pillow grayscale to three RGB channels. Reduced saturation: Pillow ImageEnhance.Color factor 0.3. Both are applied to bilinearly resized 224x224 RGB images before ImageNet normalization. Identical test frames, labels, checkpoints and target scalers across conditions. No prediction clipping.\n\nFOOD SHAPE\n'+f'Random sample: 100 distinct test dishes, seed 42, lexicographically first frame per dish. {shape_info["screened"]} masks passed visual screening; {shape_info["rejected"]} were excluded for segmentation defects before examining errors. This analysis uses one selected frame per dish and is separate from the full-test color metrics.\n\n'+'External food segmentation model: https://huggingface.co/mawiie/food-segmentation-mobilenet (DeepLabV3+ MobileNetV2, FoodSeg103). Input 512x512 and ImageNet normalization; union of non-background argmax classes restored to original image size; connected components smaller than 25 pixels removed. Masks were visually screened, not evaluated against human ground-truth masks.\n\n'+c[['model','feature','n','spearman_rho','partial_rank_rho_controlling_true_target']].round(4).to_string(index=False)+'\n\nFeature definitions: area fraction = food pixels / image pixels; aspect ratio = longer / shorter side of the minimum-area rectangle around all food contours; circularity = 4*pi*food pixel area / squared sum of external contour perimeters. Contour complexity is circularity inverse and is not a separate independent finding. Partial rank correlation controls for the true nutritional target, using residuals of ranked variables.\n\nINTERPRETATION LIMITS\nColor perturbations measure sensitivity under changed inputs, not proof of nutritional reasoning. Shape results are exploratory associations in approximate, visually screened 2D masks. Food class, viewpoint, framing, portion size and mask quality can confound results. Excluding poor masks creates selection bias, especially for pale/dark foods; correlations using all 100 automatic masks are included as a sensitivity check. No causal or population-wide conclusion is justified; reported p-values are unadjusted exploratory values. Full test labels, including extreme calories, are retained for color evaluation. Rotation and flip were not run in this color/food-shape analysis.\n\nReproducibility metadata and checkpoint hashes: color_metadata.json. Shape selection and QA decisions: shape/shape_features_reviewed.csv. All per-frame new predictions are in predictions/. D source: https://github.com/daisytang71-ai/NutriVision\n'
    (OUT/'RESULTS_AND_METHODS.txt').write_text(text)
    with zipfile.ZipFile(OUT.parent/'NutriVision_Color_Shape_Results.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for path in OUT.rglob('*'):
            if path.is_file() and 'smoke_predictions' not in path.parts and not path.name.startswith('smoke_') and path.name!='shape_features_pending_review.csv':z.write(path,path.relative_to(OUT))
    print('PACKAGED',OUT.parent/'NutriVision_Color_Shape_Results.zip',flush=True)

if __name__=='__main__':
    import sys
    if '--shape-only' in sys.argv:shape()
    else:color();report()
