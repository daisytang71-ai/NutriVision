import argparse,json
from pathlib import Path
import cv2,numpy as np,pandas as pd,torch
from PIL import Image,ImageOps,ImageDraw
import segmentation_models_pytorch as smp
from run_color import OUT,MEAN,STD,SPECS
from analysis_paths import MANIFEST,IMAGE_ROOT,SEGMENTATION_CHECKPOINT

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--count',type=int,default=100);a=ap.parse_args()
    torch.set_num_threads(2)
    d=pd.read_csv(MANIFEST)
    sample=d[d.split=='test'].sort_values(['dish_id','image_path']).drop_duplicates('dish_id').sample(100,random_state=42).iloc[:a.count]
    dest=OUT/'shape';(dest/'masks').mkdir(parents=True,exist_ok=True);(dest/'overlays').mkdir(exist_ok=True)
    model=smp.DeepLabV3Plus(encoder_name='mobilenet_v2',encoder_weights=None,in_channels=3,classes=104)
    model.load_state_dict(torch.load(SEGMENTATION_CHECKPOINT,map_location='cpu',weights_only=True));model.eval()
    rows=[];cards=[]
    with torch.inference_mode():
        for i,row in enumerate(sample.itertuples()):
            original=Image.open(IMAGE_ROOT/row.image_path).convert('RGB')
            im=original.resize((512,512),Image.Resampling.BILINEAR)
            x=torch.from_numpy(((np.asarray(im,np.float32)/255-MEAN)/STD).transpose(2,0,1).copy())[None]
            labels=model(x).argmax(1)[0].numpy().astype('uint8')
            mask=cv2.resize((labels!=0).astype('uint8'),original.size,interpolation=cv2.INTER_NEAREST)
            n,cc,stats,_=cv2.connectedComponentsWithStats(mask,8)
            for j in range(1,n):
                if stats[j,cv2.CC_STAT_AREA]<25:mask[cc==j]=0
            contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            area=float(mask.sum());perimeter=sum(cv2.arcLength(c,True) for c in contours)
            if contours:
                pts=np.concatenate(contours);_,(w,h),_=cv2.minAreaRect(pts)
                aspect=max(w,h)/max(min(w,h),1)
            else:aspect=np.nan
            circularity=4*np.pi*area/perimeter**2 if perimeter else np.nan
            record=dict(dish_id=row.dish_id,frame_path=row.image_path,area_fraction=area/mask.size,aspect_ratio=aspect,circularity=circularity,contour_complexity=perimeter**2/(4*np.pi*area) if area else np.nan,components=max(0,len(contours)),qa_status='pending_visual_review')
            rows.append(record)
            Image.fromarray(mask*255).save(dest/'masks'/f'{row.dish_id}.png')
            rgb=np.asarray(original).copy();rgb[mask.astype(bool)]=(rgb[mask.astype(bool)]*.55+np.array([255,50,50])*.45).astype('uint8')
            cv2.drawContours(rgb,contours,-1,(255,0,0),2)
            overlay=Image.fromarray(rgb);overlay.save(dest/'overlays'/f'{row.dish_id}.jpg')
            card=Image.new('RGB',(480,170),'white');card.paste(ImageOps.contain(original,(240,140)),(0,0));card.paste(ImageOps.contain(overlay,(240,140)),(240,0));ImageDraw.Draw(card).text((4,142),f'{i+1:03d} {row.dish_id} area={record["area_fraction"]:.2f}',fill='black');cards.append(card)
            print('shape',i+1,'/',len(sample),row.dish_id,'food coverage',round(record['area_fraction'],3),flush=True)
    features=pd.DataFrame(rows)
    features.to_csv(dest/'shape_features_pending_review.csv',index=False)
    for page in range((len(cards)+19)//20):
        sheet=Image.new('RGB',(960,1700),'white')
        for k,card in enumerate(cards[page*20:(page+1)*20]):sheet.paste(card,((k%2)*480,(k//2)*170))
        sheet.save(dest/f'segmentation_review_{page+1:02d}.jpg',quality=93)
    (dest/'method.json').write_text(json.dumps(dict(model='mawiie/food-segmentation-mobilenet',source='https://huggingface.co/mawiie/food-segmentation-mobilenet',dataset='FoodSeg103 pretrained external segmentation model; not trained on NutriVision test data',sampling='100 unique test dishes randomly selected with seed 42; lexicographically first frame per dish; pilot uses first 10',count=len(rows),input='512x512 RGB, ImageNet normalization',mask='argmax food class != 0; nearest resize to original image size; remove connected components under 25 pixels',limitations='Automatic masks require visual review. Two dimensional visible food regions, not volume. Observational associations do not show causal dependence. Area and contour features reflect multiple ingredients and camera framing.'),indent=2))

if __name__=='__main__':main()
