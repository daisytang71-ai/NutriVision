"""Summarize completed CNN test predictions; does not recalculate model metrics."""
import argparse
import csv
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results-dir',type=Path,required=True)
    a=p.parse_args(); root=a.results_dir
    with (root/'dish_comparison.csv').open(encoding='utf-8-sig',newline='') as f:
        metrics=list(csv.DictReader(f))
    meta=json.loads((root/'metadata.json').read_text())
    with (root/'cnn_dishes.csv').open(encoding='utf-8-sig',newline='') as f:
        predictions=list(csv.DictReader(f))
    lines=['# CNN 测试集结果','',
           f'固定最佳权重 epoch {meta["checkpoint_epoch"]}，测试 {meta["images"]:,} 张图片、{meta["dishes"]} 份菜品。',
           '指标由项目 src/evaluation/evaluate.py 计算。以下按菜品等权，同菜品预测先按全部图片取平均。','',
           '|目标|单位|CNN MAE|均值基线 MAE|CNN RMSE|CNN R²|',
           '|---|---|---:|---:|---:|---:|']
    for row in metrics:
        if row['model']!='cnn_dishes': continue
        baseline=next(x for x in metrics if x['model']=='mean_baseline_dishes' and x['target']==row['target'])
        lines.append('|'+ '|'.join([row['target'],row['unit'],f"{float(row['MAE']):.4f}",f"{float(baseline['MAE']):.4f}",f"{float(row['RMSE']):.4f}",f"{float(row['R2']):.4f}" if row['R2'] else 'undefined'])+'|')
    lines.extend(['','CNN 图片级指标另见 image_metrics.csv，不与菜品等权指标混用。',
                  '当前清单与训练时清单的文件哈希不同，已核对当前划分及训练均值/标准差相符；二者哈希均保存在 metadata.json。详情见 ../../docs/cnn_testing.md。',
                  '本次只评估固定权重，不进行测试集调参。图表为测试菜品预测与真实值，坐标轴为原单位。'])
    (root/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    errors=[]
    for row in predictions:
        errors.append(dict(dish_id=row['dish_id'],true_calories=float(row['y_cal']),
                           predicted_calories=float(row['pred_cal']),
                           absolute_error_kcal=abs(float(row['pred_cal'])-float(row['y_cal']))))
    with (root/'calorie_errors_by_dish.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(errors[0])); w.writeheader();w.writerows(sorted(errors,key=lambda x:x['absolute_error_kcal'],reverse=True))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(10,8))
    for ax,short,target,unit in zip(axes.flat,['cal','p','c','f'],['Calories','Protein','Carbohydrate','Fat'],['kcal','g','g','g']):
        y=[float(r['y_'+short]) for r in predictions]; pred=[float(r['pred_'+short]) for r in predictions]
        ax.scatter(y,pred,s=13,alpha=.5,color='#2463a5')
        lo=min(min(y),min(pred));hi=max(max(y),max(pred));ax.plot([lo,hi],[lo,hi],'--',color='#666666')
        ax.set(title=f'{target} (n={len(y)} dishes)',xlabel=f'True ({unit})',ylabel=f'Predicted ({unit})')
    fig.suptitle('CNN test set | mean predictions across each dish\'s images')
    fig.tight_layout();fig.savefig(root/'predicted_vs_true.png',dpi=160);plt.close(fig)


if __name__=='__main__': main()
