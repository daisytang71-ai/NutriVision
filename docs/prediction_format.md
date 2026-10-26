# 验证集预测接口

每个模型一份 UTF-8 CSV，文件名作为模型名，列如下（列顺序不限）：

```text
image_path,dish_id,y_cal,y_p,y_c,y_f,pred_cal,pred_p,pred_c,pred_f
```

y 是真实值，pred 是模型预测；cal 为热量 kcal，p 为蛋白质 g，c 为碳水 g，f 为脂肪 g。必须恢复原单位。本程序不负责逆标准化。

每行一张图片，dish_id 与 image_path 联合唯一。各模型应使用相同相对图片路径、同一固定验证集及真实标签。路径用于身份核对，无需在评估电脑上存在。建议每菜品固定一张图片；多图时当前指标按图片等权。

预测文件放 predictions/，模型名如 cnn.csv、resnet18.csv、mean_baseline.csv。另附同名元数据 JSON，字段见 configs/run_metadata.example.json；所有待填项须在正式实验中替换。

缺列、无效数字、缺失值、重复样本会报错。多文件样本或真实标签不一致时停止。R² 在样本不足两条或真实值方差为零时留空并注明原因；负 R² 正常保留。负预测不自动截断。

四个目标的原始误差不会合并。程序只能检查输入文件一致，完整验证集覆盖和训练数据泄漏仍需人工对照 splits/ 清单检查。
