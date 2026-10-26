# CNN 独立测试

使用队友提供的 best.pt（最佳 epoch 9）与仓库内 Custom CNN 模型及预处理代码。最终指标仅由 src/evaluation/evaluate.py 计算；不调用队友的 evaluate.py 或 metrics.py。

## 运行

需要 Python、PyTorch、NumPy、Pillow。当前测试环境为 CPU PyTorch 2.14.1+cpu、NumPy 2.2.6、Pillow 12.3.0、Matplotlib 3.10.7；训练依赖记录见 nutrivision_cnn/requirements.txt。无需重新训练。

```powershell
python src/evaluation/test_cnn.py --checkpoint PATH/TO/best.pt --manifest data/manifests/full_frame_manifest.csv --data-root D:/MLData/side_frames_valid --output-dir results/cnn_test --split test --batch-size 32 --threads 4 --compatible-manifest
```

权重文件不随 Git 上传，输出目录必须为空。checkpoint 的哈希将写入结果元数据，用于识别所用权重。

## 当前清单兼容性

当前使用用户确认的 corrected REVIEW 清单。它与训练 checkpoint 中保存的 manifest 文件哈希不同；这里不声称二者逐字节相同。--compatible-manifest 会校验仓库三个划分清单、菜品标签一致性、跨集合隔离、图片存在，以及逐训练菜品计算的四维均值和标准差是否匹配 checkpoint。通过后使用当前清单测试，并在 metadata.json 中同时保留两个文件哈希和校验范围。统计量一致不能证明每个历史训练标签完全一致。

原始训练归一化参数直接从 checkpoint 读取，不使用测试集拟合；carbohydrate 对应模型内部字段 carbs。输入采用 EXIF 方向修正、RGB、整图双线性缩放至 224×224 和 ImageNet 均值/标准差归一化；测试不做随机增强，预测不截断。

## 输出与比较口径

- cnn_images.csv：每张图片一个预测，真实值来自当前 manifest，使用项目统一十列表头。
- cnn_dishes.csv：同一菜品所有测试图片的原单位预测取平均，一份菜品一行。
- mean_baseline_dishes.csv：checkpoint 保存的训练菜品均值，对相同测试菜品作固定预测。
- image_metrics.csv：CNN 图片等权指标。
- dish_comparison.csv：CNN 与 Mean Predictor 在相同测试菜品上的指标，菜品等权。
- metadata.json：清单/权重哈希、checkpoint epoch、设备、样本数和聚合规则。

菜品级导出中的 image_path 为 dish_average/<dish_id>，是聚合记录标识，不是真实图片路径。Mean Predictor 的历史 589 份验证菜品结果与本次 676 份测试菜品结果应分别标注，不能直接跨数据集比较。

测试集只用于已固定 checkpoint 的评估，不据此选择轮数或调参。
