# NutriVision

Single-image nutritional estimation via transfer learning. 从单张 RGB 食物图像预测热量、蛋白质、碳水化合物和脂肪，比较从零训练的 CNN 与预训练模型。

## 当前状态

已实现并用人工构造数据核对统一评估程序。现已上传全量图片清单及 train/val/test 菜品编号，本地检查覆盖 140,662 张图片、4,605 份菜品，三个划分 dish_id 无交叉。完整图片通过网盘共享，当前尚未上传网盘，链接待补充。

检查结果与范围见 [数据说明](data/README.md)，图片获取方式见 [下载说明](data/DOWNLOAD.md)。模型代码和训练记录由各负责人自行上传。

## 目录与分工

```text
NutriVision/
├── configs/                 # 实际训练配置与环境记录
├── data/                    # 数据说明；原始数据不上传
├── splits/                  # 固定菜品划分清单与版本说明
├── src/
│   ├── data/                # 数据读取、预处理、划分检查
│   ├── models/              # Custom CNN、ResNet18/MobileNetV2
│   ├── training/            # 训练入口与验证过程
│   └── evaluation/          # 指标计算、均值基线及误差分析
├── predictions/             # 各模型验证集预测 CSV
├── results/
│   ├── metrics/             # 指标表
│   └── figures/             # 散点图、曲线与误差案例
├── reports/                 # 一页进展报告、PPT 和最终报告
├── docs/                    # 接口约定、协作说明
├── tests/                   # 评估程序检查
└── requirements.txt         # 当前已实现功能的依赖说明
```

## 快速开始

当前评估程序仅需 Python 3.10 或更新版本，无需安装第三方包。以下命令从仓库根目录执行，预测文件需由实际运行产生。

```powershell
python src/evaluation/evaluate.py predictions/cnn.csv --output results/metrics/cnn_metrics.csv
python src/evaluation/evaluate.py predictions/mean_baseline.csv predictions/cnn.csv predictions/resnet18.csv --output results/metrics/comparison.csv
python -m unittest discover -s tests -v
```

多文件评估要求样本及真实标签一致，允许行顺序不同。指标逐目标输出 MAE、RMSE、R²；以热量 MAE 为主指标。完整格式见 [预测接口](docs/prediction_format.md)。

## 数据与实验协议

使用 Nutrition5k RGB 图像和营养标签；实际下载范围、数据来源、筛选规则与样本统计由数据负责人补充到 data/README.md。遵循官方菜品级 train/test 划分，仅从训练菜品划出验证集。同一菜品不得跨集合。

本阶段优先 Mean Predictor、Custom CNN 与 ResNet18；MobileNetV2 和交互演示为后续范围。均值基线仅用训练集四个标签的均值。模型使用相同验证集；目标标准化参数仅从训练集计算，导出预测前恢复 kcal/g。

每次实验记录 run_id、代码 commit、数据与划分版本、随机种子、实际配置、设备、轮数、预训练来源、checkpoint、日志和预测路径。没有真实结果时明确记录状态，不填示例分数。

## 团队协作

- Member A：蔡雨静，数据准备、划分与检查。
- Member B：曹可昕，Custom CNN 与训练。
- Member C：滕姝慧，预训练模型微调。
- Member D：汤远航，统一评估、可视化、复现与材料整合。

各成员在自己的分支提交，通过 Pull Request 合并；提交前阅读 [协作说明](docs/contributing.md)。原始图像、完整数据包、权重、缓存、密钥与本机环境文件不提交。评估代码可独立使用；训练依赖由模型成员按实际验证环境补充，当前不宣称训练流程可复现。

## Mean Predictor 当前进度

均值基线已完成实现与真实清单上的试运行。训练集 3,340 份菜品，验证集 589 份菜品；每份菜品按 image_path 字典序固定选第一张，不读取图像内容，不评估测试集。只用选中的训练标签计算四个均值。

预测、指标、训练均值、数据校验值及选图清单见 [基线运行记录](results/mean_predictor/README.md)。其他模型比较前须采用完全相同的验证样本和正确标签。

```powershell
python src/evaluation/mean_predictor.py --manifest data/manifests/full_frame_manifest.csv --output-dir results/mean_predictor --sampling one-per-dish
```

默认 one-per-dish 实现菜品等权。若模型负责人实际使用全量多图，可指定 --sampling all-images 并使用另一个输出目录；该设置按训练图片等权，均值与指标都可能不同，不能混用两种协议。
