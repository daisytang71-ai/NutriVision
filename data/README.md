# 数据目录与当前状态

截至 2026-10-26，本地已收到 140,662 张 JPG，覆盖 4,605 份菜品，图片总大小 33,752,277,043 字节（约 33.75 GB）。这些是本地检查得到的数量，不表示标签已验证或完成模型训练。

## 文件位置

| 内容 | 仓库位置 | 状态 |
|---|---|---|
| 全量图片与标签清单 | manifests/full_frame_manifest.csv | 已上传，保留原值 |
| 训练、验证、测试菜品编号 | ../splits/*_ids.txt | 已上传 |
| 本地核对结果 | checks/local_data_audit.json | 已上传 |
| 清单与划分文件校验值 | checks/SHA256SUMS.json | 已上传 |
| 完整图片 | 本地或网盘 side_frames_valid/ | 未上传 GitHub，网盘尚未上传 |
| 网盘获取与目录说明 | DOWNLOAD.md | 分享链接待补充 |

## 数据划分检查

| 划分 | 菜品数 | 图片数 |
|---|---:|---:|
| train | 3,340 | 102,087 |
| val | 589 | 17,889 |
| test | 676 | 20,686 |

清单中的图片均在本地存在，没有重复图片路径、额外未列出的图片或划分不匹配记录。三个划分的 dish_id 两两无交叉。此检查不证明它们来自官方划分，不检查图片内容重复，也不验证图像解码和标签真实性。

## 标签字段需要核对

清单字段为 dish_id,camera,frame_idx,image_path,calories,protein,carbohydrate,fat,split。image_path 相对于 side_frames_valid/。

标签尚未验证，可能存在列映射问题：4,403 份菜品出现 4 × protein 大于 calories；例如 dish_1559845513 标为 222.152008 kcal、protein 379.0 g。此处是异常筛查线索，不能仅凭它确定如何修正。需数据负责人对照原始营养元数据和生成代码核对字段；当前文件逐字节保留，不更改也不宣称标签已清洗正确。正式训练和指标解释前必须处理这一问题。

同一菜品有多张图片。当前评估程序按图片等权；本轮如果每菜品只使用一张，须记录共同选图规则，不能将全部清单行数直接写成本轮实际使用量。

## 重跑本地核对

```powershell
python src/data/audit_local_data.py --data-root D:/MLData --output data/checks/local_data_audit.json
```

数据处理和模型代码由各成员自行上传到 src/ 对应目录。原始数据来源、许可要求、筛选规则与官方划分出处仍待数据负责人补充。
