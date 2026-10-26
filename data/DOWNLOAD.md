# 完整图片获取

状态：完整图片尚未上传网盘，分享链接和提取码待提供。GitHub 已包含完整 manifest 与三个划分清单，但不包含完整图像数据。

## 网盘计划内容

上传 side_frames_valid/，保留其下 dish_id/图片文件名 层级。建议附带同版本 full_frame_manifest.csv、train_ids.txt、val_ids.txt、test_ids.txt，以便下载后运行检查。网盘由组员上传完成后在本页填写真实链接、提取码和版本日期，不填写虚构链接。

## 本地布局

下载并解压到任意非同步的数据目录，例如：

```text
D:/MLData/
├── side_frames_valid/
│   └── dish_.../
│       └── camera_A_f01.jpg
├── full_frame_manifest.csv
├── train_ids.txt
├── val_ids.txt
└── test_ids.txt
```

仓库中的 full_frame_manifest.csv 位于 data/manifests/；三个 ids 文件位于 splits/。可复制到本地上述布局，但不要改写清单内相对图片路径。程序通过配置读取各成员自己的数据根目录，不要求所有人使用 D 盘。

完整图片 140,662 张，33,752,277,043 字节。下载后运行 data/README.md 中的检查命令；仅数量相同不等于内容逐字节一致。

当前标签以仓库 data/manifests/full_frame_manifest.csv 为准（2026-10-26 更新）。准备网盘包时请使用此文件替换旧版本清单。
