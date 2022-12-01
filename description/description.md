## 运行

1. 执行 `python main.py` , 会执行所有需要的操作
2. 运行结束后，会在生成 `result/result.csv` 文件，即为预测的结果

## 代码说明
upload data采用SDK的方式分别上传到guest和host。

```
├── code
│   ├── data_process_guest.py   # 政府数据特征工程
│   ├── data_process_host.py    # 电力数据特征工程
│   ├── feature_engineering.py  # 特征工程入口文件
│   ├── predict.py              # 预测及结果处理，生成result.csv
│   ├── train.py                # 模型训练
│   ├── upload.py               # 上传数据
│   └── utils.py                # 一些工具函数
├── config.py                   # 配置文件
├── description
│   └── description.md          # 本说明
├── main.py                     # 入口文件
├── model
├── requirements.txt
└── result

```