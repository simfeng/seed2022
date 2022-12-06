## 运行

1. 执行 `python main.py` , 会执行所有需要的操作
2. 运行结束后，会在生成 `result/result.csv` 文件，即为预测的结果

## 代码说明
1. upload data采用SDK的方式分别上传到guest和host;
2. 数据处理将电力数据和政务数据分别分成 `train`, `valid`, `test` 三部分，`train` 是全部的训练数据，`test` 是全部的测试数据, `valid` 是从 `test` 数据中删除了所有企业21年四季度的数据;
3. 预测是将所有的 `test` 的数据进行预测，然后再结果中找出21年四季度的预测结果;
4. 双方数据通过 `sid` 对齐，`sid` 的格式是 **企业ID+年份+季度**。


## 目录结构
```
├── code
│   ├── data_process_guest.py   # 政府数据特征工程
│   ├── data_process_host.py    # 电力数据特征工程
│   ├── feature_engineering.py  # 特征工程入口文件
│   ├── predict.py              # 预测及结果处理，生成result.csv
│   ├── train.py                # 模型训练
│   ├── upload.py               # 上传数据
│   ├── config.py                   # 配置文件
│   └── utils.py                # 一些工具函数
├── description
│   └── description.md          # 本说明
├── main.py                     # 入口文件
├── model
├── requirements.txt
└── result

```