# seed2022

### 安装

参考 https://github.com/FederatedAI/KubeFATE/blob/master/docker-deploy/README_zh.md

### 数据处理

**Guest(政府)**

1. 初赛数据集拷贝到当前目录，并删除所有csv文件的英文表头
2. 当前目录下创建 `output`, `output/model` 两个目录
3. 运行 `python 01_data_process.py` 
4. 会在 `output` 目录下生成目标文件

**Host(电力)**

1. 初赛数据集拷贝到当前目录，并删除所有csv文件的英文表头
2. 当前目录下创建 `output` 目录
3. 运行 `python 01_data_process.py`
4. 会在 `output` 目录下生成目标文件

### 上传数据

分别在 `guest` 和 `host` 的机器上执行对应目录下执行:
1.  `pipeline init --ip 127.0.0.1 --port 9380`
2.  `python 02_upload.py` 上传数据。

### 运行模型

`guest` 端执行

1. 终端执行命令 `pipeline init --ip 127.0.0.1 --port 9380` 连接 `guest` fate flow server
2. 执行 `python 03_sbt_linR_binning.py` 运行

### 查看状态

可以在 `host_ip:8080` 和 `guest_ip:8080` 分别查看任务的进行状态

### 生成结果

运行完成后, 会在 `output` 目录下生成 `03_result_{train_job_id}.csv` 和 `04_submit_{train_job_id}.csv` 文件

## 代码说明
```
├── README.md                   # 使用及说明文件 
├── guest                       # 政府端代码
│   ├── 01_data_process.py      # 数据处理，其中处理的文件都已将第一行英文表头删除掉
│   ├── 02_upload.py            # 上传数据代码
│   ├── 03_sbt_linR_binning.py  # 训练、预测、及结果生成文件
│   ├── config.yaml             # 配置文件
│   ├── output                  # 处理后的数据及模型、结果文件存储，此目录需要手动创建
│   │   ├── model               # 存储训练的模型，需要手动创建
│   └── utils.py                # 一些工具函数
├── host                        # 电力端代码
│   ├── 01_data_process.py      # 数据处理，其中处理的文件都已将第一行英文表头删除掉 
│   ├── 02_upload.py            # 上传数据代码
│   └── output                  # 处理后的数据集，目录需要手动创建
├── requirements.txt
└── 初赛数据集
    ├── test
    │   ├── government_data
    │   │   ├── sb_xssr.csv
    │   │   ├── sz_tmp_baseinfo_ent.csv
    │   │   └── zs_jks.csv
    │   └── power_data
    │       ├── dlsj_df.csv
    │       ├── dlsj_gl_2021.csv
    │       └── dlsj_jcxx.csv
    ├── train
    │   ├── government_data
    │   │   ├── sb_xssr.csv
    │   │   ├── sz_tmp_baseinfo_ent.csv
    │   │   └── zs_jks.csv
    │   └── power_data
    │       ├── dlsj_df.csv
    │       ├── dlsj_gl_2021.csv
    │       └── dlsj_jcxx.csv
    ├── 提交样例.csv
    └── 数据说明.md
```