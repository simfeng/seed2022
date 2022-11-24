# seed2022

### 安装

参考 https://github.com/FederatedAI/KubeFATE/blob/master/docker-deploy/README_zh.md

### 数据处理

**Guest(政府)**

1. 初赛数据集拷贝到当前目录，并删除所有csv文件的英文表头
2. 当前目录下创建 `output`, `output/model` 两个目录
3. 运行 `01_data_process.py` 文件, 注意文件结尾有一些参数需要配置
4. 会在 `output` 目录下生成目标文件

**Host(电力)**

1. 初赛数据集拷贝到当前目录，并删除所有csv文件的英文表头
2. 当前目录下创建 `output`两个目录
3. 运行 `01_data_process.py` 文件, 注意文件结尾有一些参数需要配置
4. 会在 `output` 目录下生成目标文件

### 上传数据

分别在 `guest` 和 `host` 的机器上执行对应目录下的 `02_upload.py` 文件上传数据。

### 运行模型

`guest` 端执行

1. 终端执行命令 `pipeline init --ip {guest_ip} --port 9380` 连接 `guest` fate flow server
2. 执行 `python 03_sbt_linR_binning.py` 运行

### 查看状态

可以在 `host_ip:8080` 和 `guest_ip:8080` 分别查看任务的进行状态

### 生成结果

运行完成后, 会在 `output` 目录下生成 `03_result.csv` 和 `04_submit.csv` 文件

