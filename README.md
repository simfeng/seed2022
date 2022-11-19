# seed2022


### 数据处理

1. 初赛数据集拷贝到当前目录，并删除所有csv文件的英文表头
2. 当前目录下创建 `output`, `output/model` 两个目录
3. 运行 `data_process.py` 文件, 注意文件结尾有一些参数需要配置
4. 会在 `output` 目录下生成 `{gover|power}_data_{train|test}_3.csv` 和 `ID_date_templ_test.csv` 至少五个文件

### 上传数据

执行 `upload.ipynb` 里面的代码分别上传 `guest` 和 `host` 数据

### 运行模型

执行 `python sbt_lr.py` 运行

### 查看状态

可以在 `host_ip:8080` 和 `guest_ip:8080` 分别查看任务的进行状态

### 生成结果

1. 运行完成后, 会在 `output` 目录下生成 `result.csv` 文件
2. 执行 `load_and_predict.ipynb` 里面的命令可以生成提交文件 `submit.csv`

**注意**

模型文件 `sbt_lr.py` 使用的配置来自 `config.yaml`, 但是 'upload.ipynb` 里面的配置直接写在里面了
