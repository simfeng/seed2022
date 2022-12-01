#/bin/bash

# 创建目录
mkdir model result

# 处理数据
python code/feature_engineering.py

# 上传host数据
pipeline init --ip 127.0.0.1 --port 12345

python code/upload_host.py

# 上传guest数据
pipeline init --ip 127.0.0.1 --port 12346

python code/upload_guest.py

# 训练
pipeline init --ip 127.0.0.1 --port 12346

python code/train_and_predict.py
