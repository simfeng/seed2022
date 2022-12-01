#/bin/bash

# 创建目录
mkdir ../model ../result

# 处理数据
python feature_engineering.py

# 上传host数据
pipeline init --ip 10.211.55.19 --port 9380

python upload_host.py

# 上传guest数据
pipeline init --ip 10.211.55.18 --port 9380

python upload_guest.py

# 训练
pipeline init --ip 10.211.55.18 --port 9380

python train_and_predict.py
