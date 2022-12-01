#/bin/bash

# 创建目录
echo '===> Make dir model result <==='
mkdir model result

# 处理数据
echo '===> Feature Enginerring <==='
python code/feature_engineering.py

# 上传host数据
echo '===> host pipeline init --ip 127.0.0.1 --port 12345 <==='
pipeline init --ip 127.0.0.1 --port 12345
echo '===> Upload host data <==='
python code/upload_host.py

# 上传guest数据
echo '===> guest pipeline init --ip 127.0.0.1 --port 12346 <==='
pipeline init --ip 127.0.0.1 --port 12346
echo '===> Upload guest data <==='
python code/upload_guest.py

# 训练
echo '===> guest pipeline init --ip 127.0.0.1 --port 12346 <==='
pipeline init --ip 127.0.0.1 --port 12346
echo '===> Train and Predict <==='
python code/train_and_predict.py
