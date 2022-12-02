import os
from .code.config import CONFIG

os.system(f'pipeline init --ip {CONFIG.guest_ip} --port {CONFIG.guest_port}')

from code.feature_engineering import FeatureEng
from code.upload import upload_data
from code.train import SecureBoostModel
from code.predict import predict

if __name__ == '__main__':

    os.makedirs(CONFIG.output_dir, exist_ok=True)
    os.makedirs(CONFIG.model_output, exist_ok=True)

    print('\n==> Feature Engineering <==\n')
    feat_eng = FeatureEng()
    feat_eng.process_guest_data()
    feat_eng.process_host_data()

    print('\n==> Upload data <==\n')
    upload_data()

    print('\n==> Start Training <==\n')

    train_job_id = SecureBoostModel()
    print('train_job_id', train_job_id)

    print('\n==> Start Predicting <==\n')
    predict(train_job_id=train_job_id)

    print('\n==> The End <==\n')
