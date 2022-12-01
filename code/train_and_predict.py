from train import SecureBoostModel
from predict import predict

if __name__ == '__main__':

    train_job_id = SecureBoostModel()
    print('train_job_id', train_job_id)
    predict(train_job_id=train_job_id)
