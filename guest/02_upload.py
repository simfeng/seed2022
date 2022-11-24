from pathlib import Path

data_base = Path.cwd()
guest = 9999  # ip 18
host = 10000  # 19
arbiter = 10000
partition = 1
namespace = 'seed2022'
print(data_base)

suffix = '_simple_mean'  # or _simple

train_data = {
    "name": f"001_gover_data_train{suffix}",
    "namespace": namespace,
    'path': str(data_base / f"output/001_gover_data_train{suffix}.csv")
}

test_data = {
    "name": f"001_gover_data_test{suffix}",
    "namespace": namespace,
    'path': str(data_base / f"output/001_gover_data_test{suffix}.csv")
}

valid_data = {
    "name": f"001_gover_data_valid{suffix}",
    "namespace": namespace,
    'path': str(data_base / f"output/001_gover_data_valid{suffix}.csv")
}

from pipeline.backend.pipeline import PipeLine

def upload():

    pipeline = PipeLine()
    pipeline.set_initiator(role='guest', party_id=guest)
    pipeline.set_roles(guest=guest, host=host, arbiter=arbiter)

    pipeline.add_upload_data(
        file=train_data["path"],
        table_name=train_data["name"],  # table name
        namespace=train_data["namespace"],  # namespace
        head=1,
        partition=partition)
    pipeline.add_upload_data(
        file=test_data['path'],
        table_name=test_data["name"],  # table name
        namespace=test_data["namespace"],  # namespace
        head=1,
        partition=partition)
    pipeline.add_upload_data(
        file=valid_data['path'],
        table_name=valid_data["name"],  # table name
        namespace=valid_data["namespace"],  # namespace
        head=1,
        partition=partition)

    pipeline.upload(drop=1)

if __name__ == '__main__':
    upload()
