import os
from config import CONFIG

output_dir = CONFIG.output_dir
guest = CONFIG.guest  # ip 18
host = CONFIG.host  # 19
arbiter = CONFIG.arbiter
partition = 1
namespace = CONFIG.namespace

suffix = CONFIG.dataset_suffix

train_data = {
    "name": f"001_gover_data_train{suffix}",
    "namespace": namespace,
    'path': str(output_dir / f"001_gover_data_train{suffix}.csv")
}

test_data = {
    "name": f"001_gover_data_test{suffix}",
    "namespace": namespace,
    'path': str(output_dir / f"001_gover_data_test{suffix}.csv")
}

valid_data = {
    "name": f"001_gover_data_valid{suffix}",
    "namespace": namespace,
    'path': str(output_dir / f"001_gover_data_valid{suffix}.csv")
}


def upload():
    from pipeline.backend.pipeline import PipeLine

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
