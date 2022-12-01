import os
import time

from config import CONFIG

output_dir = CONFIG.output_dir
guest = CONFIG.guest  # ip 18
host = CONFIG.host  # 19
arbiter = CONFIG.arbiter
partition = 1
namespace = CONFIG.namespace

suffix = CONFIG.dataset_suffix

partition = 2

guest_data = [{
    "file": str(output_dir / f"001_gover_data_train{suffix}.csv"),
    "table_name": f"001_gover_data_train{suffix}",
    "namespace": namespace,
    "head": 1,
    "partition": partition
}, {
    "table_name": f"001_gover_data_test{suffix}",
    "namespace": namespace,
    'file': str(output_dir / f"001_gover_data_test{suffix}.csv"),
    "head": 1,
    "partition": partition
}, {
    "table_name": f"001_gover_data_valid{suffix}",
    "namespace": namespace,
    'file': str(output_dir / f"001_gover_data_valid{suffix}.csv"),
    "head": 1,
    "partition": partition
}]

host_data = [{
    "file": str(output_dir / f"002_power_data_train{suffix}.csv"),
    "table_name": f"002_power_data_train{suffix}",
    "namespace": namespace,
    "head": 1,
    "partition": partition
}, {
    "table_name": f"002_power_data_test{suffix}",
    "namespace": namespace,
    'file': str(output_dir / f"002_power_data_test{suffix}.csv"),
    "head": 1,
    "partition": partition
}, {
    "table_name": f"002_power_data_valid{suffix}",
    "namespace": namespace,
    'file': str(output_dir / f"002_power_data_valid{suffix}.csv"),
    "head": 1,
    "partition": partition
}]


def _upload(flow_ip, flow_port, data):
    from flow_sdk.client import FlowClient

    client = FlowClient(flow_ip, flow_port, 'v1')
    # client.job.submit('conf/upload_conf.json')
    for d in data:
        ret = client.data.upload(d, drop=1, verbose=1)
        job_id = ret['jobId']
        while 1:
            _ = client.job.query(job_id=job_id)
            status = _['data'][0]['f_status']
            print('--> ', d['table_name'], status)
            if status == 'success':
                break

            time.sleep(1)

def upload_data():
    _upload(CONFIG.guest_ip, CONFIG.guest_port, guest_data)
    _upload(CONFIG.host_ip, CONFIG.host_port, host_data)

if __name__ == '__main__':
    upload_data()
