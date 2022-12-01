from pathlib import Path
class CONFIG:
    guest = 9999
    host = [10000]
    arbiter = 10000
    namespace = 'seed2022'

    dataset_dir = Path('/Users/zhengquan/Code/SEED2022/seed2022/初赛数据集')

    _base_dir = '/Users/zhengquan/Code/SEED2022/seed2022'
    base_dir = Path(_base_dir)

    model_output = base_dir / 'model'
    output_dir = base_dir / 'result'
    dataset_suffix = '_simple_mean'
