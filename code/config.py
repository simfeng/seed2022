from pathlib import Path
class CONFIG:
    guest = 10000
    host = [9999]
    arbiter = 9999
    namespace = 'seed2022'

    dataset_dir = Path('/dataset/energy-management')

    _base_dir = '/opt/project/project'
    base_dir = Path(_base_dir)

    model_output = base_dir / 'model'
    output_dir = base_dir / 'result'
    dataset_suffix = '_simple_mean'
