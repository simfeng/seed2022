from .config import CONFIG
from .data_process_guest import data_process as data_process_guest
from .data_process_host import data_process as data_process_host

class FeatureEng:

    def __init__(self) -> None:

        self.dataset_dir = CONFIG.dataset_dir
        self.output_dir = CONFIG.output_dir
        self.suffix = CONFIG.dataset_suffix
        self.data_type = ''

    def process_guest_data(self) -> None:
        data_process_guest(self)


    def process_host_data(self) -> None:
        data_process_host(self)


if __name__ == '__main__':
    # process_guest_data()
    feat_eng = FeatureEng()
    feat_eng.process_guest_data()
    feat_eng.process_host_data()
