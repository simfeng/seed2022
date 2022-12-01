from config import CONFIG
from data_process_guest import gover_data
from data_process_host import power_data

class FeatureEng:

    def __init__(self) -> None:

        self.dataset_dir = CONFIG.dataset_dir
        self.output_dir = CONFIG.output_dir
        self.suffix = CONFIG.dataset_suffix
        self.data_type = ''

    def process_guest_data(self) -> None:
        print('self.data:', self.data_type)
        self.data_type = 'train'
        gover_data(self)

        self.data_type = 'test'
        gover_data(self)


    def process_host_data(self) -> None:
        print('self.data:', self.data_type)
        self.data_type = 'train'
        power_data(self)

        self.data_type = 'test'
        power_data(self)


if __name__ == '__main__':
    # process_guest_data()
    feat_eng = FeatureEng()
    feat_eng.process_guest_data()
    feat_eng.process_host_data()
