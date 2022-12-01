from pipeline.backend.pipeline import PipeLine

from pipeline.component import Reader
from pipeline.interface import Data

from pipeline.utils.tools import load_job_config

from config import CONFIG

from utils import generate_result


def predict(train_job_id):

    guest = CONFIG.guest
    host = CONFIG.host
    arbiter = CONFIG.arbiter
    namespace = CONFIG.namespace
    suffix = CONFIG.dataset_suffix
    model_output_dir = CONFIG.model_output
    output_dir = CONFIG.output_dir

    guest_test_data = {
        "name": f"001_gover_data_test{suffix}",
        "namespace": namespace,
    }
    host_test_data = {
        "name": f"002_power_data_test{suffix}",
        "namespace": namespace,
    }


    # init pipeline

    pipeline = PipeLine.load_model_from_file(model_output_dir /
                                             f"{train_job_id}.pkl")
    pipeline.deploy_component([
        pipeline.data_transform_0,
        pipeline.intersection_0,
        # scale_train_0,
        pipeline.hetero_feature_binning_0,
        pipeline.hetero_secure_boost_0,
        pipeline.evaluation_0
    ])
    predict_pipeline = PipeLine()
    # add data reader onto predict pipeline
    reader_0 = Reader(name="reader_0")
    reader_0.get_party_instance(
        role="guest", party_id=guest).component_param(table=guest_test_data)
    reader_0.get_party_instance(
        role="host", party_id=host).component_param(table=host_test_data)
    predict_pipeline.add_component(reader_0)
    # add selected components from train pipeline onto predict pipeline
    # specify data source
    predict_pipeline.add_component(
        pipeline,
        data=Data(predict_input={
            pipeline.data_transform_0.input.data: reader_0.output.data
        }))

    # run predict model
    predict_pipeline.predict()
    predict_result = predict_pipeline.get_component(
        "hetero_secure_boost_0").get_output_data()
    print("Showing 10 data of predict result")
    print(predict_result.head(10))
    predict_result_file = output_dir / f'predict_{train_job_id}.csv'
    submit_result_file = output_dir / f'result_{train_job_id}.csv'
    predict_result.to_csv(predict_result_file)

    generate_result(input_file=predict_result_file,
                    output_file=submit_result_file)
