import argparse

from pipeline.backend.pipeline import PipeLine
from pipeline.component import DataTransform
from pipeline.component import HeteroLR
from pipeline.component import HeteroFeatureBinning, HeteroFeatureSelection, DataStatistics, Evaluation
from pipeline.component import Intersection
from pipeline.component import Reader
from pipeline.interface import Data
from pipeline.component import Evaluation
from pipeline.component import HeteroDataSplit
from pipeline.component import FeatureScale
from pipeline.interface import Model

from pipeline.utils.tools import load_job_config


def main(config="config.yaml", namespace="seed2022"):
    # obtain config
    if isinstance(config, str):
        config = load_job_config(config)
    parties = config.parties
    guest = parties.guest[0]
    host = parties.host[0]
    arbiter = parties.arbiter[0]

    # data sets
    guest_train_data = {"name": "gover_data_train", "namespace": namespace}
    host_train_data = {"name": "power_data_train", "namespace": namespace}

    guest_test_data = {"name": "gover_data_test", "namespace": namespace}
    host_test_data = {"name": "power_data_test", "namespace": namespace}

    guest_train_data = {"name": "gover_data_train_3", "namespace": namespace}
    host_train_data = {"name": "power_data_train_3", "namespace": namespace}

    guest_test_data = {"name": "gover_data_test_3", "namespace": namespace}
    host_test_data = {"name": "power_data_test_3", "namespace": namespace}

    # initialize pipeline
    pipeline = PipeLine()
    # set job initiator
    pipeline.set_initiator(role='guest', party_id=guest)
    # set participants information
    pipeline.set_roles(guest=guest, host=host, arbiter=arbiter)

    # define Reader components to read in data
    reader_0 = Reader(name="reader_0")
    reader_1 = Reader(name="reader_1")
    # configure Reader for guest
    reader_0.get_party_instance(
        role='guest', party_id=guest).component_param(table=guest_train_data)
    reader_1.get_party_instance(
        role='guest', party_id=guest).component_param(table=guest_test_data)
    # configure Reader for host
    reader_0.get_party_instance(
        role='host', party_id=host).component_param(table=host_train_data)
    reader_1.get_party_instance(
        role='host', party_id=host).component_param(table=host_test_data)

    # define DataTransform components
    data_transform_0 = DataTransform(
        name="data_transform_0")  # start component numbering at 0
    data_transform_1 = DataTransform(
        name="data_transform_1")  # start component numbering at 1

    param = {
        "with_label": True,
        "label_name": "sjje_per_month",
        "label_type": "float",
        "output_format": "dense",
        "missing_fill": True,
        "missing_fill_method": "mean",
        "outlier_replace": False,
        "outlier_replace_method": "designated",
        "outlier_replace_value": 0.66,
        "outlier_impute": "-9999"
    }
    # get DataTransform party instance of guest
    data_transform_0_guest_party_instance = data_transform_0.get_party_instance(
        role='guest', party_id=guest)
    # configure DataTransform for guest
    data_transform_0_guest_party_instance.component_param(**param)
    # get and configure DataTransform party instance of host
    data_transform_1.get_party_instance(
        role='guest', party_id=guest).component_param(**param)

    param = {
        # "input_format": "tag",
        "with_label": False,
        # "tag_with_value": True,
        # "delimitor": ";",
        "output_format": "dense"
    }
    data_transform_0.get_party_instance(role='host',
                                        party_id=host).component_param(**param)
    data_transform_1.get_party_instance(role='host',
                                        party_id=host).component_param(**param)

    # define Intersection components
    intersection_0 = Intersection(name="intersection_0",
                                  intersect_method="raw")
    intersection_1 = Intersection(name="intersection_1",
                                  intersect_method="raw")

    param = {
        "name": 'hetero_feature_binning_0',
        "method": 'optimal',
        "optimal_binning_param": {
            "metric_method": "iv",
            "init_bucket_method": "quantile"
        },
        "bin_indexes": -1
    }
    hetero_feature_binning_0 = HeteroFeatureBinning(**param)
    statistic_0 = DataStatistics(name='statistic_0')
    param = {
        "name": 'hetero_feature_selection_0',
        "filter_methods": ["unique_value",
                            # "iv_filter",
                            "statistic_filter"],
        "unique_param": {
            "eps": 1e-6
        },
        # "iv_param": {
        #     "metrics": ["iv", "iv"],
        #     "filter_type": ["top_k", "threshold"],
        #     "take_high": [True, True],
        #     "threshold": [10, 0.1]
        # },
        "statistic_param": {
            "metrics": ["coefficient_of_variance", "skewness"],
            "filter_type": ["threshold", "threshold"],
            "take_high": [True, False],
            "threshold": [0.001, -0.01]
        },
        "select_col_indexes": -1
    }
    hetero_feature_selection_0 = HeteroFeatureSelection(**param)
    hetero_feature_selection_1 = HeteroFeatureSelection(
        name='hetero_feature_selection_1')
    param = {"name": "hetero_scale_0", "method": "standard_scale"}
    hetero_scale_0 = FeatureScale(**param)
    hetero_scale_1 = FeatureScale(name='hetero_scale_1')
    param = {
        "penalty": "L2",
        "optimizer": "nesterov_momentum_sgd",
        "tol": 1e-4,
        "alpha": 0.01,
        "max_iter": 5,
        "early_stop": "diff",
        "batch_size": -1,
        "learning_rate": 0.15,
        "init_param": {
            "init_method": "zeros"
        },
        "validation_freqs": None,
        "early_stopping_rounds": None
    }

    hetero_lr_0 = HeteroLR(name='hetero_lr_0', **param)
    evaluation_0 = Evaluation(name='evaluation_0', eval_type="regression")
    # add components to pipeline, in order of task execution
    pipeline.add_component(reader_0)
    pipeline.add_component(reader_1)
    pipeline.add_component(data_transform_0,
                           data=Data(data=reader_0.output.data))
    pipeline.add_component(data_transform_1,
                           data=Data(data=reader_1.output.data),
                           model=Model(data_transform_0.output.model))

    # set data input sources of intersection components
    pipeline.add_component(intersection_0,
                           data=Data(data=data_transform_0.output.data))
    pipeline.add_component(intersection_1,
                           data=Data(data=data_transform_1.output.data))

    # set train & validate data of hetero_lr_0 component
    # pipeline.add_component(hetero_feature_binning_0,
    #                        data=Data(data=intersection_0.output.data))

    pipeline.add_component(statistic_0,
                           data=Data(data=intersection_0.output.data))

    pipeline.add_component(
        hetero_feature_selection_0,
        data=Data(data=intersection_0.output.data),
        model=Model(isometric_model=[
            # hetero_feature_binning_0.output.model,
            statistic_0.output.model
        ]))
    pipeline.add_component(hetero_feature_selection_1,
                           data=Data(data=intersection_1.output.data),
                           model=Model(
                               hetero_feature_selection_0.output.model))

    pipeline.add_component(
        hetero_scale_0, data=Data(data=hetero_feature_selection_0.output.data))
    pipeline.add_component(
        hetero_scale_1,
        data=Data(data=hetero_feature_selection_1.output.data),
        model=Model(hetero_scale_0.output.model))

    # set train & validate data of hetero_lr_0 component
    pipeline.add_component(hetero_lr_0,
                           data=Data(train_data=hetero_scale_0.output.data,
                                     validate_data=hetero_scale_1.output.data))

    pipeline.add_component(evaluation_0,
                           data=Data(data=[hetero_lr_0.output.data]))
    # compile pipeline once finished adding modules, this step will form conf and dsl files for running job
    pipeline.compile()
    pipeline.fit()

    print("fitting hetero secureboost done, result:")
    summ = pipeline.get_component("hetero_secure_boost_0").get_summary()
    print(summ)

    # save
    pipeline.dump(f"output/model/sbt_lr.pkl")

    # from pipeline.backend.pipeline import PineLine


    # PipeLine.load_model_from_file("pipeline_saved.pkl")

    # predict
    # deploy required components
    pipeline.deploy_component([
        data_transform_0, intersection_0,
        statistic_0, hetero_feature_selection_0, hetero_scale_0, hetero_lr_0,
        evaluation_0
    ])

    predict_pipeline = PipeLine()
    # add data reader onto predict pipeline
    predict_pipeline.add_component(reader_1)
    # add selected components from train pipeline onto predict pipeline
    # specify data source
    predict_pipeline.add_component(
        pipeline,
        data=Data(predict_input={
            pipeline.data_transform_0.input.data: reader_1.output.data
        }))

    # run predict model
    predict_pipeline.predict()
    predict_result = predict_pipeline.get_component(
        "hetero_secure_boost_0").get_output_data()
    print("Showing 10 data of predict result")
    print(predict_result.head(10))
    predict_result.to_csv('output/result.csv')


if __name__ == "__main__":
    parser = argparse.ArgumentParser("PIPELINE DEMO")
    parser.add_argument("-config", type=str, help="config file")
    args = parser.parse_args()
    if args.config is not None:
        main(args.config)
    else:
        main()
