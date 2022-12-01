from pipeline.backend.pipeline import PipeLine
from pipeline.component import DataTransform
from pipeline.component import HeteroSecureBoost
from pipeline.component import Intersection
from pipeline.component import Reader
from pipeline.interface import Data
from pipeline.component import Evaluation, Union, HeteroFeatureBinning
from pipeline.component import HeteroDataSplit
from pipeline.component import FeatureScale, OneHotEncoder
from pipeline.interface import Model

from pipeline.utils.tools import load_job_config

from config import CONFIG


def SecureBoostModel():
    guest = CONFIG.guest
    host = CONFIG.host
    arbiter = CONFIG.arbiter
    namespace = CONFIG.namespace
    suffix = CONFIG.dataset_suffix
    model_output_dir = CONFIG.model_output

    # data sets
    guest_train_data = {
        "name": f"001_gover_data_train{suffix}",
        "namespace": namespace,
    }
    host_train_data = {
        "name": f"002_power_data_train{suffix}",
        "namespace": namespace,
    }

    guest_valid_data = {
        "name": f"001_gover_data_valid{suffix}",
        "namespace": namespace,
    }
    host_valid_data = {
        "name": f"002_power_data_valid{suffix}",
        "namespace": namespace,
    }

    # init pipeline
    pipeline = PipeLine().set_initiator(role="guest",
                                        party_id=guest).set_roles(
                                            guest=guest,
                                            host=host,
                                            arbiter=arbiter
                                        )

    # set data reader and data-io

    reader_0 = Reader(name="reader_0")
    reader_1 = Reader(name="reader_1")


    reader_0.get_party_instance(
        role="guest", party_id=guest).component_param(table=guest_train_data)
    reader_1.get_party_instance(
        role="guest", party_id=guest).component_param(table=guest_valid_data)


    reader_0.get_party_instance(
        role="host", party_id=host).component_param(table=host_train_data)
    reader_1.get_party_instance(
        role="host", party_id=host).component_param(table=host_valid_data)


    data_transform_0 = DataTransform(name="data_transform_0")
    data_transform_1 = DataTransform(name="data_transform_1")

    data_transform_0.get_party_instance(role="guest",
                                        party_id=guest).component_param(
                                            with_label=True,
                                            output_format="dense",
                                            label_name="sjje_per_month",
                                            label_type="float")
    data_transform_1.get_party_instance(role="guest",
                                        party_id=guest).component_param(
                                            with_label=True,
                                            output_format="dense",
                                            label_name="sjje_per_month",
                                            label_type="float")

    data_transform_0.get_party_instance(
        role="host", party_id=host).component_param(with_label=False)
    data_transform_1.get_party_instance(
        role="host", party_id=host).component_param(with_label=False)


    # data intersect component
    intersect_0 = Intersection(name="intersection_0")
    intersect_1 = Intersection(name="intersection_1")

    # secure boost component
    """ 19333101.099907372
    hetero_secure_boost_0 = HeteroSecureBoost(
        name="hetero_secure_boost_0",
        learning_rate=0.2,
        num_trees=14,
        task_type="regression",
        objective_param={"objective": "lse"},
        encrypt_param={"method": "Paillier"},
        tree_param={"max_depth": 4},
        validation_freqs=2,
        # boosting_strategy='layered',
        bin_num=1000,
        run_goss=True,
        # subsample_feature_rate=0.9,
        # work_mode=1,
        tree_num_per_party=10,
        guest_depth=4,
        host_depth=4

    )
    """

    hetero_secure_boost_0 = HeteroSecureBoost(
        name="hetero_secure_boost_0",
        learning_rate=0.2,
        num_trees=2,
        task_type="regression",
        objective_param={"objective": "lse"},
        encrypt_param={"method": "Paillier"},
        tree_param={"max_depth": 2},
        validation_freqs=2,
        # boosting_strategy='layered',
        bin_num=1000,
        run_goss=True,
        # subsample_feature_rate=0.9,
        # work_mode=1,
        tree_num_per_party=10,
        guest_depth=4,
        host_depth=4

    )

    # evaluation component
    evaluation_0 = Evaluation(name="evaluation_0", eval_type="regression")

    pipeline.add_component(reader_0)
    pipeline.add_component(reader_1)
    pipeline.add_component(data_transform_0,
                           data=Data(data=reader_0.output.data))
    pipeline.add_component(data_transform_1,
                           data=Data(data=reader_1.output.data),
                           model=Model(data_transform_0.output.model))

    pipeline.add_component(intersect_0,
                           data=Data(data=data_transform_0.output.data))
    pipeline.add_component(intersect_1,
                           data=Data(data=data_transform_1.output.data))

    # pipeline.add_component(
    #     union_0,
    #     data=Data(data=[intersect_0.output.data, intersect_1.output.data]))

    param = {
        "method": "quantile",
        "compress_thres": 10000,
        "head_size": 10000,
        "error": 0.001,
        "bin_num": 500,
        "bin_indexes": -1,
        "bin_names": None,
        "category_indexes": None,
        "category_names": None,
        "adjustment_factor": 0.5,
        "local_only": False,
        "skip_static": True,
        "transform_param": {
            "transform_cols": -1,
            "transform_names": None,
            "transform_type": "bin_num"
        }
    }

    hetero_feature_binning_0 = HeteroFeatureBinning(name="hetero_feature_binning_0", **param)
    hetero_feature_binning_1 = HeteroFeatureBinning(name="hetero_feature_binning_1")

    pipeline.add_component(hetero_feature_binning_0,
                           data=Data(data=intersect_0.output.data))
    pipeline.add_component(hetero_feature_binning_1,
                           data=Data(data=intersect_1.output.data),
                           model=Model(hetero_feature_binning_0.output.model))

    pipeline.add_component(
        hetero_secure_boost_0,
        data=Data(train_data=hetero_feature_binning_0.output.data,
                  validate_data=hetero_feature_binning_1.output.data))
    pipeline.add_component(evaluation_0,
                           data=Data(data=hetero_secure_boost_0.output.data))

    pipeline.compile()
    pipeline.fit()

    print("fitting hetero secureboost done, result:")
    summ = pipeline.get_component("hetero_secure_boost_0").get_summary()
    print(summ)
    train_job_id = pipeline.get_train_job_id()
    # save
    pipeline.dump(model_output_dir / f"{train_job_id}.pkl")

    return train_job_id
