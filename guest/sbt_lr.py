import argparse

from pipeline.backend.pipeline import PipeLine
from pipeline.component import DataTransform
from pipeline.component import HeteroSecureBoost
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

    guest_train_data = {"name": "gover_data_train_3_all", "namespace": namespace}
    host_train_data = {"name": "power_data_train_3_all", "namespace": namespace}

    guest_test_data = {"name": "gover_data_test_3_all", "namespace": namespace}
    host_test_data = {"name": "power_data_test_3_all", "namespace": namespace}

    # guest_train_data = {
    #     "name": "gover_data_train_3_all_one_hot",
    #     "namespace": namespace
    # }
    # host_train_data = {
    #     "name": "power_data_train_3_all_one_hot",
    #     "namespace": namespace
    # }

    # guest_test_data = {
    #     "name": "gover_data_test_3_all_one_hot",
    #     "namespace": namespace
    # }
    # host_test_data = {
    #     "name": "power_data_test_3_all_one_hot",
    #     "namespace": namespace
    # }


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
        role="guest", party_id=guest).component_param(table=guest_test_data)

    reader_0.get_party_instance(
        role="host", party_id=host).component_param(table=host_train_data)
    reader_1.get_party_instance(
        role="host", party_id=host).component_param(table=host_test_data)


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

    # feature scale
    scale_train_0 = FeatureScale(name="scale_train_0")
    scale_train_1 = FeatureScale(name="scale_train_1")

    # secure boost component
    """ 成绩31835889.69259772  gover_data_train_3
    hetero_secure_boost_0 = HeteroSecureBoost(
        name="hetero_secure_boost_0",
        learning_rate=0.2,
        num_trees=10,
        task_type="regression",
        objective_param={"objective": "lse"},
        encrypt_param={"method": "Paillier"},
        tree_param={"max_depth": 15},
        validation_freqs=1,
        bin_num=200,
        run_goss=True,
        # work_mode=1,
        tree_num_per_party=10,
        guest_depth=13,
        host_depth=15,
    )
    """
    hetero_secure_boost_0 = HeteroSecureBoost(
        name="hetero_secure_boost_0",
        learning_rate=0.3,
        num_trees=10,
        task_type="regression",
        objective_param={"objective": "lse"},
        encrypt_param={"method": "Paillier"},
        tree_param={"max_depth": 16},
        validation_freqs=1,
        bin_num=500,
        run_goss=True,
        # work_mode=1,
        tree_num_per_party=15,
        guest_depth=16,
        host_depth=16,
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

    # pipeline.add_component(scale_train_0,
    #                        data=Data(data=intersect_0.output.data))
    # pipeline.add_component(scale_train_1,
    #                        data=Data(data=intersect_1.output.data),
    #                        model=Model(scale_train_0.output.model))

    pipeline.add_component(hetero_secure_boost_0,
                           data=Data(train_data=intersect_0.output.data,
                                     validate_data=intersect_1.output.data))
    pipeline.add_component(evaluation_0,
                           data=Data(data=hetero_secure_boost_0.output.data))

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
        data_transform_0, intersect_0, hetero_secure_boost_0,
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
