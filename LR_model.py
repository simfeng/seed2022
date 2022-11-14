import argparse
import os
from pathlib import Path

from pipeline.backend.pipeline import PipeLine
from pipeline.component import DataTransform
from pipeline.component import HeteroDataSplit
from pipeline.component import HeteroLinR
from pipeline.component import Intersection
from pipeline.component import FeatureScale
from pipeline.component import Reader
from pipeline.component import Evaluation
from pipeline.interface import Data, Model

from pipeline.utils.tools import load_job_config


def main(config="config.yaml", namespace="seed2022"):
    guest_train_data = {"name": "gover_data_train", "namespace": namespace}
    host_train_data = {"name": "power_data_train", "namespace": namespace}

    guest_test_data = {"name": "gover_data_test", "namespace": namespace}
    host_test_data = {"name": "power_data_test", "namespace": namespace}

    # guest_train_data = {"name": "gover_data_train_3", "namespace": namespace}
    # host_train_data = {"name": "power_data_train_3", "namespace": namespace}

    # guest_test_data = {"name": "gover_data_test_3", "namespace": namespace}
    # host_test_data = {"name": "power_data_test_3", "namespace": namespace}
    # obtain config
    if isinstance(config, str):
        config = load_job_config(config)

    parties = config.parties
    guest = parties.guest[0]
    host = parties.host[0]
    arbiter = parties.arbiter[0]
    pipeline = PipeLine().set_initiator(role='guest', party_id=guest).set_roles(guest=guest, host=host, arbiter=arbiter)

    reader_0 = Reader(name="reader_0")
    reader_1 = Reader(name="reader_1")
    reader_0.get_party_instance(
        role='guest', party_id=guest).component_param(table=guest_train_data)
    reader_1.get_party_instance(
        role='guest', party_id=guest).component_param(table=guest_test_data)
    reader_0.get_party_instance(
        role='host', party_id=host).component_param(table=host_train_data)
    reader_1.get_party_instance(
        role='host', party_id=host).component_param(table=host_test_data)

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
    intersection_0 = Intersection(name="intersection_0")
    intersection_1 = Intersection(name="intersection_1")

    # feature scale
    scale_train_0 = FeatureScale(name="scale_train_0")
    scale_train_1 = FeatureScale(name="scale_train_1")

    hetero_linr_0 = HeteroLinR(name="hetero_linr_0",
                               penalty="L2",
                               optimizer="sgd",
                               tol=0.001,
                               alpha=0.01,
                               max_iter=10,
                               early_stop="weight_diff",
                               batch_size=-1,
                               learning_rate=0.3,
                               decay=0.0,
                               decay_sqrt=False,
                               init_param={"init_method": "zeros"})
    evaluation_0 = Evaluation(name='evaluation_0', eval_type='regression')

    pipeline.add_component(reader_0)
    pipeline.add_component(reader_1)
    pipeline.add_component(data_transform_0, data=Data(data=reader_0.output.data))
    pipeline.add_component(data_transform_1,
                           data=Data(data=reader_1.output.data),
                           model=Model(data_transform_0.output.model))
    pipeline.add_component(intersection_0,
                           data=Data(data=data_transform_0.output.data))
    pipeline.add_component(intersection_1,
                           data=Data(data=data_transform_1.output.data))

    pipeline.add_component(scale_train_0,
                           data=Data(data=intersection_0.output.data))
    pipeline.add_component(scale_train_1,
                           data=Data(data=intersection_1.output.data),
                           model=Model(scale_train_0.output.model))
    pipeline.add_component(
        hetero_linr_0,
        data=Data(train_data=scale_train_0.output.data,
                  validate_data=scale_train_1.output.data))
    pipeline.add_component(evaluation_0,
                           data=Data(data=hetero_linr_0.output.data))

    pipeline.compile()

    pipeline.fit()

    print("fitting hetero linR done, result:")
    summ = pipeline.get_component("hetero_linr_0").get_summary()
    print(summ)

    # save
    pipeline.dump(f"output/model/hetero_linr.pkl")

    # from pipeline.backend.pipeline import PineLine


    # PipeLine.load_model_from_file("pipeline_saved.pkl")

    # predict
    # deploy required components
    pipeline.deploy_component([
        data_transform_0, intersection_0, scale_train_0, hetero_linr_0,
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
        "hetero_linr_0").get_output_data()
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
