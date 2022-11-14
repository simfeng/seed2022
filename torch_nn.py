import argparse

import os
from pipeline.component import Reader
from pipeline.component import DataTransform
from pipeline.backend.pipeline import PipeLine
from pipeline.component import Intersection
from pipeline.component import HeteroNN
from pipeline.component import Evaluation
from pipeline.component import HeteroDataSplit
from pipeline.interface import Data
from pipeline.interface import Model
from pipeline import fate_torch_hook

import torch as t
from torch import nn
from torch.nn import init
from torch import optim
from pipeline import fate_torch as ft

# this is important, modify torch modules so that Sequential model be parsed by pipeline
fate_torch_hook(t)
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
    guest_train_data = {"name": "gover_data", "namespace": namespace}
    host_train_data = {"name": "power_data", "namespace": namespace}

    # init pipeline
    pipeline = PipeLine().set_initiator(role="guest",
                                        party_id=guest).set_roles(
                                            guest=guest,
                                            host=host,
                                            arbiter=arbiter
                                        )

    # set data reader and data-io

    reader_0 = Reader(name="reader_0")
    reader_0.get_party_instance(
        role="guest", party_id=guest).component_param(table=guest_train_data)
    reader_0.get_party_instance(
        role="host", party_id=host).component_param(table=host_train_data)


    data_transform_0 = DataTransform(name="data_transform_0")

    data_transform_0.get_party_instance(role="guest",
                                        party_id=guest).component_param(
                                            with_label=True,
                                            output_format="dense",
                                            label_name="sjje_per_month",
                                            label_type="float")
    data_transform_0.get_party_instance(
        role="host", party_id=host).component_param(with_label=False)


    # data intersect component
    intersect_0 = Intersection(name="intersection_0")

    # data split
    hetero_data_split_0 = HeteroDataSplit(name="hetero_data_split_0",
                                          stratified=True,
                                          test_size=0.3,
                                          split_points=[0.0, 0.2])



    from collections import OrderedDict
    # guest bottom model
    guest_bottom_a = nn.Linear(4, 10, True)
    guest_bottom_b = nn.Linear(10, 8, True)
    guest_bottom_seq = t.nn.Sequential(
        OrderedDict([
            ('layer_0', guest_bottom_a),
            ('relu_0', nn.ReLU()),
            ('layer_1', guest_bottom_b),
            ('relu_1', nn.ReLU())
        ])
    )

    # host bottom model
    host_bottom_model = nn.Sequential(
        nn.Linear(4, 16, True),
        nn.ReLU(),
        nn.Linear(16, 8, True),
        nn.ReLU()
    )

    # interactive layer
    interactive_layer = nn.Linear(8, 4, True)

    # guest top model
    guest_top_seq = t.nn.Sequential(
        nn.Linear(4, 1, True),
        # nn.Sigmoid()
    )

    # init bottom models
    init.normal_(guest_bottom_seq)
    init.normal_(host_bottom_model)
    init.normal_(guest_bottom_seq, init='bias')
    # init interactive layer
    init.normal_(interactive_layer, init='weight')
    init.constant_(interactive_layer, val=0, init='bias')
    # init top model
    init.xavier_normal_(guest_top_seq)

    # loss function
    ce_loss_fn = nn.MSELoss()
    # optimizer, after fate torch hook optimizer can be created without parameters
    opt = optim.Adam(lr=0.01)

    # make HeteroNN components and define some parameters
    hetero_nn_0 = HeteroNN(name="hetero_nn_0",
                           epochs=5,
                           floating_point_precision=None,
                           interactive_layer_lr=0.01,
                           batch_size=-1,
                           early_stop="diff")

    # add sub-component to hetero_nn_0
    guest_nn_0 = hetero_nn_0.get_party_instance(role='guest', party_id=guest)
    guest_nn_0.add_bottom_model(guest_bottom_seq)
    guest_nn_0.add_top_model(guest_top_seq)
    guest_nn_0.set_interactve_layer(interactive_layer)

    host_nn_0 = hetero_nn_0.get_party_instance(role='host', party_id=host)
    host_nn_0.add_bottom_model(host_bottom_model)
    # compile model with torch optimizer
    hetero_nn_0.compile(opt, loss=ce_loss_fn)

    evaluation_0 = Evaluation(name="evaluation_0", eval_type="regression")

    hetero_nn_1 = HeteroNN(name="hetero_nn_1")

    pipeline.add_component(reader_0)
    pipeline.add_component(data_transform_0, data=Data(data=reader_0.output.data))
    pipeline.add_component(intersect_0, data=Data(data=data_transform_0.output.data))
    if 0:
        pipeline.add_component(hetero_data_split_0,
                            data=Data(data=intersect_0.output.data))
        pipeline.add_component(
            hetero_nn_0,
            data=Data(train_data=hetero_data_split_0.output.data.train_data))
        pipeline.add_component(
            hetero_nn_1,
            data=Data(test_data=hetero_data_split_0.output.data.train_data),
            model=Model(model=hetero_nn_0.output.model))
    else:
        pipeline.add_component(hetero_nn_0,
                               data=Data(train_data=intersect_0.output.data))
        pipeline.add_component(hetero_nn_1,
                               data=Data(test_data=intersect_0.output.data),
                               model=Model(model=hetero_nn_0.output.model))
    pipeline.add_component(evaluation_0, data=Data(data=hetero_nn_0.output.data))
    pipeline.compile()

    pipeline.fit()

    print("fitting hetero secureboost done, result:")
    summ = pipeline.get_component("hetero_secure_boost_0").get_summary()
    print(summ)

    # save
    pipeline.dump(f"output/model/torch_nn.pkl")

    # from pipeline.backend.pipeline import PineLine


    # PipeLine.load_model_from_file("pipeline_saved.pkl")

    # predict
    # deploy required components
    pipeline.deploy_component(
        [data_transform_0, intersect_0, hetero_nn_0, evaluation_0])

    predict_pipeline = PipeLine()
    # add data reader onto predict pipeline
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
    predict_result.to_csv('output/result.csv')


if __name__ == "__main__":
    parser = argparse.ArgumentParser("PIPELINE DEMO")
    parser.add_argument("-config", type=str, help="config file")
    args = parser.parse_args()
    if args.config is not None:
        main(args.config)
    else:
        main()
