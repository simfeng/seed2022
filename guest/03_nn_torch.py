import argparse

from collections import OrderedDict
from pipeline.backend.pipeline import PipeLine
from pipeline.component import DataTransform
from pipeline.component import HeteroNN
from pipeline.component import Intersection
from pipeline.component import Reader
from pipeline.component import Evaluation
from pipeline.interface import Data
from pipeline.utils.tools import load_job_config
from pipeline.interface import Model

from pipeline import fate_torch_hook
import torch as t
from torch import nn
from torch.nn import init
from torch import optim
from pipeline import fate_torch as ft

# this is important, modify torch modules so that Sequential model be parsed by pipeline
fate_torch_hook(t)


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

    guest_train_data = {"name": "gover_data_train_3_all_one_hot", "namespace": namespace}
    host_train_data = {"name": "power_data_train_3_all_one_hot", "namespace": namespace}

    guest_test_data = {"name": "gover_data_test_3_all_one_hot", "namespace": namespace}
    host_test_data = {"name": "power_data_test_3_all_one_hot", "namespace": namespace}

    gover_in_feature = 318
    power_in_feature = 74

    pipeline = PipeLine().set_initiator(
        role='guest', party_id=guest).set_roles(guest=guest,
                                                host=host,
                                                arbiter=arbiter)

    reader_0 = Reader(name="reader_0")
    reader_0.get_party_instance(
        role='guest', party_id=guest).component_param(table=guest_train_data)
    reader_0.get_party_instance(
        role='host', party_id=host).component_param(table=host_train_data)

    data_transform_0 = DataTransform(name="data_transform_0")
    data_transform_0.get_party_instance(role='guest',
                                        party_id=guest).component_param(
                                            with_label=True,
                                            output_format="dense",
                                            label_name="sjje_per_month",
                                            label_type="float")
    data_transform_0.get_party_instance(
        role='host', party_id=host).component_param(with_label=False)

    intersection_0 = Intersection(name="intersection_0")

    # define network structure in torch style #
    # define guest model
    guest_bottom_a = nn.Linear(gover_in_feature, 256, True)
    guest_bottom_b = nn.Linear(256, 128, True)
    seq = t.nn.Sequential(
        OrderedDict([
            ('layer_0', guest_bottom_a),
            ('relu_0', nn.ReLU()),
            ('layer_1', guest_bottom_b),
            ('relu_1', nn.ReLU())
        ])
    )

    guest_top_layer_0 = nn.Linear(64, 32, False)
    guest_top_layer_1 = nn.Linear(32, 1, True)
    seq2 = t.nn.Sequential(
        guest_top_layer_0,
        guest_top_layer_1
        # nn.Sigmoid()
    )

    # define host model
    host_bottom_model = nn.Sequential(nn.Linear(power_in_feature, 256, True),
                                      nn.ReLU(), nn.Linear(256, 128, True),
                                      nn.ReLU())

    interactive_layer = nn.Linear(128, 64, True)

    # init model weights, init funcs are modified, so it can initialize whole sequential
    init.xavier_normal_(seq)
    init.normal_(interactive_layer)
    init.xavier_normal_(seq2)

    # loss function
    ce_loss_fn = nn.MSELoss()

    # optimizer, after fate torch hook optimizer can be created without parameters
    opt: ft.optim.Adam = optim.Adam(lr=0.1)

    hetero_nn_0 = HeteroNN(name="hetero_nn_0",
                           task_type='regression',
                           epochs=2,
                           floating_point_precision=23,
                           interactive_layer_lr=0.1,
                           batch_size=120,
                           early_stop="diff")
    guest_nn_0 = hetero_nn_0.get_party_instance(role='guest', party_id=guest)
    guest_nn_0.add_bottom_model(seq)
    guest_nn_0.add_top_model(seq2)
    guest_nn_0.set_interactve_layer(interactive_layer)
    host_nn_0 = hetero_nn_0.get_party_instance(role='host', party_id=host)
    host_nn_0.add_bottom_model(host_bottom_model)
    # compile model with torch optimizer
    hetero_nn_0.compile(opt, loss=ce_loss_fn)

    hetero_nn_1 = HeteroNN(name="hetero_nn_1")
    evaluation_0 = Evaluation(name="evaluation_0", eval_type="regression")

    pipeline.add_component(reader_0)
    pipeline.add_component(data_transform_0, data=Data(data=reader_0.output.data))
    pipeline.add_component(intersection_0, data=Data(data=data_transform_0.output.data))
    pipeline.add_component(hetero_nn_0, data=Data(train_data=intersection_0.output.data))
    pipeline.add_component(hetero_nn_1, data=Data(test_data=intersection_0.output.data),
                           model=Model(model=hetero_nn_0.output.model))
    pipeline.add_component(evaluation_0, data=Data(data=hetero_nn_0.output.data))
    pipeline.compile()
    pipeline.fit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("PIPELINE DEMO")
    parser.add_argument("-config", type=str,
                        help="config file")
    args = parser.parse_args()
    if args.config is not None:
        main(args.config)
    else:
        main()
