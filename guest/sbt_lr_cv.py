import argparse

from pipeline.backend.pipeline import PipeLine
from pipeline.component import DataTransform
from pipeline.component import HeteroSecureBoost
from pipeline.component import Intersection
from pipeline.component import Reader
from pipeline.interface import Data
from pipeline.component import FeatureScale
from pipeline.component import Evaluation

from pipeline.utils.tools import load_job_config


def main(config="./config.yaml", namespace="seed2022"):
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
    pipeline = PipeLine().set_initiator(
        role="guest", party_id=guest).set_roles(guest=guest,
                                                host=host,
                                                arbiter=arbiter)

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

    # feature scale
    scale_train_0 = FeatureScale(name="scale_train_0")

    # secure boost component
    hetero_secure_boost_0 = HeteroSecureBoost(
        name="hetero_secure_boost_0",
        num_trees=6,
        task_type="regression",
        objective_param={"objective": "lse"},
        encrypt_param={"method": "Paillier"},
        tree_param={"max_depth": 6},
        validation_freqs=1,
        cv_param={
            "need_cv": True,
            "n_splits": 5,
            "shuffle": False,
            "random_seed": 3
        })

    # evaluation
    evaluation_0 = Evaluation(name='evaluation_0', eval_type='regression')

    pipeline.add_component(reader_0)
    pipeline.add_component(data_transform_0,
                           data=Data(data=reader_0.output.data))
    pipeline.add_component(intersect_0,
                           data=Data(data=data_transform_0.output.data))
    pipeline.add_component(scale_train_0,
                           data=Data(data=intersect_0.output.data))
    pipeline.add_component(hetero_secure_boost_0,
                           data=Data(train_data=scale_train_0.output.data))

    pipeline.add_component(evaluation_0,
                           data=Data(data=hetero_secure_boost_0.output.data))

    pipeline.compile()
    pipeline.fit()

    print("fitting hetero secureboost done, result:")
    print(pipeline.get_component("hetero_secure_boost_0").get_summary())


if __name__ == "__main__":
    parser = argparse.ArgumentParser("PIPELINE DEMO")
    parser.add_argument("-config", type=str, help="config file")
    args = parser.parse_args()
    if args.config is not None:
        main(args.config)
    else:
        main()