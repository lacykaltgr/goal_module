from models.gmc import DummyGoalGMC
from training.dca_evaluation_trainer import DCAEvaluator
from data_modules.class_dataset import *


def setup_model(scenario, model, model_config, scenario_config, data_module=None):
    if model == "gmc":
        if scenario == "dummy_goal":
            return DummyGoalGMC(
                name=model_config["model"],
                common_dim=model_config["common_dim"],
                latent_dim=model_config["latent_dim"],
                loss_type=model_config["loss_type"],
            )
        else:
            raise ValueError(
                "[Model Setup] Selected scenario not yet implemented for GMC model: "
                + str(scenario)
            )

    else:
        raise ValueError(
            "[Model Setup] Selected model not yet implemented: " + str(model)
        )


def setup_data_module(scenario, experiment_config, scenario_config, train_config):
    if experiment_config["stage"] == "evaluate_dca":
        return DCADataModule(
            dataset=scenario,
            data_dir=scenario_config["data_dir"],
            data_config=train_config,
        )
    else:
        return ClassificationDataModule(
            dataset=scenario,
            data_dir=scenario_config["data_dir"],
            data_config=train_config,
        )


def setup_dca_evaluation_trainer(model, machine_path, scenario, config):
    return DCAEvaluator(
        model=model,
        scenario=scenario,
        machine_path=machine_path,
        minimum_cluster_size=config["minimum_cluster_size"],
        unique_modality_idxs=config["unique_modality_idxs"],
        unique_modality_dims=config["unique_modality_dims"],
        partial_modalities_idxs=config["partial_modalities_idxs"],
    )



"""

Loading functions

"""



def load_model(config, model_file):

    model = setup_model(
        scenario=config["experiment"]["scenario"],
        model=config["experiment"]["model"],
        model_config=config["model"],
        scenario_config=config["scenario"],
    )

    checkpoint = torch.load(model_file)
    model.load_state_dict(checkpoint["state_dict"])

    # Freeze model
    model.freeze()

    return model



"""


General functions

"""


def flatten_dict(dd, separator="_", prefix=""):
    return (
        {
            prefix + separator + k if prefix else k: v
            for kk, vv in dd.items()
            for k, v in flatten_dict(vv, separator, kk).items()
        }
        if isinstance(dd, dict)
        else {prefix: dd}
    )

