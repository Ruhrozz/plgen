import argparse

import lightning as L
import torch
from clearml import Task
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import RichProgressBar
from omegaconf import DictConfig
from omegaconf import OmegaConf

# from plgen.ema import EMACallback
from plgen.dataset.datamodule import get_datamodule

# from plgen.pruner import ENOTPruning
from plgen.stable_diffusion import StableDiffusionModule

# def get_pruning_callback(cfg):
#     example_inputs = (
#         torch.rand(1, 4, 64, 64),
#         torch.ones(1),
#         torch.rand(1, 77, 1024),
#     )
#
#     diff_thresh = cfg.pruning.diff_pruning_threshold if hasattr(cfg.pruning, "diff_pruning_threshold") else 0.05
#     diff_step = cfg.pruning.diff_pruning_step if hasattr(cfg.pruning, "diff_pruning_step") else 1
#
#     return ENOTPruning(
#         model_attr=cfg.pruning.model_attr,
#         inplace=cfg.pruning.inplace,
#         acceleration=cfg.pruning.acceleration,
#         steps=cfg.pruning.steps,
#         label_selector=cfg.pruning.label_selector,
#         example_inputs=example_inputs,
#         pruning_info=cfg.pruning.pruning_info,
#         method=cfg.pruning.method,
#         diff_pruning_threshold=diff_thresh,
#         diff_pruning_step=diff_step,
#     )


def get_callbacks(cfg):
    callbacks = [
        RichProgressBar(),
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(every_n_train_steps=cfg.trainer_params.max_steps * 0.05),
        # EMACallback(decay=0.999, use_ema_weights=True),
    ]

    #     if cfg.pruning.enable:
    #         callbacks.append(get_pruning_callback(cfg))

    return callbacks


def prepare_hardware(cfg):
    torch.set_float32_matmul_precision(cfg.common.mm_precision)
    seed_everything(cfg.common.seed, workers=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to config file")
    parser.add_argument("--task", type=str, help="Task name for clearml", default="default_task")
    parser.add_argument("--project", type=str, help="Project name for clearml", default="Stable Diffusion")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task = Task.init(task_name=args.task, project_name=args.project)

    config_path = task.connect_configuration(args.config)
    cfg = OmegaConf.load(config_path)

    prepare_hardware(cfg)

    model = StableDiffusionModule(cfg)
    if cfg.common.ckpt_path:
        model = model.load_from_checkpoint(cfg.common.ckpt_path)

    trainer = L.Trainer(
        callbacks=get_callbacks(cfg),
        **cfg.trainer_params,
    )

    trainer.fit(
        model=model,
        datamodule=get_datamodule(cfg),
    )


if __name__ == "__main__":
    main()
