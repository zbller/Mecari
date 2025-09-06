#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

# Disable tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import random
from datetime import datetime
from importlib import import_module
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

from mecari.config.config import get_model_config, override_config, save_config
from mecari.data.data_module import DataModule


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed value.
        deterministic: If True, enforce deterministic behavior (slower).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    pl.seed_everything(seed)


def get_config_sections(config: dict) -> dict:
    """Extract structured sections from a unified config dict."""
    return {
        "model": config["model"],
        "training": config["training"],
        "features": config.get("features", {}),
        "edge": config.get("edge_features", {}),
    }


def calculate_feature_dim(config: dict) -> int:
    """Return feature dimension from config (lexical features by default)."""
    features_cfg = config.get("features", {})

    lexical_dim = features_cfg.get("lexical_feature_dim", 100000)
    return lexical_dim


def create_data_module(config: dict) -> DataModule:
    """Create DataModule from config (lexical-only pipeline)."""
    features_cfg = config.get("features", {})
    training_cfg = config["training"]
    edge_cfg = config.get("edge_features", {})

    lexical_feature_dim = features_cfg.get("lexical_feature_dim", 100000)

    return DataModule(
        annotations_dir=training_cfg["annotations_dir"],
        batch_size=training_cfg["batch_size"],
        num_workers=training_cfg["num_workers"],
        max_files=training_cfg.get("max_files"),
        use_bidirectional_edges=edge_cfg.get("use_bidirectional_edges", True),
        annotations_override_dir=training_cfg.get("annotations_override_dir"),
        lexical_feature_dim=lexical_feature_dim,
    )


def setup_loggers(config: dict, experiment_name: str):
    """Configure optional loggers (e.g., Weights & Biases)."""
    import subprocess

    from pytorch_lightning.loggers import WandbLogger

    loggers = []

    if config["training"]["use_wandb"]:
        try:
            tags = []
            try:
                branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True).strip()
                tags.append(f"branch:{branch}")
                commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
                tags.append(f"commit:{commit}")
            except:
                pass

            wandb_logger = WandbLogger(
                project=config["training"]["project_name"],
                name=experiment_name,
                save_dir=f"experiments/{experiment_name}",
                save_code=True,
                log_model=False,
                tags=tags,
            )
            loggers.append(wandb_logger)
            print("✓ Added WandB logger (metrics only)")
        except Exception as e:
            print(f"WandbLogger initialization error: {e}")
    else:
        print("WandB logging disabled")

    if not loggers:
        loggers = False

    return loggers


def create_trainer(config: dict, callbacks: list, loggers, deterministic: bool) -> pl.Trainer:
    """Create a PyTorch Lightning Trainer."""
    if torch.cuda.is_available():
        accelerator = "gpu"
        devices = 1
    else:
        accelerator = "cpu"
        devices = 1

    max_steps = config["training"].get("max_steps", 8600)
    max_epochs = -1  # use max_steps only

    trainer_kwargs = {
        "max_epochs": max_epochs,
        "max_steps": max_steps,
        "callbacks": callbacks,
        "logger": loggers,
        "accelerator": accelerator,
        "devices": devices,
        "log_every_n_steps": config["training"]["log_every_n_steps"],
        "val_check_interval": config["training"]["val_check_interval"],
        "gradient_clip_val": config["training"]["gradient_clip_val"],
        "enable_checkpointing": True,
        "enable_progress_bar": True,
        "limit_train_batches": 1.0,
        "limit_val_batches": 1.0,
        "limit_test_batches": 1.0,
        "limit_predict_batches": 1.0,
        "fast_dev_run": False,
        "deterministic": deterministic,
        "benchmark": not deterministic,
        "precision": "16-mixed",
    }

    if "gradient_clip_algorithm" in config["training"]:
        trainer_kwargs["gradient_clip_algorithm"] = config["training"]["gradient_clip_algorithm"]

    if "accumulate_grad_batches" in config["training"]:
        trainer_kwargs["accumulate_grad_batches"] = config["training"]["accumulate_grad_batches"]

    return pl.Trainer(**trainer_kwargs)


def create_model_and_datamodule(config: dict, feature_dim: int, data_module: Optional[DataModule] = None):
    """Create model and ensure DataModule is available (lexical-only)."""
    cfg = get_config_sections(config)
    model_cfg = cfg["model"]
    training_cfg = cfg["training"]
    features_cfg = cfg["features"]

    if data_module is None:
        data_module = create_data_module(config)

    common_params = {
        "hidden_dim": model_cfg["hidden_dim"],
        "num_classes": model_cfg["num_classes"],
        "learning_rate": training_cfg["learning_rate"],
        "lexical_feature_dim": features_cfg.get("lexical_feature_dim", 100000),
    }

    if model_cfg["type"] == "gatv2":
        MecariGATv2 = getattr(import_module("mecari.models.gatv2"), "MecariGATv2")
        model = MecariGATv2(
            **common_params,
            num_heads=model_cfg["num_heads"],
            share_weights=model_cfg.get("share_weights", False),
            dropout=model_cfg.get("dropout", 0.1),
            attn_dropout=model_cfg.get("attn_dropout", model_cfg.get("attention_dropout", 0.1)),
            add_self_loops_flag=model_cfg.get("add_self_loops", True),
            edge_dropout=model_cfg.get("edge_dropout", 0.0),
            norm=model_cfg.get("norm", "layer"),
        )
    else:
        raise ValueError(f"Unsupported model type: {model_cfg['type']}")

    return model, data_module


def main():
    parser = argparse.ArgumentParser(description="Train the morphological analysis model")
    parser.add_argument(
        "--model",
        "-m",
        choices=["gatv2"],
        default="gatv2",
        help="Model type (only gatv2 supported). If a config is provided, config.model.type takes precedence.",
    )
    parser.add_argument("--config", "-c", help="Path to config file (overrides model type if present)")
    parser.add_argument("--batch-size", "-b", type=int, help="Batch size")
    parser.add_argument("--steps", "-s", type=int, help="Max training steps")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--hidden-dim", type=int, help="Hidden dimension size")
    parser.add_argument("--patience", type=int, help="Early stopping patience")
    parser.add_argument("--weight-decay", type=float, help="Weight decay")
    parser.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases logging")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--no-deterministic", action="store_true", help="Disable deterministic mode for speed")
    parser.add_argument("--resume", type=str, help="Experiment name to resume (e.g., gatv2_20250806_162945)")
    args = parser.parse_args()

    # Load/merge config
    if args.config:
        from mecari.config.config import load_config

        config = load_config(args.config)
        if "model" in config and "type" in config["model"]:
            args.model = config["model"]["type"]
    else:
        config = get_model_config(args.model)

    overrides = {}

    # Training overrides
    training_overrides = {}
    if args.batch_size:
        training_overrides["batch_size"] = args.batch_size
    if args.steps:
        training_overrides["max_steps"] = args.steps
    if args.lr:
        training_overrides["learning_rate"] = args.lr
    if args.no_wandb:
        training_overrides["use_wandb"] = False
    if args.patience:
        training_overrides["patience"] = args.patience
    if args.seed:
        training_overrides["seed"] = args.seed
    if args.no_deterministic:
        training_overrides["deterministic"] = False

    if training_overrides:
        overrides["training"] = training_overrides

    # Model overrides
    if args.hidden_dim:
        overrides["model"] = {"hidden_dim": args.hidden_dim}

    # Optimizer overrides
    if args.weight_decay:
        overrides.setdefault("training", {})
        overrides["training"]["optimizer"] = {"weight_decay": args.weight_decay}

    if overrides:
        config = override_config(config, overrides)

    deterministic = config["training"].get("deterministic", True)
    set_seed(config["training"]["seed"], deterministic=deterministic)

    if not deterministic:
        print("⚡ Performance mode: deterministic=False (reproducibility not guaranteed)")

    resume_from_checkpoint = None
    experiment_name = None
    if args.resume:
        experiment_path = os.path.join("experiments", args.resume)
        if os.path.exists(experiment_path):
            checkpoint_dir = os.path.join(experiment_path, "checkpoints")
            if os.path.exists(checkpoint_dir):
                checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".ckpt")]
                if checkpoints:
                    checkpoints.sort()
                    resume_from_checkpoint = os.path.join(checkpoint_dir, checkpoints[-1])
                    print(f"Resuming training from: {resume_from_checkpoint}")
                    experiment_name = args.resume

                    config_path = os.path.join(experiment_path, "config.yaml")
                    if os.path.exists(config_path):
                        from mecari.config.config import load_config

                        config = load_config(config_path)
                        print(f"Restored config from: {config_path}")
                else:
                    print(f"Warning: No checkpoints found in: {checkpoint_dir}")
            else:
                print(f"Warning: Checkpoint directory not found: {checkpoint_dir}")
        else:
            print(f"Warning: Experiment directory not found: {experiment_path}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"{config['model']['type']}_{timestamp}"

    print(f"Experiment: {experiment_name}")
    print(f"Model: {config['model']['type'].upper()}")
    print("Lexical features: enabled (default)")

    if torch.cuda.is_available():
        print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("💻 Using CPU")

    data_module = create_data_module(config)

    feature_dim = calculate_feature_dim(config)

    model, _ = create_model_and_datamodule(config, feature_dim, data_module)

    # Attach training config for schedulers, etc.
    model.training_config = config["training"]

    experiment_dir = f"experiments/{experiment_name}"
    if not args.resume:
        os.makedirs(experiment_dir, exist_ok=True)
        save_config(config, f"{experiment_dir}/config.yaml")

    checkpoint_callback_error = ModelCheckpoint(
        dirpath=f"experiments/{experiment_name}/checkpoints",
        filename=f"{config['model']['type']}-{{epoch:02d}}-{{val_error_epoch:.3f}}",
        monitor="val_error_epoch",
        mode="min",
        save_top_k=1,
        save_last=True,
    )

    early_stopping = EarlyStopping(
        monitor="val_error_epoch", mode="min", patience=config["training"]["patience"], verbose=True, strict=False
    )

    loggers = setup_loggers(config, experiment_name)

    callbacks = [checkpoint_callback_error, early_stopping]
    try:
        if loggers:
            lr_monitor = LearningRateMonitor(logging_interval="step")
            callbacks.append(lr_monitor)
    except Exception:
        pass
    trainer = create_trainer(config, callbacks, loggers, deterministic)

    print("Starting training...")

    try:
        if resume_from_checkpoint:
            trainer.fit(model, data_module, ckpt_path=resume_from_checkpoint)
        else:
            trainer.fit(model, data_module)
        training_status = "completed"

        if data_module.test_dataset:
            print("Evaluating on test data...")
            trainer.test(model, data_module)
        print("Training complete!")
    except KeyboardInterrupt:
        print("\nTraining interrupted...")
        training_status = "interrupted"
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback

        traceback.print_exc()
        training_status = "error"

    print(f"Experiment: {experiment_name}")
    print(f"Experiment dir: experiments/{experiment_name}")

    print("\n=== Saved models ===")

    if checkpoint_callback_error.best_model_path:
        best_error = (
            float(checkpoint_callback_error.best_model_score)
            if checkpoint_callback_error.best_model_score is not None
            else 1.0
        )
        print(f"  Best val_error: {best_error:.6f}")
        print(f"    → {os.path.basename(checkpoint_callback_error.best_model_path)}")

    print(f"\nFinal epoch: {trainer.current_epoch}")
    print(f"Training status: {training_status}")


if __name__ == "__main__":
    main()
