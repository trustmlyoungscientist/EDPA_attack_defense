import os

import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from openpi.training import config
from openpi.shared import download
import jax.numpy as jnp

from openpi.models import model as _model
from openpi.models import tokenizer as _tokenizer

import draccus

from utils.data_utils import RLDSDataLoader
# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class FinetuneConfig:
    model_family: str = "pi0"                                       # Model family (e.g., `openvla`)
    vla_path: str = "s3://openpi-assets/checkpoints/pi0_fast_libero"# Path to OpenVLA model (on HuggingFace Hub)
    
    # Directory Paths
    data_root_dir: Path = Path("dataset/modified_libero_rlds")      # Path to Open-X dataset directory
    dataset_name: str = "libero_spatial_no_noops"                   # Name of fine-tuning dataset (e.g., `droid_wipe`)
    run_root_dir: Path = Path("runs")                               # Path to directory to store logs & checkpoints
    adapter_tmp_dir: Path = Path("adapter-tmp")                     # Temporary directory for LoRA weights before fusing
    local_log_dir: str = "./logs"

    # Attack Configuration
    image_size: int = 224                                           # Image size (e.g., 224 for 224x224 images)
    perturbation_ratio: float = 0.04                                # Ratio of perturbation to apply (e.g., 0.1 for 10% perturbation)
    alpha: float = 0.8                                              # Alpha value for perturbation blending
    max_steps: int = 50000                                          # Maximum number of perturbation steps
    iterations: int = 1                                             # Number of perturbation iterations per step
    step_size: float = 2 / 255                                      # Step size for perturbation updates
    save_path: str = ""                                             # Path to save perturbations
    verbose: bool = True                                            # Whether to print verbose output during training

    # Fine-tuning Parameters
    batch_size: int = 2                                             # Fine-tuning batch size
    save_steps: int = 10                                            # Interval for checkpoint saving
    grad_accumulation_steps: int = 1                                # Gradient accumulation steps
    image_aug: bool = True                                          # Whether to train with image augmentations
    shuffle_buffer_size: int = 2000                                 # Dataloader shuffle buffer size (can reduce if OOM)

    # Tracking Parameters
    experiment: bool = False                                        # Whether to run the experiment
    use_wandb: bool = False                                         # Whether to use Weights & Biases for tracking
    wandb_project: str = ""                                         # Name of W&B project to log to (use default!)
    wandb_entity: str = ""                                          # Name of entity to log under
    run_id_note: Optional[str] = None                               # Extra note for logging, Weights & Biases
    camera_view: str = "primary"                                    # Camera view to use (e.g., `front`, `top`, `side`)


DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")


@draccus.wrap()
def train_up(cfg: FinetuneConfig) -> None:

    print("Loading the dataset")

    # Load OpenVLA Processor and Model using HF AutoClasses

    dataloader = RLDSDataLoader(cfg=cfg)
    checkpoint_dir = download.maybe_download(cfg.vla_path)
    train_config = config.get_config("pi0_fast_libero")


    vla = train_config.model.load(_model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16))
    tokenizer = _tokenizer.PaligemmaTokenizer()
    os.makedirs(f"{cfg.save_path}-{cfg.perturbation_ratio}", exist_ok=True)

    from VLAAttacker.jax.EDPA import EDPA
    attacker = EDPA(cfg)

    attacker.generate(vla, dataloader, tokenizer)



if __name__ == "__main__":
    train_up()
