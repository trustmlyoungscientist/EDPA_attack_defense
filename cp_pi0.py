import os

import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import torchvision.transforms as transforms
from accelerate import PartialState
import torch
import random
from torchvision.utils import save_image
import tqdm
import numpy as np
from openpi.training import config as _config
from openpi.models import tokenizer as _tokenizer

import draccus

from utils.data_utils import RLDSDataLoader
# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"

@dataclass
class FinetuneConfig:
    model_family: str = "pi0"                                       # Model family (e.g., `openvla`)
    vla_path: str = "checkpoints/pi0_libero/pytorch"                    # Path to OpenVLA model (on HuggingFace Hub)
    config_name: str = ""                                              # openpi TrainConfig name; empty -> pi0_libero / pi05_libero from model_family
    
    # Directory Paths
    data_root_dir: Path = Path("dataset/modified_libero_rlds")      # Path to Open-X dataset directory
    dataset_name: str  = (
        "libero_object_no_noops",
    )                    
    run_root_dir: Path = Path("runs")                               # Path to directory to store logs & checkpoints
    adapter_tmp_dir: Path = Path("adapter-tmp")                     # Temporary directory for LoRA weights before fusing
    local_log_dir: str = "./logs"

    # Attack Configuration
    image_size: int = 224                                           # Image size (e.g., 224 for 224x224 images)
    perturbation_ratio: float = 0.05                                # Ratio of perturbation to apply (e.g., 0.1 for 10% perturbation)
    alpha: float = 0.2                                              # Alpha value for perturbation blending
    max_steps: int = 1200                                           # Maximum number of perturbation steps
    iterations: int = 1                                             # Number of perturbation iterations per step
    step_size: float = 2 / 255                                      # Step size for perturbation updates
    save_path: str = "perturbation/pi0"                             # Path to save perturbations
    verbose: bool = True                                            # Whether to print verbose output during training

    # Fine-tuning Parameters
    batch_size: int = 2                                             # Fine-tuning batch size
    save_steps: int = 10                                            # Interval for checkpoint saving
    grad_accumulation_steps: int = 1                                # Gradient accumulation steps
    image_aug: bool = True                                          # Whether to train with image augmentations
    shuffle_buffer_size: int = 2000                                 # Dataloader shuffle buffer size (can reduce if OOM)

    # Tracking Parameters
    experiment: bool = False                                        # Whether to run the experiment
    use_wandb: bool = True                                          # Whether to use Weights & Biases for tracking
    wandb_project: str = ""                                         # Name of W&B project to log to (use default!)
    wandb_entity: str = ""                                          # Name of entity to log under
    run_id_note: Optional[str] = None                               # Extra note for logging, Weights & Biases
    camera_view: str = "wrist"                                      # Camera view to use (e.g., `front`, `top`, `side`)


DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")


@draccus.wrap()
def train_up(cfg: FinetuneConfig) -> None:

    print("Loading the dataset")

    distributed_state = PartialState()
    torch.cuda.set_device(device_id := distributed_state.local_process_index)
    torch.cuda.empty_cache()

    # Load OpenVLA Processor and Model using HF AutoClasses

    dataloader = RLDSDataLoader(cfg=cfg)

    if cfg.config_name:
        config = _config.get_config(cfg.config_name)
    elif cfg.model_family == "pi0":
        config = _config.get_config("pi0_libero")
    elif cfg.model_family == "pi05":
        config = _config.get_config("pi05_libero")
    else:
        raise ValueError(f"Unsupported model family: {cfg.model_family}")
    
    tokenizer = _tokenizer.PaligemmaTokenizer()

    weight_path = os.path.join(cfg.vla_path, "model.safetensors")
    
    vla = config.model.load_pytorch(config, weight_path).to(dtype=torch.bfloat16, device=device_id)

    tokenizer = _tokenizer.PaligemmaTokenizer()
    os.makedirs(f"{cfg.save_path}-{cfg.perturbation_ratio}", exist_ok=True)

    from VLAAttacker.pytorch.EDPA import EDPA
    attacker = EDPA(cfg)

    patch_size = int(np.sqrt(cfg.image_size ** 2 * cfg.perturbation_ratio))
    perturbation = torch.zeros((3, patch_size, patch_size), dtype=torch.float32).uniform_(0, 1)

    if cfg.use_wandb:
        import wandb
        exp_id = (
            f"{cfg.vla_path.split('/')[-1]}+{cfg.dataset_name}"
            f"+ratio-{cfg.perturbation_ratio}"
        )
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=f"ft+{exp_id}")

    with tqdm.tqdm(total=cfg.max_steps, leave=False) as progress:
        for idx, batch in enumerate(dataloader):

            if cfg.camera_view == "random":
                images = [batch["image"][i] if random.random() < 0.5 else batch["wrist_image"][i] for i in range(len(batch["image"]))]
            elif cfg.camera_view == "primary":
                images = batch["image"]
            else:
                images = batch["wrist_image"]

            images = torch.stack([transforms.ToTensor()(image) for image in images])

            instructions, instruction_masks = map(
                lambda x: torch.from_numpy(np.stack(x)).long(),
                zip(*[tokenizer.tokenize(i) for i in batch["language_instruction"]])
            )

            perturbation, cost, patch_loss, align_loss = attacker.generate_one_step(vla, images, instructions, instruction_masks, perturbation, eval=True)
            
            if cfg.use_wandb:
                wandb.log({
                    "cost": cost.item(),
                    "patch_loss": patch_loss.mean(),
                    "align_loss": align_loss.mean(),
                }, step=idx)

            if idx % cfg.save_steps == 0:
                torch.save(perturbation, f"{cfg.save_path}-{cfg.perturbation_ratio}/perturbation.pt")
                save_image(perturbation, f"{cfg.save_path}-{cfg.perturbation_ratio}/perturbation.png")

            progress.update()
            torch.cuda.empty_cache()

            if idx >= cfg.max_steps:
                print("Finished perturbation generation.")
                break


if __name__ == "__main__":
    train_up()
