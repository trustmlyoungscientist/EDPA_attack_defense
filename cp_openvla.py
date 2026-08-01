import os
import torch
import torchvision.transforms as transforms
import numpy as np
import tqdm
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from accelerate import PartialState
import time
import draccus
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from transformers import AutoModelForVision2Seq, AutoProcessor
from transformers import AutoConfig, AutoImageProcessor

from openvla.prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from openvla.prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from openvla.prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

from utils.data_utils import RLDSDataLoader
from utils.openvla import build_prompt
# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class FinetuneConfig:
    # fmt: off
    model_family: str = "openvla"                                     # Path to OpenVLA model (on HuggingFace Hub)
    vla_path: str = "openvla/openvla-7b-finetuned-libero-spatial"      # Path to OpenVLA model (on HuggingFace Hub)
    # Directory Paths
    data_root_dir: Path = Path("dataset/modified_libero_rlds")        # Path to Open-X dataset directory
    dataset_name: str = "libero_spatial_no_noops"                     # Name of fine-tuning dataset (e.g., `droid_wipe`)
    run_root_dir: Path = Path("runs")                                 # Path to directory to store logs & checkpoints
    adapter_tmp_dir: Path = Path("adapter-tmp")                       # Temporary directory for LoRA weights before fusing
    local_log_dir: str = "./logs"

    # Attack Configuration
    image_size: int = 224                                           # Image size (e.g., 224 for 224x224 images)
    perturbation_ratio: float = 0.05                                 # Ratio of perturbation to apply (e.g., 0.1 for 10% perturbation)
    alpha: float = 0.2                                              # Alpha value for perturbation blending
    max_steps: int = 1200                                           # Maximum number of perturbation steps
    iterations: int = 1                                             # Number of perturbation iterations per step
    step_size: float = 2 / 255                                      # Step size for perturbation updates
    save_path: str = "perturbation/openvla-uoa"                     # Path to save perturbations
    verbose: bool = True                                            # Whether to print verbose output during training

    # Fine-tuning Parameters
    batch_size: int = 2                                             # Fine-tuning batch size
    save_steps: int = 10                                            # Interval for checkpoint saving
    grad_accumulation_steps: int = 1                                # Gradient accumulation steps
    shuffle_buffer_size: int = 2000                                 # Dataloader shuffle buffer size (can reduce if OOM)

    # Tracking Parameters
    experiment: bool = False                                        # Whether to run the experiment
    use_wandb: bool = True                                          # Whether to use Weights & Biases for tracking
    wandb_project: str = "openvla"                                  # Name of W&B project to log to (use default!)
    wandb_entity: str = ""                                          # Name of entity to log under
    run_id_note: Optional[str] = None                               # Extra note for logging, Weights & Biases
    camera_view: str = "primary"                                    # Camera view to use (e.g., `front`, `top`, `side`)


DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")


@draccus.wrap()
def train_up(cfg: FinetuneConfig) -> None:
    
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    distributed_state = PartialState()
    torch.cuda.set_device(device_id := distributed_state.local_process_index)
    torch.cuda.empty_cache()

    # Load OpenVLA Processor and Model using HF AutoClasses
    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16,
        quantization_config=None,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).to(device_id)

    dataloader = RLDSDataLoader(cfg=cfg)

    os.makedirs(f"{cfg.save_path}-{cfg.perturbation_ratio}", exist_ok=True)

    from VLAAttacker.pytorch.EDPA import EDPA
    attacker = EDPA(cfg, device_id=device_id)
    
    patch_size = int(np.sqrt(cfg.image_size ** 2 * cfg.perturbation_ratio))
    perturbation = torch.zeros((3, patch_size, patch_size), dtype=torch.float32).uniform_(0, 1)

    if cfg.use_wandb:
        import wandb
        exp_id = (
            f"{cfg.vla_path.split('/')[-1]}+{cfg.dataset_name}"
            f"+ratio-{cfg.perturbation_ratio}"
            f"+iter-{cfg.iterations}"        
        )
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=f"ft+{exp_id}")

    with tqdm.tqdm(total=cfg.max_steps, leave=False) as progress:
        for idx, batch in enumerate(dataloader):

            if cfg.camera_view == "random":
                p, w = batch["image"], batch["wrist_image"]
                m = torch.rand(p.shape[0], device=p.device) < 0.5
                while m.dim() < p.dim():
                    m = m.unsqueeze(-1)
                images = torch.where(m, w, p)
            elif cfg.camera_view == "primary":
                images = batch["image"]
            else:
                images = batch["wrist_image"]

            images = torch.stack([transforms.ToTensor()(image) for image in images])
            prompt = build_prompt(batch["language_instruction"])

            instructions, instructions_masks = ( 
                 processor.tokenizer(prompt, padding=True, return_tensors="pt")[k] for k in ("input_ids", "attention_mask") 
            )

            perturbation, cost, patch_loss, align_loss = attacker.generate_one_step(
                vla, images, instructions, instructions_masks, perturbation,
            )

            if cfg.use_wandb:
                wandb.log({
                    "cost": cost.item(),
                    "patch_loss": patch_loss.mean(),
                    "align_loss": align_loss.mean(),
                }, step=idx)

            if idx % cfg.save_steps == 0:
                torch.save(perturbation, f"{cfg.save_path}-{cfg.perturbation_ratio}/perturbation.pt")
                save_image(perturbation, f"{cfg.save_path}-{cfg.perturbation_ratio}/perturbation.png")

            if idx >= cfg.max_steps:
                break

            progress.update()
            torch.cuda.empty_cache()

if __name__ == "__main__":
    train_up()
