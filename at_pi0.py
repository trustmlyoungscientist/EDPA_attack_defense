import os
import torch
import random
from torch.optim import AdamW
import torch.distributed as dist
from collections import deque
import wandb
import torchvision.transforms as transforms
import numpy as np
from accelerate import PartialState
import time
import draccus
import tqdm
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import safetensors.torch
import shutil

from openpi.training import config as _config
from openpi.models import tokenizer as _tokenizer

from utils.pi0 import image_transform
from utils.data_utils import RLDSDataLoader
# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class FinetuneConfig:
    # fmt: off
    # vla_path: str = "openvla/openvla-7b"
    model_family: str = "pi0"                                         # Path to OpenVLA model (on HuggingFace Hub)
    vla_path: str = "./checkpoints/pi0_libero/pytorch"                # Path to OpenVLA model (on HuggingFace Hub)
    config_name: str = ""                                              # openpi TrainConfig name override; empty -> pi0_libero/pi05_libero from model_family
    # Directory Paths
    data_root_dir: Path = Path("dataset/modified_libero_rlds")        # Path to Open-X dataset directory
    dataset_name: str  = (
        "libero_spatial_no_noops",
        "libero_object_no_noops",
        "libero_goal_no_noops",
        "libero_10_no_noops",
    )                    
    run_root_dir: Path = Path("runs")                                 # Path to directory to store logs & checkpoints
    local_log_dir: str = "./logs"
    learning_rate: float = 5e-4                                      # Learning rate for fine-tuning
    
    # Attack Configuration
    image_size: int = 224                                           # Image size (e.g., 224 for 224x224 images)
    perturbation_ratio: float = 0.05                                # Ratio of perturbation to apply (e.g., 0.1 for 10% perturbation)
    alpha: float = 0.2                                              # Alpha value for perturbation blending
    iterations: int = 1200                                          # Number of perturbation iterations per step
    step_size: float = 2 / 255                                      # Step size for perturbation updates
    verbose: bool = True                                            # Whether to print verbose output during training

    # Fine-tuning Parameters
    batch_size: int = 2                                             # Fine-tuning batch size
    max_steps: int = 30                                             # Max number of fine-tuning steps
    save_steps: int = 10                                            # Interval for checkpoint saving
    learning_rate: float = 5e-4                                     # Fine-tuning learning rate
    grad_accumulation_steps: int = 1                                # Gradient accumulation steps
    image_aug: bool = True                                          # Whether to train with image augmentations
    shuffle_buffer_size: int = 100_000                              # Dataloader shuffle buffer size (can reduce if OOM)
    save_latest_checkpoint_only: bool = False                       # Whether to save only one checkpoint per run and
    resume: bool = True

    # Tracking Parameters
    experiment: bool = False                                        # Whether to run the experiment
    use_wandb: bool = True                                          # Whether to use Weights & Biases for tracking
    wandb_project: str = ""                                         # Name of W&B project to log to (use default!)
    wandb_entity: str = ""                                          # Name of entity to log under
    run_id_note: Optional[str] = None                               # Extra note for logging, Weights & Biases
    camera_view: str = "random"                                     # Camera view to use (e.g., `primary`, `wrist`)


DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")


def apply_perturbation(images, perturbation, position=(-1, -1)):

    image_tensors = torch.stack([transforms.ToTensor()(image) for image in images])
    
    B, C, H, W = image_tensors.shape
    pc, ph, pw = perturbation.shape

    assert pc == C, "Perturbation must have the same number of channels as the input images."

    perturbated_images = torch.zeros_like(image_tensors)

    if position == (-1, -1):
        top, left = torch.randint(0, H - ph + 1, (1,)).item(), torch.randint(0, W - pw + 1, (1,)).item()
    else:
        top, left = position

    assert top >= 0 and left >= 0 and top + ph <= H and left + pw <= W, "Perturbation must fit within the image dimensions."

    for i in range(len(perturbated_images)):

        mask = torch.zeros_like(image_tensors[i])
        mask[:, top:top + ph, left:left + pw] = 1.0

        padded_perturb = torch.zeros_like(image_tensors[i])
        padded_perturb[:, top:top + ph, left:left + pw] = perturbation

        perturbated_images[i] = (1 - mask) * image_tensors[i] + padded_perturb
    
    return [
        transforms.ToPILImage()(img.cpu()) for img in perturbated_images
    ]

def load_checkpoint(run_dir, latest):

    checkpoint_dirs = [d for d in run_dir.iterdir() if d.is_dir() and "chkpt" in d.name]
    
    if len(checkpoint_dirs) == 0:
        return run_dir

    elif latest:
        return max(checkpoint_dirs, key=lambda d: int(d.name.split("--")[-1].split("_")[0]))
    
    else:
        checkpoint_dir = min(checkpoint_dirs, key=lambda d: int(d.name.split("--")[-1].split("_")[0]))

        for d in checkpoint_dirs:
            if d != checkpoint_dir:
                shutil.rmtree(d)

        return checkpoint_dir
    
def cleanup_checkpoints(run_dir, keep_last: int = 2):
    chkpts = sorted(
        [d for d in run_dir.iterdir() if d.is_dir() and "chkpt" in d.name],
        key=lambda d: int(d.name.split("--")[-1].split("_")[0])
    )

    if len(chkpts) <= keep_last:
        return

    for d in chkpts[:-keep_last]:
        shutil.rmtree(d)

def save_checkpoint(vla, optimizer, run_dir, idx, loss, max_loss, save_latest_checkpoint_only=True):
    
    os.makedirs(run_dir, exist_ok=True)
    
    save_dir = run_dir

    if save_latest_checkpoint_only:
        # Save Processor & Weights
        safetensors.torch.save_model(vla, save_dir / "model.safetensors")
        torch.save(
            {
                "optimizer": optimizer.state_dict(),
                "idx": idx,
                "loss": loss,
                "max_loss": max_loss
            },
            save_dir / "state.pt",
        )
        print(f"Saved Model Checkpoint for Step {idx} at: {save_dir}")
    else:
        # Prepare to save checkpoint in new directory
        
        checkpoint_dir = run_dir / f"{idx}_chkpt"
        os.makedirs(checkpoint_dir, exist_ok=True)

        safetensors.torch.save_model(vla, checkpoint_dir / "model.safetensors")
        torch.save(
            {
                "optimizer": optimizer.state_dict(),
                "idx": idx,
                "loss": loss,
                "max_loss": max_loss
            },
            checkpoint_dir / "state.pt",
        )

        print(f"Saved Model Checkpoint for Step {idx} at: {checkpoint_dir}")
        cleanup_checkpoints(run_dir, keep_last=2)

@draccus.wrap()
def train_up(cfg: FinetuneConfig) -> None:

    distributed_state = PartialState()
    torch.cuda.set_device(device_id := distributed_state.local_process_index)
    torch.cuda.empty_cache()
    # Configure Unique Experiment ID & Log Directory
    exp_id = (
        f"v1.0.1-pi0"
        f"+view-{cfg.camera_view}"
        f"+adv-encoder-finetune"
    )
    
    if cfg.config_name:
        config = _config.get_config(cfg.config_name)
    elif cfg.model_family == "pi0":
        config = _config.get_config("pi0_libero")
    elif cfg.model_family == "pi05":
        config = _config.get_config("pi05_libero")
    else:
        raise ValueError(f"Unsupported model family: {cfg.model_family}")
    
    tokenizer = _tokenizer.PaligemmaTokenizer()

    orig_encoder = config.model.load_pytorch(config, os.path.join(cfg.vla_path, "model.safetensors")).paligemma_with_expert.paligemma.model.vision_tower.to(dtype=torch.bfloat16, device=device_id)

    run_dir = cfg.run_root_dir / exp_id
    
    if cfg.resume and run_dir.exists():
        checkpoint_dir = load_checkpoint(run_dir, latest=True)
        weight_path = os.path.join(checkpoint_dir, "model.safetensors")
    else:
        weight_path = os.path.join(cfg.vla_path, "model.safetensors")
        
    vla = config.model.load_pytorch(config, weight_path).to(dtype=torch.bfloat16, device=device_id)
    train_encoder = vla.paligemma_with_expert.paligemma.model.vision_tower

    # # Wrap VLA in PyTorch DDP Wrapper for Multi-GPU Training
    # Initialize Logging =>> W&B
    if distributed_state.is_main_process:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=f"Tr+{exp_id}",
            resume="allow",
            id=exp_id,
        )
    
    # Create Optimizer =>> note that we default to a simple constant learning rate!
    trainable_params = [param for param in train_encoder.parameters() if param.requires_grad]
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate)

    start_idx = 0
    previous_loss, previous_max_loss = float("inf"), float("inf")

    if cfg.resume and run_dir.exists():
        # Load optimizer state
        state_path = checkpoint_dir / "state.pt"
        if state_path.exists():
            state_dict = torch.load(state_path, map_location="cpu")
            
            if 'optimizer' in state_dict:
                optimizer.load_state_dict(state_dict["optimizer"])

            start_idx = state_dict["idx"] + 1
            previous_loss = state_dict.get("loss", float("inf"))
            previous_max_loss = state_dict.get("max_loss", float("inf"))
            print(f"Resumed state from step {start_idx}")
        else:
            print(f"No state found at {state_path}, starting fresh.")

    dataloader = RLDSDataLoader(cfg)

    patch_size = int(np.sqrt(cfg.image_size ** 2 * cfg.perturbation_ratio))
    
    from VLAAttacker.pytorch.EDPA import EDPA
    attacker = EDPA(cfg, device_id=device_id)
    
    recent_losses = deque(maxlen=cfg.grad_accumulation_steps)

    if distributed_state.is_main_process:
        loss_window = deque(maxlen=cfg.save_steps)

    orig_encoder.eval()
    with tqdm.tqdm(total=cfg.max_steps, leave=False) as progress:
        train_encoder.train()
        for idx, batch in enumerate(dataloader):

            optimizer.zero_grad()
            attacker.reset_ema()

            perturbation = torch.zeros((3, patch_size, patch_size), dtype=torch.float32).uniform_(0, 1).to(device_id)

            if cfg.camera_view == "random":
                images = [batch["image"][i] if random.random() < 0.5 else batch["wrist_image"][i] for i in range(len(batch["image"]))]
            elif cfg.camera_view == "primary":
                images = batch["image"]
            else:
                images = batch["wrist_image"]

            instructions, instruction_masks = map(
                lambda x: torch.from_numpy(np.stack(x)).long(),
                zip(*[tokenizer.tokenize(i) for i in batch["language_instruction"]])
            )

            tensors = torch.stack([transforms.ToTensor()(image) for image in images])

            perturbation = attacker.generate_one_step(vla, tensors, instructions, instruction_masks, perturbation, eval=False)

            clean = image_transform(torch.stack([transforms.ToTensor()(image) for image in images])).detach()
            adv = image_transform(torch.stack([transforms.ToTensor()(image) for image in apply_perturbation(images, perturbation)])).detach()


            # with torch.autocast("cuda", dtype=torch.bfloat16):
            with torch.no_grad():
                orig_embed = orig_encoder(clean.to(device=device_id)).last_hidden_state
            clean_embed = train_encoder(clean.to(device=device_id)).last_hidden_state
            adv_embed = train_encoder(adv.to(device=device_id)).last_hidden_state
        
            
            loss = torch.nn.functional.mse_loss(clean_embed, orig_embed) + torch.nn.functional.mse_loss(adv_embed, orig_embed)

            # Normalize loss to account for gradient accumulation
            normalized_loss = loss / cfg.grad_accumulation_steps

            # Backward pass
            normalized_loss.backward()
            recent_losses.append(loss.item())

            # Compute gradient step index
            gradient_step_idx = idx // cfg.grad_accumulation_steps

            smoothened_loss = sum(recent_losses) / len(recent_losses)

            if distributed_state.is_main_process:
                loss_window.append(smoothened_loss)

            # Push Metrics to W&B (every 10 gradient steps)
            if distributed_state.is_main_process and (gradient_step_idx + start_idx) % 10 == 0:
                wandb.log(
                    {
                        "MSE Loss": smoothened_loss,
                    },
                    step=gradient_step_idx + start_idx,
                )

            # Optimizer Step
            if (idx + 1) % cfg.grad_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                progress.update()
            
            # Save Model Checkpoint =>> by default, only keeps the latest checkpoint, continually overwriting it!
            if gradient_step_idx > 0 and (gradient_step_idx + start_idx) % cfg.save_steps == 0:

                rollback = 1 if max(loss_window) > previous_max_loss and smoothened_loss > previous_loss else 0

                if rollback == 0:
                    save_checkpoint(
                        vla,
                        optimizer,
                        run_dir,
                        gradient_step_idx + start_idx,
                        smoothened_loss,
                        max(loss_window),
                        save_latest_checkpoint_only=cfg.save_latest_checkpoint_only
                    )
                    previous_loss = smoothened_loss
                    previous_max_loss = max(loss_window)

                else:

                    print(f"Rolling back to the earlist saved checkpoint for Step {gradient_step_idx + start_idx}...")
                    previous_checkpoint_path = load_checkpoint(run_dir, latest=False)

                    # BUGFIX: used to say `checkpoint_dir` here, which is the
                    # initial-resume path from line ~209 and gets deleted by
                    # `load_checkpoint(latest=False)` (that call rmtree's every
                    # ckpt that isn't the earliest). Use `previous_checkpoint_path`,
                    # which is the surviving earliest ckpt we're rolling back to.
                    weight_path = os.path.join(previous_checkpoint_path, "model.safetensors")

                    train_encoder.load_state_dict(
                        config.model.load_pytorch(config, weight_path).paligemma_with_expert.paligemma.model.vision_tower.state_dict()
                    )

                    state_dict = torch.load(previous_checkpoint_path / "state.pt", map_location="cpu")
                    optimizer.load_state_dict(state_dict["optimizer"])
                    previous_loss = state_dict.get("loss", float("inf"))
                    previous_max_loss = state_dict.get("max_loss", float("inf"))


            
            # Stop training when max_steps is reached
            if gradient_step_idx == cfg.max_steps:
                print(f"Max step {cfg.max_steps} reached! Stopping training...")
                break


if __name__ == "__main__":
    train_up()