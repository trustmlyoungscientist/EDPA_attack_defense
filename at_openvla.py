import os
import torch
from torch.optim import AdamW, SGD
import torch.distributed as dist
from collections import deque
from torch.nn.parallel import DistributedDataParallel as DDP
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
from transformers import AutoModelForVision2Seq, AutoProcessor
from transformers import AutoConfig, AutoImageProcessor

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

from utils.openvla import build_prompt
from utils.data_utils import RLDSDataLoader
# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class FinetuneConfig:
    # fmt: off
    # vla_path: str = "openvla/openvla-7b"
    model_family: str = "openvla"                                     # Path to OpenVLA model (on HuggingFace Hub)
    vla_path: str = "openvla/openvla-7b-finetuned-libero-spatial"     # Path to OpenVLA model (on HuggingFace Hub)
    # Directory Paths
    data_root_dir: Path = Path("dataset/modified_libero_rlds")        # Path to Open-X dataset directory
    dataset_name: str = "libero_spatial_no_noops"                     # Name of fine-tuning dataset (e.g., `droid_wipe`)
    run_root_dir: Path = Path("runs")                                 # Path to directory to store logs & checkpoints
    local_log_dir: str = "./logs"
    optimizer: str = "adamw"                                          # Optimizer to use for fine-tuning
    learning_rate: float = 5e-4                                       # Learning rate for fine-tuning
    resume : bool = True                                              # Whether to resume from latest checkpoint in run_root_dir

    # Attack Configuration
    image_size: int = 224                                           # Image size (e.g., 224 for 224x224 images)
    perturbation_ratio: float = 0.05                                # Ratio of perturbation to apply (e.g., 0.1 for 10% perturbation)
    alpha: float = 0.2                                              # Alpha value for perturbation blending
    iterations: int = 1200                                          # Number of perturbation iterations per step
    step_size: float = 2 / 255                                      # Step size for perturbation updates
    verbose: bool = True                                            # Whether to print verbose output during training
    reset_steps: int = 1000                                         # Steps after which to reset perturbation
    
    # Fine-tuning Parameters
    batch_size: int = 2                                             # Fine-tuning batch size
    max_steps: int = 200_000                                        # Max number of fine-tuning steps
    save_steps: int = 100                                           # Interval for checkpoint saving
    learning_rate: float = 5e-4                                     # Fine-tuning learning rate
    grad_accumulation_steps: int = 1                                # Gradient accumulation steps
    shuffle_buffer_size: int = 100_000                              # Dataloader shuffle buffer size (can reduce if OOM)
    save_latest_checkpoint_only: bool = True                        # Whether to save only one checkpoint per run and

    # Tracking Parameters
    experiment: bool = False                                        # Whether to run the experiment
    use_wandb: bool = True                                          # Whether to use Weights & Biases for tracking
    wandb_project: str = ""                                         # Name of W&B project to log to (use default!)
    wandb_entity: str = ""                                          # Name of entity to log under
    run_id_note: Optional[str] = None                               # Extra note for logging, Weights & Biases
    camera_view: str = "primary"                                    # Camera view to use (e.g., `primary`, `wrist`)


DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")


def apply_perturbation(images, perturbation, position=(-1, -1)):

    image_tensors = torch.stack([transforms.ToTensor()(image) for image in images])

    B, C, H, W = image_tensors.shape

    if perturbation.dim() == 3:
        perturbation = perturbation.unsqueeze(0).expand(B, -1, -1, -1)
    elif perturbation.dim() != 4:
        raise ValueError("perturbation must be 3D or 4D")

    _, pc, ph, pw = perturbation.shape
    assert pc == C

    # ---- positions ----
    if position == (-1, -1):
        tops = torch.randint(0, H - ph + 1, (B,))
        lefts = torch.randint(0, W - pw + 1, (B,))
    else:
        tops = [position[0]] * B
        lefts = [position[1]] * B

    perturbated_images = torch.zeros_like(image_tensors)

    for i in range(B):
        top, left = tops[i], lefts[i]

        mask = torch.zeros_like(image_tensors[i])
        mask[:, top:top + ph, left:left + pw] = 1.0

        padded = torch.zeros_like(image_tensors[i])
        padded[:, top:top + ph, left:left + pw] = perturbation[i]

        perturbated_images[i] = (1 - mask) * image_tensors[i] + padded

    return [transforms.ToPILImage()(img.cpu()) for img in perturbated_images]

@draccus.wrap()
def train_up(cfg: FinetuneConfig) -> None:
    
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    distributed_state = PartialState()
    torch.cuda.set_device(device_id := distributed_state.local_process_index)
    torch.cuda.empty_cache()

    # Configure Unique Experiment ID & Log Directory
    exp_id = (
        f"v2.2.0-{cfg.vla_path.split('/')[-1]}+{cfg.dataset_name}"
        f"+view-{cfg.camera_view}"
        f"+adv-encoder-finetune"
        f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
        f"+lr-{cfg.learning_rate}"
    )

    if cfg.run_id_note is not None:
        exp_id += f"--{cfg.run_id_note}"

    # Start =>> Build Directories
    run_dir = cfg.run_root_dir / exp_id

    orig_encoder = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16,
        quantization_config=None,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).vision_backbone.to(device_id)

    if cfg.resume and run_dir.exists():

        checkpoint_dirs = [d for d in run_dir.iterdir() if d.is_dir() and "chkpt" in d.name]
        if len(checkpoint_dirs) == 0:
            latest_checkpoint_dir = run_dir
        else:
            latest_checkpoint_dir = max(checkpoint_dirs, key=lambda d: int(d.name.split("--")[-1].split("_")[0]))

        print(f"Resuming from checkpoint directory: {latest_checkpoint_dir}")

        # Load Processor and Model Weights
        processor = AutoProcessor.from_pretrained(latest_checkpoint_dir, trust_remote_code=True)
        vla = AutoModelForVision2Seq.from_pretrained(
            latest_checkpoint_dir,
            torch_dtype=torch.bfloat16,
            quantization_config=None,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).to(device_id)

    else:
        # Load OpenVLA Processor and Model using HF AutoClasses
        processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
        vla = AutoModelForVision2Seq.from_pretrained(
            cfg.vla_path,
            torch_dtype=torch.bfloat16,
            quantization_config=None,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).to(device_id)

    train_encoder = vla.vision_backbone.to(device_id)

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
    # optimizer = AdamW(trainable_params, lr=cfg.learning_rate)

    if cfg.optimizer.lower() == "adamw":
        optimizer = AdamW(trainable_params, lr=cfg.learning_rate)
    elif cfg.optimizer.lower() == "sgd":
        optimizer = SGD(trainable_params, lr=cfg.learning_rate, momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer: {cfg.optimizer!r}. Choose 'adamw' or 'sgd'.")

    start_idx = 0
    previous_loss, previous_max_loss = float("inf"), float("inf")

    if cfg.resume and run_dir.exists():
        # Load optimizer state
        state_path = latest_checkpoint_dir / "state.pt"
        if state_path.exists():
            state_dict = torch.load(state_path, map_location="cpu")
            start_idx = state_dict["idx"] + 1
            previous_loss = state_dict.get("loss", float("inf"))
            previous_max_loss = state_dict.get("max_loss", float("inf"))
            print(f"Resumed state from step {start_idx}")
        else:
            print(f"No state found at {state_path}, starting fresh.")

    train_encoder = DDP(train_encoder, device_ids=[device_id], find_unused_parameters=True, gradient_as_bucket_view=True)

    dataloader = RLDSDataLoader(cfg)

    patch_size = int(np.sqrt(cfg.image_size ** 2 * cfg.perturbation_ratio))
    
    from VLAAttacker.pytorch.EDPA import EDPA
    attacker = EDPA(cfg, device_id=device_id)
    
    if distributed_state.is_main_process:
        
        loss_window = deque(maxlen=cfg.save_steps)
        previous_checkpoints = deque(maxlen=2)
        
        previous_checkpoints.append({
            "encoder_state": {
                k: v.cpu().detach().clone() for k, v in train_encoder.module.state_dict().items()  # detach 和 clone
            },
            "loss": previous_loss,
            "max_loss": previous_max_loss
        })

    recent_losses = deque(maxlen=cfg.grad_accumulation_steps)

    orig_encoder.eval()
    with tqdm.tqdm(total=cfg.max_steps, leave=False) as progress:
        train_encoder.train()
        for idx, batch in enumerate(dataloader):
            
            attacker.reset_ema()
            optimizer.zero_grad()

            # perturbation = torch.zeros(
            #     (cfg.batch_size, 3, patch_size, patch_size),
            #     dtype=torch.float32,
            #     device=device_id
            # ).uniform_(0, 1)
            
            # if idx % cfg.reset_steps == 0:
            perturbation = torch.zeros((3, patch_size, patch_size), dtype=torch.float32).uniform_(0, 1).to(device_id)

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

            prompt = build_prompt(batch["language_instruction"])

            tensors = torch.stack([transforms.ToTensor()(image) for image in images])
            instructions, instructions_masks = ( 
                 processor.tokenizer(prompt, padding=True, return_tensors="pt")[k] for k in ("input_ids", "attention_mask") 
            )
    
            perturbation = attacker.generate_one_step(vla, tensors, instructions, instructions_masks, perturbation, eval=False)

            clean = processor.image_processor(images)
            adv = processor.image_processor(apply_perturbation(images, perturbation))

            with torch.autocast("cuda", dtype=torch.bfloat16):
                orig_embed = orig_encoder(torch.tensor(clean["pixel_values"]).to(device=device_id, dtype=torch.bfloat16))
                clean_embed = train_encoder(torch.tensor(clean["pixel_values"]).to(device=device_id, dtype=torch.bfloat16))
                adv_embed = train_encoder(torch.tensor(adv["pixel_values"]).to(device=device_id, dtype=torch.bfloat16))
            
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
            if gradient_step_idx > 0 and gradient_step_idx % cfg.save_steps == 0:
                if distributed_state.is_main_process:

                    if max(loss_window) > previous_max_loss and smoothened_loss > previous_loss:
                        # Revert to previous checkpoint if loss has increased significantly
                        print(f"Loss increased from {previous_loss:.4f} to {smoothened_loss:.4f}. Reverting to second previous checkpoint.")
                        
                        if len(previous_checkpoints) > 1:
                            previous_checkpoints.pop()
                        checkpoint = previous_checkpoints[-1]

                        train_encoder.module.load_state_dict(checkpoint["encoder_state"])
                        previous_loss = checkpoint["loss"]
                        previous_max_loss = checkpoint["max_loss"]

                        for p in train_encoder.module.parameters():
                            dist.broadcast(p.data, src=0)

                    else:
                        print(f"Saving Model Checkpoint for Step {gradient_step_idx + start_idx}")
                        save_dir = run_dir

                        if cfg.save_latest_checkpoint_only:
                            # Save Processor & Weights
                            processor.save_pretrained(run_dir)
                            vla.save_pretrained(save_dir)

                            torch.save(
                                {
                                    "idx": gradient_step_idx + start_idx,
                                    "loss": smoothened_loss,
                                    "max_loss": max(loss_window)
                                },
                                save_dir / "state.pt",
                            )

                            print(f"Saved Model Checkpoint for Step {gradient_step_idx + start_idx} at: {save_dir}")
                        else:
                            # Prepare to save checkpoint in new directory
                            checkpoint_dir = Path(str(run_dir) + f"--{gradient_step_idx + start_idx}_chkpt")
                            os.makedirs(checkpoint_dir, exist_ok=True)

                            # Save processor and model weights to new directory
                            processor.save_pretrained(checkpoint_dir)
                            vla.save_pretrained(checkpoint_dir)

                            torch.save(
                                {
                                    "idx": gradient_step_idx + start_idx,
                                    "loss": smoothened_loss,
                                    "max_loss": max(loss_window)
                                },
                                checkpoint_dir / "state.pt",
                            )

                            print(f"Saved Model Checkpoint for Step {gradient_step_idx + start_idx} at: {checkpoint_dir}")

                        previous_checkpoints.append({
                            "encoder_state": {
                                k: v.cpu().detach().clone() for k, v in train_encoder.module.state_dict().items()  # detach 和 clone
                            },
                            "loss": smoothened_loss,
                            "max_loss": max(loss_window)
                        })
                        
                        previous_loss = smoothened_loss
                        previous_max_loss = max(loss_window)
                            
                # Wait for processor and adapter weights to be saved by main process
                dist.barrier()

            
            # Stop training when max_steps is reached
            if gradient_step_idx == cfg.max_steps:
                print(f"Max step {cfg.max_steps} reached! Stopping training...")
                break


if __name__ == "__main__":
    train_up()