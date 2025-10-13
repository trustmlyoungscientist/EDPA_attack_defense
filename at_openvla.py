import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
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

from openvla.prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from openvla.prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from openvla.prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from openvla.prismatic.vla.action_tokenizer import ActionTokenizer
from openvla.prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder

from utils.data_utils_openvla import RLDSBatchTransform, RLDSDataset, PaddedCollatorForActionPrediction
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
    run_root_dir: Path = Path("runs")                               # Path to directory to store logs & checkpoints
    adapter_tmp_dir: Path = Path("adapter-tmp")                     # Temporary directory for LoRA weights before fusing
    local_log_dir: str = "./logs"
    learning_rate: float = 5e-4                                      # Learning rate for fine-tuning
    
    # Attack Configuration
    image_size: int = 224                                           # Image size (e.g., 224 for 224x224 images)
    perturbation_ratio: float = 0.04                                 # Ratio of perturbation to apply (e.g., 0.1 for 10% perturbation)
    alpha: float = 0.8                                              # Alpha value for perturbation blending
    max_steps: int = 50000                                          # Maximum number of perturbation steps
    iterations: int = 1                                             # Number of perturbation iterations per step
    step_size: float = 2 / 255                                      # Step size for perturbation updates
    verbose: bool = True                                            # Whether to print verbose output during training
    reset_steps: int = 1000                                         # Steps after which to reset perturbation

    # Fine-tuning Parameters
    batch_size: int = 1                                             # Fine-tuning batch size
    max_steps: int = 200_000                                        # Max number of fine-tuning steps
    save_steps: int = 5000                                          # Interval for checkpoint saving
    learning_rate: float = 5e-4                                     # Fine-tuning learning rate
    grad_accumulation_steps: int = 1                                # Gradient accumulation steps
    image_aug: bool = True                                          # Whether to train with image augmentations
    shuffle_buffer_size: int = 100_000                              # Dataloader shuffle buffer size (can reduce if OOM)
    save_latest_checkpoint_only: bool = True                        # Whether to save only one checkpoint per run and
                                                                    #   continually overwrite the latest checkpoint
                                                                    #   (If False, saves all checkpoints)

    # Tracking Parameters
    experiment: bool = True                                         # Whether to run the experiment
    use_wandb: bool = True                                          # Whether to use Weights & Biases for tracking
    wandb_project: str = ""                                         # Name of W&B project to log to (use default!)
    wandb_entity: str = "kaustuoa"                                  # Name of entity to log under
    run_id_note: Optional[str] = None                               # Extra note for logging, Weights & Biases
    camera_view: str = "primary"                                    # Camera view to use (e.g., `front`, `top`, `side`)


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
        f"{cfg.vla_path.split('/')[-1]}+{cfg.dataset_name}"
        f"+view-{cfg.camera_view}"
        f"+adv-encoder-finetune"
        f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
        f"+lr-{cfg.learning_rate}"
    )

    if cfg.run_id_note is not None:
        exp_id += f"--{cfg.run_id_note}"
    if cfg.image_aug:
        exp_id += "--image_aug"

    # Start =>> Build Directories
    run_dir = cfg.run_root_dir / exp_id
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

    orig_encoder = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16,
        quantization_config=None,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).vision_backbone.to(device_id)

    # # Wrap VLA in PyTorch DDP Wrapper for Multi-GPU Training
        # Initialize Logging =>> W&B
    if distributed_state.is_main_process:
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=f"Tr+{exp_id}")
    
    train_encoder = DDP(train_encoder, device_ids=[device_id], find_unused_parameters=True, gradient_as_bucket_view=True)

    # Create Optimizer =>> note that we default to a simple constant learning rate!
    trainable_params = [param for param in train_encoder.parameters() if param.requires_grad]
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate)

    action_tokenizer = ActionTokenizer(processor.tokenizer)

    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder if "v01" not in cfg.vla_path else VicunaV15ChatPromptBuilder,
        view=cfg.camera_view,
    )

    vla_dataset = RLDSDataset(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        resize_resolution=tuple(vla.config.image_sizes),
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        train=True,  
    )

    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )

    dataloader = DataLoader(
        vla_dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,  # Important =>> Set to 0 if using RLDS; TFDS rolls its own parallelism!
    )

    patch_size = int(np.sqrt(cfg.image_size ** 2 * 0.04))
    
    from VLAAttacker.pytorch.EDPA import UPL
    attacker = UPL(cfg, device_id=device_id)
    
    recent_losses = deque(maxlen=cfg.grad_accumulation_steps)

    orig_encoder.eval()
    with tqdm.tqdm(total=cfg.max_steps, leave=False) as progress:
        train_encoder.train()
        for idx, batch in enumerate(dataloader):

            optimizer.zero_grad()
            
            if idx % cfg.reset_steps == 0:
                perturbation = torch.zeros((3, patch_size, patch_size), dtype=torch.float32).uniform_(0, 1).to(device_id)
                vla.vision_backbone = train_encoder.module

            images, instructions, instructions_masks = batch["images"], batch["input_ids"], batch["attention_mask"]

            perturbation = attacker.generate_one_step(vla, images, instructions, instructions_masks, processor, perturbation, eval=False)

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

            # Push Metrics to W&B (every 10 gradient steps)
            if distributed_state.is_main_process and gradient_step_idx % 10 == 0:
                wandb.log(
                    {
                        "MSE Loss": smoothened_loss,
                    },
                    step=gradient_step_idx,
                )

            # Optimizer Step
            if (idx + 1) % cfg.grad_accumulation_steps == 0:

                optimizer.step()
                optimizer.zero_grad()
                progress.update()

                
            
            # Save Model Checkpoint =>> by default, only keeps the latest checkpoint, continually overwriting it!
            if gradient_step_idx > 0 and gradient_step_idx % cfg.save_steps == 0:
                if distributed_state.is_main_process:
                    print(f"Saving Model Checkpoint for Step {gradient_step_idx}")

                    # If LoRA, we first save adapter weights, then merge into full model; otherwise, default save!
                    save_dir = run_dir

                    if cfg.save_latest_checkpoint_only:
                        # Save Processor & Weights
                        processor.save_pretrained(run_dir)
                        vla.save_pretrained(save_dir)
                        print(f"Saved Model Checkpoint for Step {gradient_step_idx} at: {save_dir}")
                    else:
                        # Prepare to save checkpoint in new directory
                        checkpoint_dir = Path(str(run_dir) + f"--{gradient_step_idx}_chkpt")
                        os.makedirs(checkpoint_dir, exist_ok=True)

                        # Save processor and model weights to new directory
                        processor.save_pretrained(checkpoint_dir)
                        vla.save_pretrained(checkpoint_dir)

                        print(f"Saved Model Checkpoint for Step {gradient_step_idx} at: {checkpoint_dir}")

                # Wait for processor and adapter weights to be saved by main process
                dist.barrier()

            
            # Stop training when max_steps is reached
            if gradient_step_idx == cfg.max_steps:
                print(f"Max step {cfg.max_steps} reached! Stopping training...")
                vla.vision_backbone = train_encoder.module
                break


if __name__ == "__main__":
    train_up()