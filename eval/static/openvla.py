import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt

import draccus
import torch
import torch.distributed as dist
import tqdm
from accelerate import PartialState
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from transformers import AutoConfig, AutoImageProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast

import wandb
from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

from rlds_dataset import RLDSDataset, RLDSBatchTransform

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# # === Utilities ===
# # fmt: off
# def create_vision_transform(vla: nn.Module, input_size: int) -> Callable[[Image.Image], torch.Tensor]:
#     """Gets image transform for the vision encoder."""
#     data_cfg = timm.data.resolve_model_data_config(vla.vision_backbone)
#     data_cfg["input_size"] = (3, input_size, input_size)
#     return timm.data.create_transform(
#         input_size=data_cfg["input_size"],
#         interpolation=data_cfg["interpolation"],
#         mean=data_cfg["mean"],
#         std=data_cfg["std"],
#         crop_pct=1.0,           # Set to 1.0 to disable cropping
#         crop_mode="center",     # Default crop mode --> no-op when `crop_pct == 1.0`
#         is_training=False,      # Disable image_aug when loading transform; handled by RLDS dataloader
#     )
#
# # fmt: on


@dataclass
class FinetuneConfig:
    # fmt: off
    # vla_path: str = "openvla/openvla-7b-finetuned-libero-spatial"     # Path to OpenVLA model (on HuggingFace Hub)
    vla_path: str = "/media/haochuanxu/Working/githubRepo/model/openvla-7b-libero_spatial-wrist-img-encoder"
    # vla_path: str = "/media/haochuanxu/Working/githubRepo/model/openvla-7b-libero_spatial-wrist"
    # Directory Paths
    data_root_dir: Path = Path("dataset/modified_libero_rlds")        # Path to Open-X dataset directory
    dataset_name: str = "libero_spatial_no_noops"                     # Name of fine-tuning dataset (e.g., `droid_wipe`)
    run_root_dir: Path = Path("runs")                                 # Path to directory to store logs & checkpoints
    adapter_tmp_dir: Path = Path("adapter-tmp")                       # Temporary directory for LoRA weights before fusing

    # Fine-tuning Parameters
    batch_size: int = 1                                             # Fine-tuning batch size
    max_steps: int = 50_000                                         # Max number of fine-tuning steps
    save_steps: int = 5000                                          # Interval for checkpoint saving
    learning_rate: float = 5e-4                                     # Fine-tuning learning rate
    shuffle_buffer_size: int = 10_000                               # Dataloader shuffle buffer size (can reduce if OOM)
                                                                    # continually overwrite the latest checkpoint
                                                                    # (If False, saves all checkpoints

    # Tracking Parameters
    wandb_project: str = "openvla-defense"                          # Name of W&B project to log to (use default!)
    wandb_entity: str = "hxu612"                                    # Name of entity to log under
    run_id_note: Optional[str] = None                               # Extra note for logging, Weights & Biases

    # camera view for finetuning
    attack: bool = True                                             # Whether to use attack mode (e.g., `True` for adversarial attacks)
    camera_view: str = "wrist"                                      # Camera view for fine-tuning (e.g., `primary`, `wrist`, etc.)
    perturbation_path: str = ""                                     # Path to perturbation file


@draccus.wrap()
def eval(cfg: FinetuneConfig) -> None:
    print(f"Fine-tuning OpenVLA Model `{cfg.vla_path}` on `{cfg.dataset_name}`")

    # [Validate] Ensure GPU Available & Set Device / Distributed Context
    assert torch.cuda.is_available(), "Fine-tuning assumes at least one GPU is available!"
    distributed_state = PartialState()
    torch.cuda.set_device(device_id := distributed_state.local_process_index)
    torch.cuda.empty_cache()

    # Configure Unique Experiment ID & Log Directory
    exp_id = (
        f"{cfg.vla_path.split('/')[-1]}+{cfg.dataset_name}"
        f"+view-{cfg.camera_view}"
        f"+attack-{cfg.attack}"
    )

    # Start =>> Build Directories
    run_dir, adapter_dir = cfg.run_root_dir / exp_id, cfg.adapter_tmp_dir / exp_id
    os.makedirs(run_dir, exist_ok=True)


    # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    # Load OpenVLA Processor and Model using HF AutoClasses
    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device_id)

    # Create Action Tokenizer
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    perturbation = None
    if cfg.attack:
        perturbation = torch.load(cfg.perturbation_path) if cfg.perturbation_path else torch.zeros((3, 50, 50),
                                                                                          dtype=torch.float32).uniform_(0, 1)

    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder if "v01" not in cfg.vla_path else VicunaV15ChatPromptBuilder,
        view=cfg.camera_view,
        perturbation=perturbation,  # Use perturbation if in attack mode
    )

    vla_dataset = RLDSDataset(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        resize_resolution=tuple(vla.config.image_sizes),
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        train=False,  # Set to False for evaluation
    )

    # # [Important] Save Dataset Statistics =>> used to de-normalize actions for inference!

    # save_dataset_statistics(vla_dataset.dataset_statistics, run_dir)

    # Create Collator and DataLoader
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

    # Initialize Logging =>> W&B
    if distributed_state.is_main_process:
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=f"eval+{exp_id}")

    total_loss, total_action_accuracy, total_l1_loss = 0.0, 0.0, 0.0

    with tqdm.tqdm(total=cfg.max_steps, desc="Evaluating", leave=False) as progress:
        for idx, batch in enumerate(dataloader):  
            with torch.no_grad():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output: CausalLMOutputWithPast = vla(
                        input_ids=batch["input_ids"].to(device_id),
                        attention_mask=batch["attention_mask"].to(device_id),
                        pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
                        labels=batch["labels"],
                    )
                    loss = output.loss

                # Compute Accuracy and L1 Loss
                action_logits = output.logits[:, vla.vision_backbone.featurizer.patch_embed.num_patches : -1]
                action_preds = action_logits.argmax(dim=2)
                action_gt = batch["labels"][:, 1:].to(action_preds.device)
                mask = action_gt > action_tokenizer.action_token_begin_idx

                correct_preds = (action_preds == action_gt) & mask
                action_accuracy = correct_preds.sum().float() / mask.sum().float()

                continuous_actions_pred = torch.tensor(
                    action_tokenizer.decode_token_ids_to_actions(action_preds[mask].cpu().numpy())
                )
                continuous_actions_gt = torch.tensor(
                    action_tokenizer.decode_token_ids_to_actions(action_gt[mask].cpu().numpy())
                )
                action_l1_loss = torch.nn.functional.l1_loss(continuous_actions_pred, continuous_actions_gt)

                # Accumulate
                total_loss += loss.item() 
                total_action_accuracy += action_accuracy.item()
                total_l1_loss += action_l1_loss.item()

                # Push Metrics to W&B (every 10 gradient steps)
                if distributed_state.is_main_process and idx % 10 == 0:
                    
                    num_batches = idx + 1
                    # print(f"[Step {num_batches}] Avg Loss: {total_loss / num_batches:.4f} | Acc: {total_action_accuracy / num_batches:.4f} | L1: {total_l1_loss / num_batches:.4f}")

                    wandb.log(
                        {
                            "eval_loss": total_loss / num_batches,
                            "eval_action_accuracy": total_action_accuracy / num_batches,
                            "eval_l1_loss": total_l1_loss / num_batches,
                        },
                        step=num_batches,
                    )
                
                if idx >= cfg.max_steps:
                    print(f"Reached max steps: {cfg.max_steps}. Stopping evaluation.")
                    break

                progress.update()

    # Final Averages
    avg_loss = total_loss / num_batches
    avg_accuracy = total_action_accuracy / num_batches
    avg_l1_loss = total_l1_loss / num_batches

    if distributed_state.is_main_process:
        print(f"[Eval Result] Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}, L1 Loss: {avg_l1_loss:.4f}")
        wandb.log(
            {
                "eval_loss": avg_loss,
                "eval_action_accuracy": avg_accuracy,
                "eval_l1_loss": avg_l1_loss,
            }
        )
if __name__ == "__main__":
    eval()
