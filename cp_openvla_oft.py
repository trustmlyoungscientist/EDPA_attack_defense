import os
import torch
from torch.utils.data import DataLoader
from accelerate import PartialState
import time
import draccus
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from transformers import AutoModelForVision2Seq, AutoProcessor
from transformers import AutoConfig, AutoImageProcessor

from openvla_oft.prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from openvla_oft.prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from openvla_oft.prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from openvla_oft.prismatic.vla.action_tokenizer import ActionTokenizer
from openvla_oft.prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder

# from transformers.modeling_outputs import CausalLMOutputWithPast
from utils.data_utils_openvla_oft import RLDSBatchTransform, RLDSDataset, PaddedCollatorForActionPrediction

import tqdm
import draccus

from utils.data_utils import RLDSDataLoader

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class FinetuneConfig:
    # fmt: off
    # vla_path: str = "openvla/openvla-7b"
    model_family: str = "openvla"                                     # Path to OpenVLA model (on HuggingFace Hub)
    vla_path: str = "moojink/openvla-7b-oft-finetuned-libero-spatial"     # Path to OpenVLA model (on HuggingFace Hub)
    # Directory Paths
    data_root_dir: str =  "/ibex/user/zhanj0a/haochuan_data/dataset/modified_libero_rlds"
    # data_root_dir: Path = Path("dataset/modified_libero_rlds")        # Path to Open-X dataset directory
    dataset_name: str = "libero_spatial_no_noops"                     # Name of fine-tuning dataset (e.g., `droid_wipe`)
    run_root_dir: Path = Path("runs")                               # Path to directory to store logs & checkpoints
    adapter_tmp_dir: Path = Path("adapter-tmp")                     # Temporary directory for LoRA weights before fusing
    local_log_dir: str = "./logs"

    # Attack Configuration
    attacker: str = "UPL"                                          # Attacker to use (e.g., `UUP`, `UAP`)
    image_size: int = 256                                           # Image size (e.g., 224 for 224x224 images)
    perturbation_ratio: float = 0.04                                # Ratio of perturbation to apply (e.g., 0.1 for 10% perturbation)
    alpha: float = 0.8                                              # Alpha value for perturbation blending
    max_steps: int = 1000                                           # Maximum number of perturbation steps
    iterations: int = 1                                             # Number of perturbation iterations per step
    step_size: float = 2 / 255                                      # Step size for perturbation updates
    save_path: str = "perturbation/openvla-base-wrist/openvla-upl-test-1"                     # Path to save perturbations
    verbose: bool = True                                            # Whether to print verbose output during training

    # Fine-tuning Parameters
    batch_size: int = 4                                             # Fine-tuning batch size
    save_steps: int = 10                                            # Interval for checkpoint saving
    grad_accumulation_steps: int = 1                                # Gradient accumulation steps
    image_aug: bool = True                                          # Whether to train with image augmentations
    shuffle_buffer_size: int = 2000                                 # Dataloader shuffle buffer size (can reduce if OOM)

    # Tracking Parameters
    experiment: bool = False                                         # Whether to run the experiment
    use_wandb: bool = True                                          # Whether to use Weights & Biases for tracking
    wandb_project: str = "openvla-test"                                  # Name of W&B project to log to (use default!)
    wandb_entity: str = "shua754-university-of-auckland"                                   # Name of entity to log under
    run_id_note: Optional[str] = None                               # Extra note for logging, Weights & Biases

    camera_view: str = "wrist"                                    # Camera view to use (e.g., `front`, `top`, `side`)


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
        trust_remote_code=True,
    ).to(device_id)

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

    os.makedirs(f"{cfg.save_path}-{cfg.perturbation_ratio}", exist_ok=True)

    from VLAAttacker.pytorch.EDPA import EDPA
    attacker = EDPA(cfg, device_id=device_id) # initialize adverserial attacker based on the devices
    
    # Generate perturbation
    # print(vla)
    attacker.generate(vla, dataloader, processor, action_tokenizer)



if __name__ == "__main__":
    train_up()
