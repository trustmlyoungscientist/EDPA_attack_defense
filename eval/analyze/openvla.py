import os
from PIL import Image
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
import draccus
import torch
import torch.distributed as dist
from torchvision import transforms
from accelerate import PartialState
import matplotlib.cm as cm

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from transformers import AutoConfig, AutoImageProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast

import wandb
from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

from rlds_dataset import RLDSDataset, RLDSBatchTransform, PaddedCollatorForActionPrediction

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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

def generate_attention_heatmaps_per_token(
    attention_matrix: torch.Tensor,  # shape: [num_text_tokens, 256]
    save_dir: str,
    prefix: str = "text_token",
    yticklabels = None,
    image: Image.Image = None,
    cmap: str = "viridis"
) -> list[list[np.ndarray]]:


    os.makedirs(save_dir, exist_ok=True)
    attention_matrix = attention_matrix.to(torch.float32)
    num_tokens = attention_matrix.shape[0]

    for i in range(num_tokens):
        attn = attention_matrix[i].reshape(int(np.sqrt(attention_matrix.shape[1])),
                                           int(np.sqrt(attention_matrix.shape[1])))
        attn = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)

        attn_colormap = cm.get_cmap(cmap)
        attn_colored = Image.fromarray(np.uint8(attn_colormap(attn.cpu().numpy()) * 255))

        if image is not None:
            base_img = image.convert("RGBA")
            attn_colored = attn_colored.convert("RGBA").resize(base_img.size)
            combined_img = Image.blend(base_img, attn_colored, alpha=0.5)
        else:
            combined_img = attn_colored

        img_array = np.array(combined_img)
        if img_array.shape[-1] == 4:
            img_array = img_array[..., :3]

        print(f"token {i}: {yticklabels[i]}")
        save_path = os.path.join(save_dir, f"token_{i}.png")
        combined_img.save(save_path)


def get_avg_patch_text_attention(attn, patch_len, text_mask, layer=-1):


    cls_offset = 1
    seq_offset_patch = cls_offset
    seq_offset_text = cls_offset + patch_len

    # 取出该层 attention：attn[layer]: (1, num_heads, seq_len, seq_len)
    attn_layer = attn[layer][0]  # shape: (num_heads, seq_len, seq_len)
    avg_attn = attn_layer.mean(dim=0)  # shape: (seq_len, seq_len)

    # 原始 text_len
    text_len = text_mask.shape[0]

    # patch ➝ text
    patch2text = avg_attn[seq_offset_patch:seq_offset_text, seq_offset_text:seq_offset_text + text_len]
    patch2text = patch2text[:, text_mask[1:]]  # 按 mask 筛选 text 方向的列

    # text ➝ patch
    text2patch = avg_attn[seq_offset_text:seq_offset_text + text_len, seq_offset_patch:seq_offset_text]
    text2patch = text2patch[text_mask[1:], :]  # 按 mask 筛选 text 方向的行
    
    return patch2text, text2patch

@dataclass
class FinetuneConfig:
    # fmt: off
    vla_path: str = "openvla/openvla-7b-finetuned-libero-spatial"     # Path to OpenVLA model (on HuggingFace Hub)
    # vla_path: str = "/media/haochuanxu/Working/githubRepo/model/openvla-7b-libero_spatial-wrist-img-encoder"     # Path to OpenVLA model (on HuggingFace Hub)
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
    wandb_project: str = "openvla"                                  # Name of W&B project to log to (use default!)
    wandb_entity: str = "hxu612"                                    # Name of entity to log under
    run_id_note: Optional[str] = None                               # Extra note for logging, Weights & Biases

    # camera view for finetuning
    attack: bool = True                                              # Whether to use attack mode (e.g., `True` for adversarial attacks)
    camera_view: str = "wrist"                                       # Camera view for fine-tuning (e.g., `primary`, `wrist`, etc.)
    perturbation_path: str = "perturbation/openvla-uupp/perturbation.pt"    # Path to perturbation file


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


    loaded_sample = torch.load('results/sample_data.pt')

    image = loaded_sample['image']
    input_ids = loaded_sample['input_ids'].unsqueeze(0)  # Ensure input_ids is a 2D tensor
    attention_mask = loaded_sample['attention_mask'].unsqueeze(0)  # Ensure attention_mask is a 2D tensor
    labels = loaded_sample['labels'].unsqueeze(0)  # Ensure labels is a 2D tensor


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

    image = apply_perturbation(
        images=[image],
        perturbation=torch.load(cfg.perturbation_path).to(torch.float32),
        position=(0, 0)
    )[0]
     # Save perturbed image for reference

    image.save("perturbed_image.png")
    pixel_values = torch.tensor(processor.image_processor(image)["pixel_values"])

    total_loss, total_action_accuracy, total_l1_loss = 0.0, 0.0, 0.0

    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = vla(
                input_ids=input_ids.to(device_id),
                attention_mask=attention_mask.to(device_id),
                pixel_values=pixel_values.to(torch.bfloat16).to(device_id),
                labels=labels,
                output_attentions=True,
            )
            loss = output.loss

        mask = input_ids < action_tokenizer.action_token_begin_idx
        masked_input_ids = input_ids[mask]

        text_label = [processor.tokenizer.decode(token.cpu().numpy()) for token in masked_input_ids]

        patch2text_attn, text2patch_attn = get_avg_patch_text_attention(
            output.attentions, vla.vision_backbone.featurizer.patch_embed.num_patches, mask[0], layer=0
        )

        generate_attention_heatmaps_per_token(
            attention_matrix=text2patch_attn,  # shape: [num_text_tokens, 256]
            save_dir="outputs/attention_maps_first_layer",
            prefix="token",
            yticklabels=text_label[1:],
            image=image
        )
        
        patch2text_attn, text2patch_attn = get_avg_patch_text_attention(
            output.attentions, vla.vision_backbone.featurizer.patch_embed.num_patches, mask[0], layer=-1
        )

        generate_attention_heatmaps_per_token(
            attention_matrix=text2patch_attn,  # shape: [num_text_tokens, 256]
            save_dir="outputs/attention_maps_last_layer",
            prefix="token",
            yticklabels=text_label[1:],
            image=image
        )

        # Compute Accuracy and L1 Loss
        action_logits = output.logits[:, vla.vision_backbone.featurizer.patch_embed.num_patches : -1]
        action_preds = action_logits.argmax(dim=2)
        action_gt = labels[:, 1:].to(action_preds.device)

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


if __name__ == "__main__":
    eval()
