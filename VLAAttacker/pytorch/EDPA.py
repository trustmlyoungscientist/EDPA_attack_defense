import torch
from torchvision import transforms
from torchvision.utils import save_image

import numpy as np
import torch.nn.functional as F

import tqdm


## If using OFT, uncomment the following line and comment the next line
from utils.openvla import image_transform, get_img_embedding, get_lang_embedding
# from utils.openvla_oft import image_transform, get_img_embedding, get_lang_embedding


class EDPA:
    def __init__(self, cfg, device_id = None):
    
        self.max_steps = cfg.max_steps
        self.save_steps = cfg.save_steps

        self.step_size = cfg.step_size
        self.alpha = cfg.alpha
        self.iterations = cfg.iterations

        self.image_size = cfg.image_size
        self.perturbation_ratio = cfg.perturbation_ratio

        self.ema_step = 0
        self.ema_warmup = 100

        self.ema_patch, self.ema_align = 0, 0
        self.ema_decay = 0.9

        self.cfg = cfg
        self.device_id = device_id if device_id is not None else torch.device("cpu")

    def convert_to_tensor(self, images):
        return torch.stack([transforms.ToTensor()(image) for image in images])
    
    def preprocess_tensor_images(self, images, processor=None):
        return torch.stack([
            image_transform(img, processor) for img in images
        ])
    
    def apply_random_perturbation_to_tensor(self, images, perturbation):

        B, C, H, W = images.shape
        pc, ph, pw = perturbation.shape

        assert pc == C, "Perturbation must have the same number of channels as the input images."

        # perturbated_images = image_tensors.clone()
        perturbated_images = torch.zeros_like(images)

        for i in range(len(perturbated_images)):
            top = torch.randint(0, H - ph + 1, (1,)).item()
            left = torch.randint(0, W - pw + 1, (1,)).item()

            mask = torch.zeros_like(images[i])
            mask[:, top:top + ph, left:left + pw] = 1.0

            padded_perturb = torch.zeros_like(images[i])
            padded_perturb[:, top:top + ph, left:left + pw] = perturbation

            perturbated_images[i] = (1 - mask) * images[i] + padded_perturb
        
        return perturbated_images
    

    def compute_img_loss(self, out, targets, reduction='none'):
        # squared l2 - it does not divide by the latent dimension
        # should have shape (batch_size, embedding_size)
        assert out.shape == targets.shape, f'{out.shape} != {targets.shape}'
        # Compute the element-wise squared error

        squared_error_batch = torch.mean((1 - F.cosine_similarity(out, targets, dim=-1)), dim=-1)

        if reduction == 'mean':
            squared_error_batch = torch.mean(squared_error_batch)
        
        return squared_error_batch
        
    
    def compute_loss(self, adv, clean, text, text_mask, reduction='none'):

        # patch_loss = (1 - F.cosine_similarity(adv, clean, dim=-1)).mean(dim=-1)  
        # patch_loss = F.mse_loss(adv, clean, reduction='none').mean(dim=(1, 2))

        logits = torch.bmm(F.normalize(adv, dim=-1),
                        F.normalize(clean, dim=-1).transpose(1, 2)) / 0.07

        labels = torch.arange(logits.size(1), device=logits.device).unsqueeze(0).expand(logits.size(0), -1)

        patch_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),  
            labels.reshape(-1),
            reduction='none'
        ).reshape(logits.size(0), logits.size(1)).mean(dim=1)

        adv_sim = torch.bmm(F.normalize(adv, dim=-1), F.normalize(text, dim=-1).transpose(1, 2))   
        clean_sim = torch.bmm(F.normalize(clean, dim=-1), F.normalize(text, dim=-1).transpose(1, 2))
        
        # if text_mask is not None:
        if text_mask is not None:
            text_mask = text_mask.float().unsqueeze(1).to(self.device_id)
            adv_sim, clean_sim = adv_sim * text_mask, clean_sim * text_mask
            valid_counts = torch.clamp(text_mask.sum(dim=-1), min=1e-8)
            align_loss = torch.sum(torch.abs(adv_sim - clean_sim), dim=(1, 2)) / valid_counts.squeeze(-1)
        else:
            align_loss = torch.mean(torch.abs(adv_sim - clean_sim), dim=(1, 2)) 

        adjusted_decay = self.ema_decay * min(self.ema_step / self.ema_warmup, 1.0)

        ema_patch = (adjusted_decay * self.ema_patch + (1 - adjusted_decay) * torch.mean(patch_loss)).detach()
        ema_align = (adjusted_decay * self.ema_align + (1 - adjusted_decay) * torch.mean(align_loss)).detach()

        scaled_patch_loss = patch_loss / (ema_patch + 1e-8)
        scaled_align_loss = align_loss / (ema_align + 1e-8)    

        batch_loss = self.alpha * scaled_patch_loss + (1-self.alpha) * scaled_align_loss

        if reduction == 'mean':
            return torch.mean(batch_loss), ema_patch, ema_align, patch_loss, align_loss
        
        return batch_loss, ema_patch, ema_align, patch_loss, align_loss

    
    def apply_perturbation_to_raw_images(self, images, perturbation, position=(-1, -1)):

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
            np.array(transforms.ToPILImage()(img)) for img in perturbated_images
        ]

    def generate_one_step(
        self,
        vla,
        images,
        instructions,
        instructions_masks,
        processor,
        perturbation,
        images_embed=None,
        lang_embed=None,
        eval=True
    ):
        
        if images_embed is None:
            images = self.convert_to_tensor(images)
            images_embed = get_img_embedding(vla, self.preprocess_tensor_images(images, processor.image_processor))
        if lang_embed is None:
            lang_embed = get_lang_embedding(vla, instructions)

        for _ in range(self.iterations):

            perturbation = perturbation.detach().requires_grad_()

            adv_images = self.apply_random_perturbation_to_tensor(images, perturbation)
            adv_embed = get_img_embedding(vla, self.preprocess_tensor_images(adv_images, processor.image_processor))

            cost, ema_patch, ema_align, patch_loss, align_loss = self.compute_loss(
                adv_embed, images_embed, lang_embed, instructions_masks, reduction="mean"
            )

            grad = torch.autograd.grad(cost, perturbation, retain_graph=False, create_graph=False)[0]
            perturbation = torch.clamp(perturbation + self.step_size * grad.sign(), 0, 1)

            self.ema_step += 1
            self.ema_patch, self.ema_align = ema_patch, ema_align
        
        if eval:
            return perturbation, cost, patch_loss, align_loss
        else:
            return perturbation
        
    def generate(self, vla, dataloader, processor, action_tokenizer=None):
        
        patch_size = int(np.sqrt(self.image_size ** 2 * self.perturbation_ratio))
        perturbation = torch.zeros((3, patch_size, patch_size), dtype=torch.float32).uniform_(0, 1)

        if self.cfg.use_wandb:
            import wandb
            exp_id = (
                f"{self.cfg.vla_path.split('/')[-1]}+{self.cfg.dataset_name}"
                f"+ratio-{self.cfg.perturbation_ratio}"
                f"+iter-{self.iterations}"
            )
            wandb.init(entity=self.cfg.wandb_entity, project=self.cfg.wandb_project, name=f"ft+{exp_id}")

        with tqdm.tqdm(total=self.max_steps, leave=False) as progress:
            for idx, batch in enumerate(dataloader):

                input_key = "wrist_image" if self.cfg.camera_view == "wrist" else "image"
                images = self.convert_to_tensor(batch[input_key])
                instructions, instructions_masks = batch["input_ids"], batch["attention_mask"]

                images_embed = get_img_embedding(vla, self.preprocess_tensor_images(images, processor.image_processor))
                lang_embed = get_lang_embedding(vla, instructions)

                perturbation, cost, patch_loss, align_loss = self.generate_one_step(
                    vla, images, instructions, instructions_masks,
                    processor, perturbation,
                    images_embed=images_embed,
                    lang_embed=lang_embed
                )

                if self.cfg.use_wandb:
                    if self.cfg.experiment:
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            adv_images = self.apply_random_perturbation_to_tensor(images, perturbation)
                            output = vla(
                                input_ids=instructions.to(self.device_id),
                                attention_mask=instructions_masks.to(self.device_id),
                                pixel_values=self.preprocess_tensor_images(adv_images, processor.image_processor)
                                    .to(torch.bfloat16).to(self.device_id),
                                labels=batch["labels"]
                            )

                        action_logits = output.logits[:, vla.vision_backbone.featurizer.patch_embed.num_patches:-1]
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

                        wandb.log({
                            "cost": cost.item(),
                            "patch_loss": patch_loss.mean(),
                            "align_loss": align_loss.mean(),
                            "action_accuracy": action_accuracy.item(),
                            "action_l1_loss": action_l1_loss.item(),
                        }, step=idx)
                    else:
                        wandb.log({
                            "cost": cost.item(),
                            "patch_loss": patch_loss.mean(),
                            "align_loss": align_loss.mean(),
                        }, step=idx)

                if idx % self.save_steps == 0:
                    torch.save(perturbation, f"{self.cfg.save_path}-{self.perturbation_ratio}/perturbation.pt")
                    save_image(perturbation, f"{self.cfg.save_path}-{self.perturbation_ratio}/perturbation.png")

                progress.update()
                torch.cuda.empty_cache()

        return perturbation.cpu().detach()

