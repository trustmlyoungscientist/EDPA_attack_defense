import jax
import jax.numpy as jnp
from jax import random
import numpy as np
from PIL import Image
import optax

from utils.pi0 import image_transform, get_img_embedding, get_lang_embedding

import tqdm

class EDPA:
    def __init__(self, cfg):

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

    def convert_images_format_and_normalize(self, images):
        return jnp.stack([jnp.array(image).transpose(2, 0, 1).astype(np.float32) / 255.0 for image in images])
        
    
    def preprocess_tensor_images(self, images):
        images = jnp.transpose(images, (0, 2, 3, 1))  # Convert from (B, C, H, W) to (B, H, W, C)
        return [
            # image_transform(self.cfg, img) for img in images
          image_transform(img) for img in images
        ]

    def apply_random_perturbation_to_jax(self, images, perturbation):
        B, C, H, W = images.shape
        pc, ph, pw = perturbation.shape

        assert pc == C, "Perturbation channel size must match image channel size."

        perturbated_images = jnp.zeros_like(images)

        for i in range(B):
            top = np.random.randint(0, H - ph + 1)
            left = np.random.randint(0, W - pw + 1)
            
            mask = jnp.zeros_like(images[i])
            mask = mask.at[:, top:top + ph, left:left + pw].set(1.0)

            padded_perturb = jnp.zeros_like(images[i])
            padded_perturb = padded_perturb.at[:, top:top + ph, left:left + pw].set(perturbation)

            perturbed_img = (1 - mask) * images[i] + padded_perturb
            perturbated_images = perturbated_images.at[i].set(perturbed_img)

        return self.preprocess_tensor_images(perturbated_images)

    def compute_loss_jax(self, adv, clean, text, text_mask=None, reduction='none'):
        # Patch-wise cosine similarity loss
        adv_norm = adv / (jnp.linalg.norm(adv, axis=-1, keepdims=True) + 1e-8)
        clean_norm = clean / (jnp.linalg.norm(clean, axis=-1, keepdims=True) + 1e-8)
        text_norm = text / (jnp.linalg.norm(text, axis=-1, keepdims=True) + 1e-8)
        
        # patch_loss = jnp.mean(1.0 - jnp.sum(adv_norm * clean_norm, axis=-1), axis=-1)  # shape: [B]

        logits = jnp.einsum('bij,bkj->bik', adv_norm, clean_norm) / 0.07

        patch_loss = optax.softmax_cross_entropy(
            logits.reshape(-1, logits.shape[1]),
            jax.nn.one_hot(
                jnp.tile(jnp.arange(logits.shape[1])[None, :], (logits.shape[0], 1)).reshape(-1),
                logits.shape[1]
            )
        ).reshape(logits.shape[0], logits.shape[1]).mean(axis=1)

            # Compute batch similarity matrices
        adv_sim = jnp.einsum('bij,bkj->bik', adv_norm, text_norm)
        clean_sim = jnp.einsum('bij,bkj->bik', clean_norm, text_norm)

        if text_mask is not None:
            text_mask = text_mask.astype(jnp.float32)[:, None, :]
            adv_sim, clean_sim = adv_sim * text_mask, clean_sim * text_mask
            valid_counts = jnp.maximum(jnp.sum(text_mask, axis=-1), 1e-8)  # [B, 1]
            align_loss = jnp.sum(jnp.abs(adv_sim - clean_sim), axis=(1, 2)) / valid_counts.squeeze(-1)
        else:
            align_loss = jnp.mean(jnp.abs(adv_sim - clean_sim), axis=(1, 2))  # [B]
        
        adjusted_decay = self.ema_decay * jnp.minimum(self.ema_step / self.ema_warmup, 1.0)

        ema_patch = adjusted_decay * self.ema_patch + (1 - adjusted_decay) * jnp.mean(patch_loss)
        ema_align = adjusted_decay * self.ema_align + (1 - adjusted_decay) * jnp.mean(align_loss)

        scaled_patch_loss = patch_loss / (ema_patch + 1e-8)
        scaled_align_loss = align_loss / (ema_align + 1e-8)

        batch_loss = self.alpha * scaled_patch_loss + (1-self.alpha) * scaled_align_loss

        if reduction == 'mean':
            return jnp.mean(batch_loss), ema_patch, ema_align, jnp.mean(patch_loss), jnp.mean(align_loss)
        
        return batch_loss, ema_patch, ema_align, patch_loss, align_loss
    
    def perturb(self, vla, images, instructions, instruction_masks=None, perturbation=None):
        
        if perturbation is None:
            patch_size = int(np.sqrt(self.cfg.image_size ** 2 * self.cfg.perturbation_ratio))
            perturbation = jax.random.uniform(random.PRNGKey(0), (3, patch_size, patch_size), minval=0.0, maxval=1.0)

        if instruction_masks is not None:
            instruction_masks = jnp.array(instruction_masks)

        images = self.convert_images_format_and_normalize(images)

        images_embed = get_img_embedding(vla, self.preprocess_tensor_images(images))

        lang_embed = get_lang_embedding(vla, instructions)

        def compute_loss(p):
            adv_images = self.apply_random_perturbation_to_jax(images, p)
            adv_embed = get_img_embedding(vla, adv_images)
            loss, ema_patch, ema_align = self.compute_loss_jax(adv_embed, images_embed, lang_embed, instruction_masks, reduction='mean')
            return loss, (ema_patch, ema_align)

        for _ in range(self.iterations):
            (loss, (ema_patch, ema_align)), grad = jax.value_and_grad(compute_loss, has_aux=True)(perturbation)
            
            self.ema_step = self.ema_step + 1
            self.ema_patch, self.ema_align = ema_patch, ema_align

            perturbation = jnp.clip(perturbation + self.step_size * jnp.sign(grad), 0, 1)

        return np.array(perturbation)
    
    
    def generate_one_step(
        self,
        vla,
        images,
        instructions,
        instructions_masks,
        perturbation,
        images_embed=None,
        lang_embed=None,
        eval=True
    ):
        
        if images_embed is None:
            images = self.convert_images_format_and_normalize(images)
            images_embed = get_img_embedding(vla, self.preprocess_tensor_images(images))
        if lang_embed is None:
            lang_embed = get_lang_embedding(vla, instructions)

        for _ in range(self.iterations):

            def compute_loss(p):
                adv_images = self.apply_random_perturbation_to_jax(images, p)
                adv_embed = get_img_embedding(vla, adv_images)
                loss, ema_patch, ema_align, patch_loss, align_loss = self.compute_loss_jax(adv_embed, images_embed, lang_embed, instructions_masks, reduction='mean')
                return loss, (ema_patch, ema_align, patch_loss, align_loss)

            for _ in range(self.iterations):
                (loss, (ema_patch, ema_align, patch_loss, align_loss)), grad = jax.value_and_grad(compute_loss, has_aux=True)(perturbation)
                
                self.ema_step = self.ema_step + 1
                self.ema_patch, self.ema_align = ema_patch, ema_align

                perturbation = jnp.clip(perturbation + self.step_size * jnp.sign(grad), 0, 1)

        if eval:
            return np.array(perturbation), loss, patch_loss, align_loss
        else:
            return np.array(perturbation)
        
    
    
    def generate(self, vla, dataloader, tokeizer):
        
        patch_size = int(np.sqrt(self.cfg.image_size ** 2 * self.cfg.perturbation_ratio))
        perturbation = jax.random.uniform(random.PRNGKey(0), (3, patch_size, patch_size), minval=0.0, maxval=1.0)

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
                images, (instructions, instruction_masks) = batch[input_key], zip(*[tokeizer.tokenize(instruction) for instruction in batch["language_instruction"]])
                
                instructions, instruction_masks = jnp.array(instructions), jnp.array(instruction_masks)
                
                images = self.convert_images_format_and_normalize(images)

                images_embed = get_img_embedding(vla, self.preprocess_tensor_images(images))
                lang_embed = get_lang_embedding(vla, instructions)

                perturbation, cost, patch_loss, align_loss = self.generate_one_step(
                    vla, images, instructions, instruction_masks, perturbation,
                    images_embed=images_embed,
                    lang_embed=lang_embed
                )

                if self.cfg.use_wandb:
                    wandb.log({
                        "cost": cost.item(),
                        "patch_loss": patch_loss.mean(),
                        "align_loss": align_loss.mean(),
                    }, step=idx)

                if idx % self.save_steps == 0:
                    
                    np.save(f"{self.cfg.save_path}-{self.perturbation_ratio}/perturbation.npy", perturbation)
                    Image.fromarray(
                        ((perturbation - np.min(perturbation)) / (np.ptp(perturbation) + 1e-8) * 255)
                        .astype(np.uint8)
                        .transpose(1, 2, 0)
                    ).save(f"{self.cfg.save_path}-{self.perturbation_ratio}/perturbation.png")

                progress.update()

        return perturbation.cpu().detach()