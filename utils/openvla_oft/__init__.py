import torch
import torchvision.transforms.functional as TVF


def build_prompt(language_instructions, prompt_template="In: What action should the robot take to {instruction}?\nOut:"):
    return [prompt_template.format(instruction=instruction.lower()) for instruction in language_instructions]

def image_transform(image, image_processor):

    if image_processor.tvf_do_letterbox:
        image = image_processor.letterbox_pad_transform(image, image_processor.tvf_letterbox_fill)

    imgs_t = []
    for idx in range(len(image_processor.input_sizes)):
        img_idx = TVF.resize(image, **image_processor.tvf_resize_params[idx])
        img_idx = TVF.center_crop(img_idx, **image_processor.tvf_crop_params[idx])
        # img_idx_t = TVF.to_tensor(img_idx)
        img_idx_t = TVF.normalize(img_idx, **image_processor.tvf_normalize_params[idx])
        imgs_t.append(img_idx_t)

    img_t = torch.vstack(imgs_t)

    return img_t

def get_img_embedding(vla, images, language_embeddings=None, use_film=False):
    """
    Extract image feature embeddings, compatible with FiLM modulation (language_embeddings is only needed if use_film=True).
    language_embeddings: (optional) Language feature tensor (needed if using FiLM)
    use_film: Whether to use FiLM modulation
    """
    if use_film:
        patch_features = vla.vision_backbone(images.to(torch.bfloat16).to(vla.device), language_embeddings)  # Ensure the backbone supports this interface
    else:
        patch_features = vla.vision_backbone(images.to(torch.bfloat16).to(vla.device))
    return vla.projector(patch_features)

def get_lang_embedding(vla, language_tokens):
    return vla.get_input_embeddings()(language_tokens.to(vla.device))
