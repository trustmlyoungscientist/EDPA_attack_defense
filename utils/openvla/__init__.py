
import torch
import torchvision.transforms.functional as TVF


IGNORE_INDEX = -100

def generate_action_labels(
        action, action_tokenizer, tokenizer, predict_stop_token=False
    ) -> torch.Tensor:

    action_str = action_tokenizer(action)
    action_token_ids = tokenizer(action_str, add_special_tokens=False).input_ids

    labels = list(input_ids)

    labels[: -(len(action_token_ids) + 1)] = [IGNORE_INDEX] * (len(labels) - len(action_token_ids) - 1)

    if not predict_stop_token:
        labels[-1] = IGNORE_INDEX

    return torch.tensor(labels)


def build_prompt(language_instructions, prompt_template="In: What action should the robot take to {instruction}?\nOut:"):
    """
    Builds a list of prompts based on the provided language instructions.
    
    Args:
        language_instructions (list): List of language instructions.
        prompt_template (str): Template for the prompt. Defaults to "In: What action should the robot take to {instruction}?\nOut:".
    
    Returns:
        list: List of formatted prompts.
    """
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

def get_img_embedding(vla, images):
    return vla.projector(vla.vision_backbone(images.to(torch.bfloat16).to(vla.device)))
        

def get_lang_embedding(vla, language_tokens):
    return vla.get_input_embeddings()(language_tokens.to(vla.device))