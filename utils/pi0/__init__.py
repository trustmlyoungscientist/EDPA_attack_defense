
import jax
import jax.numpy as jnp

from openpi.shared import image_tools

def image_transform(image):
    images = image_tools.resize_with_pad(image * 2.0 - 1.0, 224, 224)   # Normalize to [-1, +1]
    return jax.tree.map(lambda x: jnp.asarray(x), images)

def get_img_embedding(vla, images):     
    if not hasattr(get_img_embedding, "_jit_embed_fn"):
        get_img_embedding._jit_embed_fn = jax.jit(
            lambda x: vla.PaliGemma.img(x, train=False)[0]
        )
    embed_fn = get_img_embedding._jit_embed_fn
    return embed_fn(images)

def get_lang_embedding(vla, language_tokens):
    if not hasattr(get_lang_embedding, "_jit_embed_fn"):
        get_lang_embedding._jit_embed_fn = jax.jit(
            lambda x: vla.PaliGemma.llm(x, embed_only=True)
        )
    embed_fn = get_lang_embedding._jit_embed_fn
    return embed_fn(language_tokens)
