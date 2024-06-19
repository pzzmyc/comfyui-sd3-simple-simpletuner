import torch
from accelerate import init_empty_weights
from accelerate.utils.modeling import set_module_tensor_to_device
from safetensors.torch import load_file, save_file
from transformers import CLIPTextModel, CLIPTextConfig, CLIPTextModelWithProjection, CLIPTokenizer
from typing import List
import xformers
from diffusers import AutoencoderKL, EulerDiscreteScheduler, UNet2DConditionModel

import logging
logger = logging.getLogger(__name__)
VAE_SCALE_FACTOR = 0.13025
MODEL_VERSION_SDXL_BASE_V1_0 = "sdxl_base_v1-0"

# Diffusersの設定を読み込むための参照モデル
DIFFUSERS_REF_MODEL_ID_SDXL = "stabilityai/stable-diffusion-xl-base-1.0"

DIFFUSERS_SDXL_UNET_CONFIG = {
    "act_fn": "silu",
    "addition_embed_type": "text_time",
    "addition_embed_type_num_heads": 64,
    "addition_time_embed_dim": 256,
    "attention_head_dim": [5, 10, 20],
    "block_out_channels": [320, 640, 1280],
    "center_input_sample": False,
    "class_embed_type": None,
    "class_embeddings_concat": False,
    "conv_in_kernel": 3,
    "conv_out_kernel": 3,
    "cross_attention_dim": 2048,
    "cross_attention_norm": None,
    "down_block_types": ["DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"],
    "downsample_padding": 1,
    "dual_cross_attention": False,
    "encoder_hid_dim": None,
    "encoder_hid_dim_type": None,
    "flip_sin_to_cos": True,
    "freq_shift": 0,
    "in_channels": 4,
    "layers_per_block": 2,
    "mid_block_only_cross_attention": None,
    "mid_block_scale_factor": 1,
    "mid_block_type": "UNetMidBlock2DCrossAttn",
    "norm_eps": 1e-05,
    "norm_num_groups": 32,
    "num_attention_heads": None,
    "num_class_embeds": None,
    "only_cross_attention": False,
    "out_channels": 4,
    "projection_class_embeddings_input_dim": 2816,
    "resnet_out_scale_factor": 1.0,
    "resnet_skip_time_act": False,
    "resnet_time_scale_shift": "default",
    "sample_size": 128,
    "time_cond_proj_dim": None,
    "time_embedding_act_fn": None,
    "time_embedding_dim": None,
    "time_embedding_type": "positional",
    "timestep_post_act": None,
    "transformer_layers_per_block": [1, 2, 10],
    "up_block_types": ["CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"],
    "upcast_attention": False,
    "use_linear_projection": True,
}



# v1: split from train_db_fixed.py.
# v2: support safetensors

import math
import os

import torch



import diffusers
from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextConfig, logging
from diffusers import AutoencoderKL, DDIMScheduler, StableDiffusionPipeline  # , UNet2DConditionModel
from safetensors.torch import load_file, save_file

import logging
logger = logging.getLogger(__name__)

# DiffUsers版StableDiffusionのモデルパラメータ
NUM_TRAIN_TIMESTEPS = 1000
BETA_START = 0.00085
BETA_END = 0.0120

UNET_PARAMS_MODEL_CHANNELS = 320
UNET_PARAMS_CHANNEL_MULT = [1, 2, 4, 4]
UNET_PARAMS_ATTENTION_RESOLUTIONS = [4, 2, 1]
UNET_PARAMS_IMAGE_SIZE = 64  # fixed from old invalid value `32`
UNET_PARAMS_IN_CHANNELS = 4
UNET_PARAMS_OUT_CHANNELS = 4
UNET_PARAMS_NUM_RES_BLOCKS = 2
UNET_PARAMS_CONTEXT_DIM = 768
UNET_PARAMS_NUM_HEADS = 8
# UNET_PARAMS_USE_LINEAR_PROJECTION = False

VAE_PARAMS_Z_CHANNELS = 4
VAE_PARAMS_RESOLUTION = 256
VAE_PARAMS_IN_CHANNELS = 3
VAE_PARAMS_OUT_CH = 3
VAE_PARAMS_CH = 128
VAE_PARAMS_CH_MULT = [1, 2, 4, 4]
VAE_PARAMS_NUM_RES_BLOCKS = 2

# V2
V2_UNET_PARAMS_ATTENTION_HEAD_DIM = [5, 10, 20, 20]
V2_UNET_PARAMS_CONTEXT_DIM = 1024
# V2_UNET_PARAMS_USE_LINEAR_PROJECTION = True

# Diffusersの設定を読み込むための参照モデル
DIFFUSERS_REF_MODEL_ID_V1 = "runwayml/stable-diffusion-v1-5"
DIFFUSERS_REF_MODEL_ID_V2 = "stabilityai/stable-diffusion-2-1"


# region StableDiffusion->Diffusersの変換コード
# convert_original_stable_diffusion_to_diffusers をコピーして修正している（ASL 2.0）


def shave_segments(path, n_shave_prefix_segments=1):
    """
    Removes segments. Positive values shave the first segments, negative shave the last segments.
    """
    if n_shave_prefix_segments >= 0:
        return ".".join(path.split(".")[n_shave_prefix_segments:])
    else:
        return ".".join(path.split(".")[:n_shave_prefix_segments])


def renew_resnet_paths(old_list, n_shave_prefix_segments=0):
    """
    Updates paths inside resnets to the new naming scheme (local renaming)
    """
    mapping = []
    for old_item in old_list:
        new_item = old_item.replace("in_layers.0", "norm1")
        new_item = new_item.replace("in_layers.2", "conv1")

        new_item = new_item.replace("out_layers.0", "norm2")
        new_item = new_item.replace("out_layers.3", "conv2")

        new_item = new_item.replace("emb_layers.1", "time_emb_proj")
        new_item = new_item.replace("skip_connection", "conv_shortcut")

        new_item = shave_segments(new_item, n_shave_prefix_segments=n_shave_prefix_segments)

        mapping.append({"old": old_item, "new": new_item})

    return mapping


def renew_vae_resnet_paths(old_list, n_shave_prefix_segments=0):
    """
    Updates paths inside resnets to the new naming scheme (local renaming)
    """
    mapping = []
    for old_item in old_list:
        new_item = old_item

        new_item = new_item.replace("nin_shortcut", "conv_shortcut")
        new_item = shave_segments(new_item, n_shave_prefix_segments=n_shave_prefix_segments)

        mapping.append({"old": old_item, "new": new_item})

    return mapping


def renew_attention_paths(old_list, n_shave_prefix_segments=0):
    """
    Updates paths inside attentions to the new naming scheme (local renaming)
    """
    mapping = []
    for old_item in old_list:
        new_item = old_item

        #         new_item = new_item.replace('norm.weight', 'group_norm.weight')
        #         new_item = new_item.replace('norm.bias', 'group_norm.bias')

        #         new_item = new_item.replace('proj_out.weight', 'proj_attn.weight')
        #         new_item = new_item.replace('proj_out.bias', 'proj_attn.bias')

        #         new_item = shave_segments(new_item, n_shave_prefix_segments=n_shave_prefix_segments)

        mapping.append({"old": old_item, "new": new_item})

    return mapping


def renew_vae_attention_paths(old_list, n_shave_prefix_segments=0):
    """
    Updates paths inside attentions to the new naming scheme (local renaming)
    """
    mapping = []
    for old_item in old_list:
        new_item = old_item

        new_item = new_item.replace("norm.weight", "group_norm.weight")
        new_item = new_item.replace("norm.bias", "group_norm.bias")

        if diffusers.__version__ < "0.17.0":
            new_item = new_item.replace("q.weight", "query.weight")
            new_item = new_item.replace("q.bias", "query.bias")

            new_item = new_item.replace("k.weight", "key.weight")
            new_item = new_item.replace("k.bias", "key.bias")

            new_item = new_item.replace("v.weight", "value.weight")
            new_item = new_item.replace("v.bias", "value.bias")

            new_item = new_item.replace("proj_out.weight", "proj_attn.weight")
            new_item = new_item.replace("proj_out.bias", "proj_attn.bias")
        else:
            new_item = new_item.replace("q.weight", "to_q.weight")
            new_item = new_item.replace("q.bias", "to_q.bias")

            new_item = new_item.replace("k.weight", "to_k.weight")
            new_item = new_item.replace("k.bias", "to_k.bias")

            new_item = new_item.replace("v.weight", "to_v.weight")
            new_item = new_item.replace("v.bias", "to_v.bias")

            new_item = new_item.replace("proj_out.weight", "to_out.0.weight")
            new_item = new_item.replace("proj_out.bias", "to_out.0.bias")

        new_item = shave_segments(new_item, n_shave_prefix_segments=n_shave_prefix_segments)

        mapping.append({"old": old_item, "new": new_item})

    return mapping


def assign_to_checkpoint(
    paths, checkpoint, old_checkpoint, attention_paths_to_split=None, additional_replacements=None, config=None
):
    """
    This does the final conversion step: take locally converted weights and apply a global renaming
    to them. It splits attention layers, and takes into account additional replacements
    that may arise.

    Assigns the weights to the new checkpoint.
    """
    assert isinstance(paths, list), "Paths should be a list of dicts containing 'old' and 'new' keys."

    # Splits the attention layers into three variables.
    if attention_paths_to_split is not None:
        for path, path_map in attention_paths_to_split.items():
            old_tensor = old_checkpoint[path]
            channels = old_tensor.shape[0] // 3

            target_shape = (-1, channels) if len(old_tensor.shape) == 3 else (-1)

            num_heads = old_tensor.shape[0] // config["num_head_channels"] // 3

            old_tensor = old_tensor.reshape((num_heads, 3 * channels // num_heads) + old_tensor.shape[1:])
            query, key, value = old_tensor.split(channels // num_heads, dim=1)

            checkpoint[path_map["query"]] = query.reshape(target_shape)
            checkpoint[path_map["key"]] = key.reshape(target_shape)
            checkpoint[path_map["value"]] = value.reshape(target_shape)

    for path in paths:
        new_path = path["new"]

        # These have already been assigned
        if attention_paths_to_split is not None and new_path in attention_paths_to_split:
            continue

        # Global renaming happens here
        new_path = new_path.replace("middle_block.0", "mid_block.resnets.0")
        new_path = new_path.replace("middle_block.1", "mid_block.attentions.0")
        new_path = new_path.replace("middle_block.2", "mid_block.resnets.1")

        if additional_replacements is not None:
            for replacement in additional_replacements:
                new_path = new_path.replace(replacement["old"], replacement["new"])

        # proj_attn.weight has to be converted from conv 1D to linear
        reshaping = False
        if diffusers.__version__ < "0.17.0":
            if "proj_attn.weight" in new_path:
                reshaping = True
        else:
            if ".attentions." in new_path and ".0.to_" in new_path and old_checkpoint[path["old"]].ndim > 2:
                reshaping = True

        if reshaping:
            checkpoint[new_path] = old_checkpoint[path["old"]][:, :, 0, 0]
        else:
            checkpoint[new_path] = old_checkpoint[path["old"]]


def conv_attn_to_linear(checkpoint):
    keys = list(checkpoint.keys())
    attn_keys = ["query.weight", "key.weight", "value.weight"]
    for key in keys:
        if ".".join(key.split(".")[-2:]) in attn_keys:
            if checkpoint[key].ndim > 2:
                checkpoint[key] = checkpoint[key][:, :, 0, 0]
        elif "proj_attn.weight" in key:
            if checkpoint[key].ndim > 2:
                checkpoint[key] = checkpoint[key][:, :, 0]


def linear_transformer_to_conv(checkpoint):
    keys = list(checkpoint.keys())
    tf_keys = ["proj_in.weight", "proj_out.weight"]
    for key in keys:
        if ".".join(key.split(".")[-2:]) in tf_keys:
            if checkpoint[key].ndim == 2:
                checkpoint[key] = checkpoint[key].unsqueeze(2).unsqueeze(2)


def convert_ldm_unet_checkpoint(v2, checkpoint, config):
    """
    Takes a state dict and a config, and returns a converted checkpoint.
    """

    # extract state_dict for UNet
    unet_state_dict = {}
    unet_key = "model.diffusion_model."
    keys = list(checkpoint.keys())
    for key in keys:
        if key.startswith(unet_key):
            unet_state_dict[key.replace(unet_key, "")] = checkpoint.pop(key)

    new_checkpoint = {}

    new_checkpoint["time_embedding.linear_1.weight"] = unet_state_dict["time_embed.0.weight"]
    new_checkpoint["time_embedding.linear_1.bias"] = unet_state_dict["time_embed.0.bias"]
    new_checkpoint["time_embedding.linear_2.weight"] = unet_state_dict["time_embed.2.weight"]
    new_checkpoint["time_embedding.linear_2.bias"] = unet_state_dict["time_embed.2.bias"]

    new_checkpoint["conv_in.weight"] = unet_state_dict["input_blocks.0.0.weight"]
    new_checkpoint["conv_in.bias"] = unet_state_dict["input_blocks.0.0.bias"]

    new_checkpoint["conv_norm_out.weight"] = unet_state_dict["out.0.weight"]
    new_checkpoint["conv_norm_out.bias"] = unet_state_dict["out.0.bias"]
    new_checkpoint["conv_out.weight"] = unet_state_dict["out.2.weight"]
    new_checkpoint["conv_out.bias"] = unet_state_dict["out.2.bias"]

    # Retrieves the keys for the input blocks only
    num_input_blocks = len({".".join(layer.split(".")[:2]) for layer in unet_state_dict if "input_blocks" in layer})
    input_blocks = {
        layer_id: [key for key in unet_state_dict if f"input_blocks.{layer_id}." in key] for layer_id in range(num_input_blocks)
    }

    # Retrieves the keys for the middle blocks only
    num_middle_blocks = len({".".join(layer.split(".")[:2]) for layer in unet_state_dict if "middle_block" in layer})
    middle_blocks = {
        layer_id: [key for key in unet_state_dict if f"middle_block.{layer_id}." in key] for layer_id in range(num_middle_blocks)
    }

    # Retrieves the keys for the output blocks only
    num_output_blocks = len({".".join(layer.split(".")[:2]) for layer in unet_state_dict if "output_blocks" in layer})
    output_blocks = {
        layer_id: [key for key in unet_state_dict if f"output_blocks.{layer_id}." in key] for layer_id in range(num_output_blocks)
    }

    for i in range(1, num_input_blocks):
        block_id = (i - 1) // (config["layers_per_block"] + 1)
        layer_in_block_id = (i - 1) % (config["layers_per_block"] + 1)

        resnets = [key for key in input_blocks[i] if f"input_blocks.{i}.0" in key and f"input_blocks.{i}.0.op" not in key]
        attentions = [key for key in input_blocks[i] if f"input_blocks.{i}.1" in key]

        if f"input_blocks.{i}.0.op.weight" in unet_state_dict:
            new_checkpoint[f"down_blocks.{block_id}.downsamplers.0.conv.weight"] = unet_state_dict.pop(
                f"input_blocks.{i}.0.op.weight"
            )
            new_checkpoint[f"down_blocks.{block_id}.downsamplers.0.conv.bias"] = unet_state_dict.pop(f"input_blocks.{i}.0.op.bias")

        paths = renew_resnet_paths(resnets)
        meta_path = {"old": f"input_blocks.{i}.0", "new": f"down_blocks.{block_id}.resnets.{layer_in_block_id}"}
        assign_to_checkpoint(paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config)

        if len(attentions):
            paths = renew_attention_paths(attentions)
            meta_path = {"old": f"input_blocks.{i}.1", "new": f"down_blocks.{block_id}.attentions.{layer_in_block_id}"}
            assign_to_checkpoint(paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config)

    resnet_0 = middle_blocks[0]
    attentions = middle_blocks[1]
    resnet_1 = middle_blocks[2]

    resnet_0_paths = renew_resnet_paths(resnet_0)
    assign_to_checkpoint(resnet_0_paths, new_checkpoint, unet_state_dict, config=config)

    resnet_1_paths = renew_resnet_paths(resnet_1)
    assign_to_checkpoint(resnet_1_paths, new_checkpoint, unet_state_dict, config=config)

    attentions_paths = renew_attention_paths(attentions)
    meta_path = {"old": "middle_block.1", "new": "mid_block.attentions.0"}
    assign_to_checkpoint(attentions_paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config)

    for i in range(num_output_blocks):
        block_id = i // (config["layers_per_block"] + 1)
        layer_in_block_id = i % (config["layers_per_block"] + 1)
        output_block_layers = [shave_segments(name, 2) for name in output_blocks[i]]
        output_block_list = {}

        for layer in output_block_layers:
            layer_id, layer_name = layer.split(".")[0], shave_segments(layer, 1)
            if layer_id in output_block_list:
                output_block_list[layer_id].append(layer_name)
            else:
                output_block_list[layer_id] = [layer_name]

        if len(output_block_list) > 1:
            resnets = [key for key in output_blocks[i] if f"output_blocks.{i}.0" in key]
            attentions = [key for key in output_blocks[i] if f"output_blocks.{i}.1" in key]

            resnet_0_paths = renew_resnet_paths(resnets)
            paths = renew_resnet_paths(resnets)

            meta_path = {"old": f"output_blocks.{i}.0", "new": f"up_blocks.{block_id}.resnets.{layer_in_block_id}"}
            assign_to_checkpoint(paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config)

            # オリジナル：
            # if ["conv.weight", "conv.bias"] in output_block_list.values():
            #   index = list(output_block_list.values()).index(["conv.weight", "conv.bias"])

            # biasとweightの順番に依存しないようにする：もっといいやり方がありそうだが
            for l in output_block_list.values():
                l.sort()

            if ["conv.bias", "conv.weight"] in output_block_list.values():
                index = list(output_block_list.values()).index(["conv.bias", "conv.weight"])
                new_checkpoint[f"up_blocks.{block_id}.upsamplers.0.conv.bias"] = unet_state_dict[
                    f"output_blocks.{i}.{index}.conv.bias"
                ]
                new_checkpoint[f"up_blocks.{block_id}.upsamplers.0.conv.weight"] = unet_state_dict[
                    f"output_blocks.{i}.{index}.conv.weight"
                ]

                # Clear attentions as they have been attributed above.
                if len(attentions) == 2:
                    attentions = []

            if len(attentions):
                paths = renew_attention_paths(attentions)
                meta_path = {
                    "old": f"output_blocks.{i}.1",
                    "new": f"up_blocks.{block_id}.attentions.{layer_in_block_id}",
                }
                assign_to_checkpoint(paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config)
        else:
            resnet_0_paths = renew_resnet_paths(output_block_layers, n_shave_prefix_segments=1)
            for path in resnet_0_paths:
                old_path = ".".join(["output_blocks", str(i), path["old"]])
                new_path = ".".join(["up_blocks", str(block_id), "resnets", str(layer_in_block_id), path["new"]])

                new_checkpoint[new_path] = unet_state_dict[old_path]

    # SDのv2では1*1のconv2dがlinearに変わっている
    # 誤って Diffusers 側を conv2d のままにしてしまったので、変換必要
    if v2 and not config.get("use_linear_projection", False):
        linear_transformer_to_conv(new_checkpoint)

    return new_checkpoint


def convert_ldm_vae_checkpoint(checkpoint, config):
    # extract state dict for VAE
    vae_state_dict = {}
    vae_key = "first_stage_model."
    keys = list(checkpoint.keys())
    for key in keys:
        if key.startswith(vae_key):
            vae_state_dict[key.replace(vae_key, "")] = checkpoint.get(key)
    # if len(vae_state_dict) == 0:
    #   # 渡されたcheckpointは.ckptから読み込んだcheckpointではなくvaeのstate_dict
    #   vae_state_dict = checkpoint

    new_checkpoint = {}

    new_checkpoint["encoder.conv_in.weight"] = vae_state_dict["encoder.conv_in.weight"]
    new_checkpoint["encoder.conv_in.bias"] = vae_state_dict["encoder.conv_in.bias"]
    new_checkpoint["encoder.conv_out.weight"] = vae_state_dict["encoder.conv_out.weight"]
    new_checkpoint["encoder.conv_out.bias"] = vae_state_dict["encoder.conv_out.bias"]
    new_checkpoint["encoder.conv_norm_out.weight"] = vae_state_dict["encoder.norm_out.weight"]
    new_checkpoint["encoder.conv_norm_out.bias"] = vae_state_dict["encoder.norm_out.bias"]

    new_checkpoint["decoder.conv_in.weight"] = vae_state_dict["decoder.conv_in.weight"]
    new_checkpoint["decoder.conv_in.bias"] = vae_state_dict["decoder.conv_in.bias"]
    new_checkpoint["decoder.conv_out.weight"] = vae_state_dict["decoder.conv_out.weight"]
    new_checkpoint["decoder.conv_out.bias"] = vae_state_dict["decoder.conv_out.bias"]
    new_checkpoint["decoder.conv_norm_out.weight"] = vae_state_dict["decoder.norm_out.weight"]
    new_checkpoint["decoder.conv_norm_out.bias"] = vae_state_dict["decoder.norm_out.bias"]

    new_checkpoint["quant_conv.weight"] = vae_state_dict["quant_conv.weight"]
    new_checkpoint["quant_conv.bias"] = vae_state_dict["quant_conv.bias"]
    new_checkpoint["post_quant_conv.weight"] = vae_state_dict["post_quant_conv.weight"]
    new_checkpoint["post_quant_conv.bias"] = vae_state_dict["post_quant_conv.bias"]

    # Retrieves the keys for the encoder down blocks only
    num_down_blocks = len({".".join(layer.split(".")[:3]) for layer in vae_state_dict if "encoder.down" in layer})
    down_blocks = {layer_id: [key for key in vae_state_dict if f"down.{layer_id}" in key] for layer_id in range(num_down_blocks)}

    # Retrieves the keys for the decoder up blocks only
    num_up_blocks = len({".".join(layer.split(".")[:3]) for layer in vae_state_dict if "decoder.up" in layer})
    up_blocks = {layer_id: [key for key in vae_state_dict if f"up.{layer_id}" in key] for layer_id in range(num_up_blocks)}

    for i in range(num_down_blocks):
        resnets = [key for key in down_blocks[i] if f"down.{i}" in key and f"down.{i}.downsample" not in key]

        if f"encoder.down.{i}.downsample.conv.weight" in vae_state_dict:
            new_checkpoint[f"encoder.down_blocks.{i}.downsamplers.0.conv.weight"] = vae_state_dict.pop(
                f"encoder.down.{i}.downsample.conv.weight"
            )
            new_checkpoint[f"encoder.down_blocks.{i}.downsamplers.0.conv.bias"] = vae_state_dict.pop(
                f"encoder.down.{i}.downsample.conv.bias"
            )

        paths = renew_vae_resnet_paths(resnets)
        meta_path = {"old": f"down.{i}.block", "new": f"down_blocks.{i}.resnets"}
        assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)

    mid_resnets = [key for key in vae_state_dict if "encoder.mid.block" in key]
    num_mid_res_blocks = 2
    for i in range(1, num_mid_res_blocks + 1):
        resnets = [key for key in mid_resnets if f"encoder.mid.block_{i}" in key]

        paths = renew_vae_resnet_paths(resnets)
        meta_path = {"old": f"mid.block_{i}", "new": f"mid_block.resnets.{i - 1}"}
        assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)

    mid_attentions = [key for key in vae_state_dict if "encoder.mid.attn" in key]
    paths = renew_vae_attention_paths(mid_attentions)
    meta_path = {"old": "mid.attn_1", "new": "mid_block.attentions.0"}
    assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)
    conv_attn_to_linear(new_checkpoint)

    for i in range(num_up_blocks):
        block_id = num_up_blocks - 1 - i
        resnets = [key for key in up_blocks[block_id] if f"up.{block_id}" in key and f"up.{block_id}.upsample" not in key]

        if f"decoder.up.{block_id}.upsample.conv.weight" in vae_state_dict:
            new_checkpoint[f"decoder.up_blocks.{i}.upsamplers.0.conv.weight"] = vae_state_dict[
                f"decoder.up.{block_id}.upsample.conv.weight"
            ]
            new_checkpoint[f"decoder.up_blocks.{i}.upsamplers.0.conv.bias"] = vae_state_dict[
                f"decoder.up.{block_id}.upsample.conv.bias"
            ]

        paths = renew_vae_resnet_paths(resnets)
        meta_path = {"old": f"up.{block_id}.block", "new": f"up_blocks.{i}.resnets"}
        assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)

    mid_resnets = [key for key in vae_state_dict if "decoder.mid.block" in key]
    num_mid_res_blocks = 2
    for i in range(1, num_mid_res_blocks + 1):
        resnets = [key for key in mid_resnets if f"decoder.mid.block_{i}" in key]

        paths = renew_vae_resnet_paths(resnets)
        meta_path = {"old": f"mid.block_{i}", "new": f"mid_block.resnets.{i - 1}"}
        assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)

    mid_attentions = [key for key in vae_state_dict if "decoder.mid.attn" in key]
    paths = renew_vae_attention_paths(mid_attentions)
    meta_path = {"old": "mid.attn_1", "new": "mid_block.attentions.0"}
    assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)
    conv_attn_to_linear(new_checkpoint)
    return new_checkpoint


def create_unet_diffusers_config(v2, use_linear_projection_in_v2=False):
    """
    Creates a config for the diffusers based on the config of the LDM model.
    """
    # unet_params = original_config.model.params.unet_config.params

    block_out_channels = [UNET_PARAMS_MODEL_CHANNELS * mult for mult in UNET_PARAMS_CHANNEL_MULT]

    down_block_types = []
    resolution = 1
    for i in range(len(block_out_channels)):
        block_type = "CrossAttnDownBlock2D" if resolution in UNET_PARAMS_ATTENTION_RESOLUTIONS else "DownBlock2D"
        down_block_types.append(block_type)
        if i != len(block_out_channels) - 1:
            resolution *= 2

    up_block_types = []
    for i in range(len(block_out_channels)):
        block_type = "CrossAttnUpBlock2D" if resolution in UNET_PARAMS_ATTENTION_RESOLUTIONS else "UpBlock2D"
        up_block_types.append(block_type)
        resolution //= 2

    config = dict(
        sample_size=UNET_PARAMS_IMAGE_SIZE,
        in_channels=UNET_PARAMS_IN_CHANNELS,
        out_channels=UNET_PARAMS_OUT_CHANNELS,
        down_block_types=tuple(down_block_types),
        up_block_types=tuple(up_block_types),
        block_out_channels=tuple(block_out_channels),
        layers_per_block=UNET_PARAMS_NUM_RES_BLOCKS,
        cross_attention_dim=UNET_PARAMS_CONTEXT_DIM if not v2 else V2_UNET_PARAMS_CONTEXT_DIM,
        attention_head_dim=UNET_PARAMS_NUM_HEADS if not v2 else V2_UNET_PARAMS_ATTENTION_HEAD_DIM,
        # use_linear_projection=UNET_PARAMS_USE_LINEAR_PROJECTION if not v2 else V2_UNET_PARAMS_USE_LINEAR_PROJECTION,
    )
    if v2 and use_linear_projection_in_v2:
        config["use_linear_projection"] = True

    return config


def create_vae_diffusers_config():
    """
    Creates a config for the diffusers based on the config of the LDM model.
    """
    # vae_params = original_config.model.params.first_stage_config.params.ddconfig
    # _ = original_config.model.params.first_stage_config.params.embed_dim
    block_out_channels = [VAE_PARAMS_CH * mult for mult in VAE_PARAMS_CH_MULT]
    down_block_types = ["DownEncoderBlock2D"] * len(block_out_channels)
    up_block_types = ["UpDecoderBlock2D"] * len(block_out_channels)

    config = dict(
        sample_size=VAE_PARAMS_RESOLUTION,
        in_channels=VAE_PARAMS_IN_CHANNELS,
        out_channels=VAE_PARAMS_OUT_CH,
        down_block_types=tuple(down_block_types),
        up_block_types=tuple(up_block_types),
        block_out_channels=tuple(block_out_channels),
        latent_channels=VAE_PARAMS_Z_CHANNELS,
        layers_per_block=VAE_PARAMS_NUM_RES_BLOCKS,
    )
    return config


def convert_ldm_clip_checkpoint_v1(checkpoint):
    keys = list(checkpoint.keys())
    text_model_dict = {}
    for key in keys:
        if key.startswith("cond_stage_model.transformer"):
            text_model_dict[key[len("cond_stage_model.transformer.") :]] = checkpoint[key]

    # remove position_ids for newer transformer, which causes error :(
    if "text_model.embeddings.position_ids" in text_model_dict:
        text_model_dict.pop("text_model.embeddings.position_ids")

    return text_model_dict


def convert_ldm_clip_checkpoint_v2(checkpoint, max_length):
    # 嫌になるくらい違うぞ！
    def convert_key(key):
        if not key.startswith("cond_stage_model"):
            return None

        # common conversion
        key = key.replace("cond_stage_model.model.transformer.", "text_model.encoder.")
        key = key.replace("cond_stage_model.model.", "text_model.")

        if "resblocks" in key:
            # resblocks conversion
            key = key.replace(".resblocks.", ".layers.")
            if ".ln_" in key:
                key = key.replace(".ln_", ".layer_norm")
            elif ".mlp." in key:
                key = key.replace(".c_fc.", ".fc1.")
                key = key.replace(".c_proj.", ".fc2.")
            elif ".attn.out_proj" in key:
                key = key.replace(".attn.out_proj.", ".self_attn.out_proj.")
            elif ".attn.in_proj" in key:
                key = None  # 特殊なので後で処理する
            else:
                raise ValueError(f"unexpected key in SD: {key}")
        elif ".positional_embedding" in key:
            key = key.replace(".positional_embedding", ".embeddings.position_embedding.weight")
        elif ".text_projection" in key:
            key = None  # 使われない???
        elif ".logit_scale" in key:
            key = None  # 使われない???
        elif ".token_embedding" in key:
            key = key.replace(".token_embedding.weight", ".embeddings.token_embedding.weight")
        elif ".ln_final" in key:
            key = key.replace(".ln_final", ".final_layer_norm")
        return key

    keys = list(checkpoint.keys())
    new_sd = {}
    for key in keys:
        # remove resblocks 23
        if ".resblocks.23." in key:
            continue
        new_key = convert_key(key)
        if new_key is None:
            continue
        new_sd[new_key] = checkpoint[key]

    # attnの変換
    for key in keys:
        if ".resblocks.23." in key:
            continue
        if ".resblocks" in key and ".attn.in_proj_" in key:
            # 三つに分割
            values = torch.chunk(checkpoint[key], 3)

            key_suffix = ".weight" if "weight" in key else ".bias"
            key_pfx = key.replace("cond_stage_model.model.transformer.resblocks.", "text_model.encoder.layers.")
            key_pfx = key_pfx.replace("_weight", "")
            key_pfx = key_pfx.replace("_bias", "")
            key_pfx = key_pfx.replace(".attn.in_proj", ".self_attn.")
            new_sd[key_pfx + "q_proj" + key_suffix] = values[0]
            new_sd[key_pfx + "k_proj" + key_suffix] = values[1]
            new_sd[key_pfx + "v_proj" + key_suffix] = values[2]

    # rename or add position_ids
    ANOTHER_POSITION_IDS_KEY = "text_model.encoder.text_model.embeddings.position_ids"
    if ANOTHER_POSITION_IDS_KEY in new_sd:
        # waifu diffusion v1.4
        position_ids = new_sd[ANOTHER_POSITION_IDS_KEY]
        del new_sd[ANOTHER_POSITION_IDS_KEY]
    else:
        position_ids = torch.Tensor([list(range(max_length))]).to(torch.int64)

    new_sd["text_model.embeddings.position_ids"] = position_ids
    return new_sd


# endregion


# region Diffusers->StableDiffusion の変換コード
# convert_diffusers_to_original_stable_diffusion をコピーして修正している（ASL 2.0）


def conv_transformer_to_linear(checkpoint):
    keys = list(checkpoint.keys())
    tf_keys = ["proj_in.weight", "proj_out.weight"]
    for key in keys:
        if ".".join(key.split(".")[-2:]) in tf_keys:
            if checkpoint[key].ndim > 2:
                checkpoint[key] = checkpoint[key][:, :, 0, 0]


def convert_unet_state_dict_to_sd(v2, unet_state_dict):
    unet_conversion_map = [
        # (stable-diffusion, HF Diffusers)
        ("time_embed.0.weight", "time_embedding.linear_1.weight"),
        ("time_embed.0.bias", "time_embedding.linear_1.bias"),
        ("time_embed.2.weight", "time_embedding.linear_2.weight"),
        ("time_embed.2.bias", "time_embedding.linear_2.bias"),
        ("input_blocks.0.0.weight", "conv_in.weight"),
        ("input_blocks.0.0.bias", "conv_in.bias"),
        ("out.0.weight", "conv_norm_out.weight"),
        ("out.0.bias", "conv_norm_out.bias"),
        ("out.2.weight", "conv_out.weight"),
        ("out.2.bias", "conv_out.bias"),
    ]

    unet_conversion_map_resnet = [
        # (stable-diffusion, HF Diffusers)
        ("in_layers.0", "norm1"),
        ("in_layers.2", "conv1"),
        ("out_layers.0", "norm2"),
        ("out_layers.3", "conv2"),
        ("emb_layers.1", "time_emb_proj"),
        ("skip_connection", "conv_shortcut"),
    ]

    unet_conversion_map_layer = []
    for i in range(4):
        # loop over downblocks/upblocks

        for j in range(2):
            # loop over resnets/attentions for downblocks
            hf_down_res_prefix = f"down_blocks.{i}.resnets.{j}."
            sd_down_res_prefix = f"input_blocks.{3*i + j + 1}.0."
            unet_conversion_map_layer.append((sd_down_res_prefix, hf_down_res_prefix))

            if i < 3:
                # no attention layers in down_blocks.3
                hf_down_atn_prefix = f"down_blocks.{i}.attentions.{j}."
                sd_down_atn_prefix = f"input_blocks.{3*i + j + 1}.1."
                unet_conversion_map_layer.append((sd_down_atn_prefix, hf_down_atn_prefix))

        for j in range(3):
            # loop over resnets/attentions for upblocks
            hf_up_res_prefix = f"up_blocks.{i}.resnets.{j}."
            sd_up_res_prefix = f"output_blocks.{3*i + j}.0."
            unet_conversion_map_layer.append((sd_up_res_prefix, hf_up_res_prefix))

            if i > 0:
                # no attention layers in up_blocks.0
                hf_up_atn_prefix = f"up_blocks.{i}.attentions.{j}."
                sd_up_atn_prefix = f"output_blocks.{3*i + j}.1."
                unet_conversion_map_layer.append((sd_up_atn_prefix, hf_up_atn_prefix))

        if i < 3:
            # no downsample in down_blocks.3
            hf_downsample_prefix = f"down_blocks.{i}.downsamplers.0.conv."
            sd_downsample_prefix = f"input_blocks.{3*(i+1)}.0.op."
            unet_conversion_map_layer.append((sd_downsample_prefix, hf_downsample_prefix))

            # no upsample in up_blocks.3
            hf_upsample_prefix = f"up_blocks.{i}.upsamplers.0."
            sd_upsample_prefix = f"output_blocks.{3*i + 2}.{1 if i == 0 else 2}."
            unet_conversion_map_layer.append((sd_upsample_prefix, hf_upsample_prefix))

    hf_mid_atn_prefix = "mid_block.attentions.0."
    sd_mid_atn_prefix = "middle_block.1."
    unet_conversion_map_layer.append((sd_mid_atn_prefix, hf_mid_atn_prefix))

    for j in range(2):
        hf_mid_res_prefix = f"mid_block.resnets.{j}."
        sd_mid_res_prefix = f"middle_block.{2*j}."
        unet_conversion_map_layer.append((sd_mid_res_prefix, hf_mid_res_prefix))

    # buyer beware: this is a *brittle* function,
    # and correct output requires that all of these pieces interact in
    # the exact order in which I have arranged them.
    mapping = {k: k for k in unet_state_dict.keys()}
    for sd_name, hf_name in unet_conversion_map:
        mapping[hf_name] = sd_name
    for k, v in mapping.items():
        if "resnets" in k:
            for sd_part, hf_part in unet_conversion_map_resnet:
                v = v.replace(hf_part, sd_part)
            mapping[k] = v
    for k, v in mapping.items():
        for sd_part, hf_part in unet_conversion_map_layer:
            v = v.replace(hf_part, sd_part)
        mapping[k] = v
    new_state_dict = {v: unet_state_dict[k] for k, v in mapping.items()}

    if v2:
        conv_transformer_to_linear(new_state_dict)

    return new_state_dict


def controlnet_conversion_map():
    unet_conversion_map = [
        ("time_embed.0.weight", "time_embedding.linear_1.weight"),
        ("time_embed.0.bias", "time_embedding.linear_1.bias"),
        ("time_embed.2.weight", "time_embedding.linear_2.weight"),
        ("time_embed.2.bias", "time_embedding.linear_2.bias"),
        ("input_blocks.0.0.weight", "conv_in.weight"),
        ("input_blocks.0.0.bias", "conv_in.bias"),
        ("middle_block_out.0.weight", "controlnet_mid_block.weight"),
        ("middle_block_out.0.bias", "controlnet_mid_block.bias"),
    ]

    unet_conversion_map_resnet = [
        ("in_layers.0", "norm1"),
        ("in_layers.2", "conv1"),
        ("out_layers.0", "norm2"),
        ("out_layers.3", "conv2"),
        ("emb_layers.1", "time_emb_proj"),
        ("skip_connection", "conv_shortcut"),
    ]

    unet_conversion_map_layer = []
    for i in range(4):
        for j in range(2):
            hf_down_res_prefix = f"down_blocks.{i}.resnets.{j}."
            sd_down_res_prefix = f"input_blocks.{3*i + j + 1}.0."
            unet_conversion_map_layer.append((sd_down_res_prefix, hf_down_res_prefix))

            if i < 3:
                hf_down_atn_prefix = f"down_blocks.{i}.attentions.{j}."
                sd_down_atn_prefix = f"input_blocks.{3*i + j + 1}.1."
                unet_conversion_map_layer.append((sd_down_atn_prefix, hf_down_atn_prefix))

        if i < 3:
            hf_downsample_prefix = f"down_blocks.{i}.downsamplers.0.conv."
            sd_downsample_prefix = f"input_blocks.{3*(i+1)}.0.op."
            unet_conversion_map_layer.append((sd_downsample_prefix, hf_downsample_prefix))

    hf_mid_atn_prefix = "mid_block.attentions.0."
    sd_mid_atn_prefix = "middle_block.1."
    unet_conversion_map_layer.append((sd_mid_atn_prefix, hf_mid_atn_prefix))

    for j in range(2):
        hf_mid_res_prefix = f"mid_block.resnets.{j}."
        sd_mid_res_prefix = f"middle_block.{2*j}."
        unet_conversion_map_layer.append((sd_mid_res_prefix, hf_mid_res_prefix))

    controlnet_cond_embedding_names = ["conv_in"] + [f"blocks.{i}" for i in range(6)] + ["conv_out"]
    for i, hf_prefix in enumerate(controlnet_cond_embedding_names):
        hf_prefix = f"controlnet_cond_embedding.{hf_prefix}."
        sd_prefix = f"input_hint_block.{i*2}."
        unet_conversion_map_layer.append((sd_prefix, hf_prefix))

    for i in range(12):
        hf_prefix = f"controlnet_down_blocks.{i}."
        sd_prefix = f"zero_convs.{i}.0."
        unet_conversion_map_layer.append((sd_prefix, hf_prefix))

    return unet_conversion_map, unet_conversion_map_resnet, unet_conversion_map_layer


def convert_controlnet_state_dict_to_sd(controlnet_state_dict):
    unet_conversion_map, unet_conversion_map_resnet, unet_conversion_map_layer = controlnet_conversion_map()

    mapping = {k: k for k in controlnet_state_dict.keys()}
    for sd_name, diffusers_name in unet_conversion_map:
        mapping[diffusers_name] = sd_name
    for k, v in mapping.items():
        if "resnets" in k:
            for sd_part, diffusers_part in unet_conversion_map_resnet:
                v = v.replace(diffusers_part, sd_part)
            mapping[k] = v
    for k, v in mapping.items():
        for sd_part, diffusers_part in unet_conversion_map_layer:
            v = v.replace(diffusers_part, sd_part)
        mapping[k] = v
    new_state_dict = {v: controlnet_state_dict[k] for k, v in mapping.items()}
    return new_state_dict


def convert_controlnet_state_dict_to_diffusers(controlnet_state_dict):
    unet_conversion_map, unet_conversion_map_resnet, unet_conversion_map_layer = controlnet_conversion_map()

    mapping = {k: k for k in controlnet_state_dict.keys()}
    for sd_name, diffusers_name in unet_conversion_map:
        mapping[sd_name] = diffusers_name
    for k, v in mapping.items():
        for sd_part, diffusers_part in unet_conversion_map_layer:
            v = v.replace(sd_part, diffusers_part)
        mapping[k] = v
    for k, v in mapping.items():
        if "resnets" in v:
            for sd_part, diffusers_part in unet_conversion_map_resnet:
                v = v.replace(sd_part, diffusers_part)
            mapping[k] = v
    new_state_dict = {v: controlnet_state_dict[k] for k, v in mapping.items()}
    return new_state_dict


# ================#
# VAE Conversion #
# ================#


def reshape_weight_for_sd(w):
    # convert HF linear weights to SD conv2d weights
    return w.reshape(*w.shape, 1, 1)


def convert_vae_state_dict(vae_state_dict):
    vae_conversion_map = [
        # (stable-diffusion, HF Diffusers)
        ("nin_shortcut", "conv_shortcut"),
        ("norm_out", "conv_norm_out"),
        ("mid.attn_1.", "mid_block.attentions.0."),
    ]

    for i in range(4):
        # down_blocks have two resnets
        for j in range(2):
            hf_down_prefix = f"encoder.down_blocks.{i}.resnets.{j}."
            sd_down_prefix = f"encoder.down.{i}.block.{j}."
            vae_conversion_map.append((sd_down_prefix, hf_down_prefix))

        if i < 3:
            hf_downsample_prefix = f"down_blocks.{i}.downsamplers.0."
            sd_downsample_prefix = f"down.{i}.downsample."
            vae_conversion_map.append((sd_downsample_prefix, hf_downsample_prefix))

            hf_upsample_prefix = f"up_blocks.{i}.upsamplers.0."
            sd_upsample_prefix = f"up.{3-i}.upsample."
            vae_conversion_map.append((sd_upsample_prefix, hf_upsample_prefix))

        # up_blocks have three resnets
        # also, up blocks in hf are numbered in reverse from sd
        for j in range(3):
            hf_up_prefix = f"decoder.up_blocks.{i}.resnets.{j}."
            sd_up_prefix = f"decoder.up.{3-i}.block.{j}."
            vae_conversion_map.append((sd_up_prefix, hf_up_prefix))

    # this part accounts for mid blocks in both the encoder and the decoder
    for i in range(2):
        hf_mid_res_prefix = f"mid_block.resnets.{i}."
        sd_mid_res_prefix = f"mid.block_{i+1}."
        vae_conversion_map.append((sd_mid_res_prefix, hf_mid_res_prefix))

    if diffusers.__version__ < "0.17.0":
        vae_conversion_map_attn = [
            # (stable-diffusion, HF Diffusers)
            ("norm.", "group_norm."),
            ("q.", "query."),
            ("k.", "key."),
            ("v.", "value."),
            ("proj_out.", "proj_attn."),
        ]
    else:
        vae_conversion_map_attn = [
            # (stable-diffusion, HF Diffusers)
            ("norm.", "group_norm."),
            ("q.", "to_q."),
            ("k.", "to_k."),
            ("v.", "to_v."),
            ("proj_out.", "to_out.0."),
        ]

    mapping = {k: k for k in vae_state_dict.keys()}
    for k, v in mapping.items():
        for sd_part, hf_part in vae_conversion_map:
            v = v.replace(hf_part, sd_part)
        mapping[k] = v
    for k, v in mapping.items():
        if "attentions" in k:
            for sd_part, hf_part in vae_conversion_map_attn:
                v = v.replace(hf_part, sd_part)
            mapping[k] = v
    new_state_dict = {v: vae_state_dict[k] for k, v in mapping.items()}
    weights_to_convert = ["q", "k", "v", "proj_out"]
    for k, v in new_state_dict.items():
        for weight_name in weights_to_convert:
            if f"mid.attn_1.{weight_name}.weight" in k:
                # logger.info(f"Reshaping {k} for SD format: shape {v.shape} -> {v.shape} x 1 x 1")
                new_state_dict[k] = reshape_weight_for_sd(v)

    return new_state_dict


# endregion

# region 自作のモデル読み書きなど


def is_safetensors(path):
    return os.path.splitext(path)[1].lower() == ".safetensors"


def load_checkpoint_with_text_encoder_conversion(ckpt_path, device="cpu"):
    # text encoderの格納形式が違うモデルに対応する ('text_model'がない)
    TEXT_ENCODER_KEY_REPLACEMENTS = [
        ("cond_stage_model.transformer.embeddings.", "cond_stage_model.transformer.text_model.embeddings."),
        ("cond_stage_model.transformer.encoder.", "cond_stage_model.transformer.text_model.encoder."),
        ("cond_stage_model.transformer.final_layer_norm.", "cond_stage_model.transformer.text_model.final_layer_norm."),
    ]

    if is_safetensors(ckpt_path):
        checkpoint = None
        state_dict = load_file(ckpt_path)  # , device) # may causes error
    else:
        checkpoint = torch.load(ckpt_path, map_location=device)
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
            checkpoint = None

    key_reps = []
    for rep_from, rep_to in TEXT_ENCODER_KEY_REPLACEMENTS:
        for key in state_dict.keys():
            if key.startswith(rep_from):
                new_key = rep_to + key[len(rep_from) :]
                key_reps.append((key, new_key))

    for key, new_key in key_reps:
        state_dict[new_key] = state_dict[key]
        del state_dict[key]

    return checkpoint, state_dict


# TODO dtype指定の動作が怪しいので確認する text_encoderを指定形式で作れるか未確認
def load_models_from_stable_diffusion_checkpoint(v2, ckpt_path, device="cpu", dtype=None, unet_use_linear_projection_in_v2=True):
    _, state_dict = load_checkpoint_with_text_encoder_conversion(ckpt_path, device)

    # Convert the UNet2DConditionModel model.
    unet_config = create_unet_diffusers_config(v2, unet_use_linear_projection_in_v2)
    converted_unet_checkpoint = convert_ldm_unet_checkpoint(v2, state_dict, unet_config)

    unet = UNet2DConditionModel(**unet_config).to(device)
    info = unet.load_state_dict(converted_unet_checkpoint)
    logger.info(f"loading u-net: {info}")

    # Convert the VAE model.
    vae_config = create_vae_diffusers_config()
    converted_vae_checkpoint = convert_ldm_vae_checkpoint(state_dict, vae_config)

    vae = AutoencoderKL(**vae_config).to(device)
    info = vae.load_state_dict(converted_vae_checkpoint)
    logger.info(f"loading vae: {info}")

    # convert text_model
    if v2:
        converted_text_encoder_checkpoint = convert_ldm_clip_checkpoint_v2(state_dict, 77)
        cfg = CLIPTextConfig(
            vocab_size=49408,
            hidden_size=1024,
            intermediate_size=4096,
            num_hidden_layers=23,
            num_attention_heads=16,
            max_position_embeddings=77,
            hidden_act="gelu",
            layer_norm_eps=1e-05,
            dropout=0.0,
            attention_dropout=0.0,
            initializer_range=0.02,
            initializer_factor=1.0,
            pad_token_id=1,
            bos_token_id=0,
            eos_token_id=2,
            model_type="clip_text_model",
            projection_dim=512,
            torch_dtype="float32",
            transformers_version="4.25.0.dev0",
        )
        text_model = CLIPTextModel._from_config(cfg)
        info = text_model.load_state_dict(converted_text_encoder_checkpoint)
    else:
        converted_text_encoder_checkpoint = convert_ldm_clip_checkpoint_v1(state_dict)

        # logging.set_verbosity_error()  # don't show annoying warning
        # text_model = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
        # logging.set_verbosity_warning()
        # logger.info(f"config: {text_model.config}")
        cfg = CLIPTextConfig(
            vocab_size=49408,
            hidden_size=768,
            intermediate_size=3072,
            num_hidden_layers=12,
            num_attention_heads=12,
            max_position_embeddings=77,
            hidden_act="quick_gelu",
            layer_norm_eps=1e-05,
            dropout=0.0,
            attention_dropout=0.0,
            initializer_range=0.02,
            initializer_factor=1.0,
            pad_token_id=1,
            bos_token_id=0,
            eos_token_id=2,
            model_type="clip_text_model",
            projection_dim=768,
            torch_dtype="float32",
        )
        text_model = CLIPTextModel._from_config(cfg)
        info = text_model.load_state_dict(converted_text_encoder_checkpoint)
    logger.info(f"loading text encoder: {info}")

    return text_model, vae, unet


def get_model_version_str_for_sd1_sd2(v2, v_parameterization):
    # only for reference
    version_str = "sd"
    if v2:
        version_str += "_v2"
    else:
        version_str += "_v1"
    if v_parameterization:
        version_str += "_v"
    return version_str


def convert_text_encoder_state_dict_to_sd_v2(checkpoint, make_dummy_weights=False):
    def convert_key(key):
        # position_idsの除去
        if ".position_ids" in key:
            return None

        # common
        key = key.replace("text_model.encoder.", "transformer.")
        key = key.replace("text_model.", "")
        if "layers" in key:
            # resblocks conversion
            key = key.replace(".layers.", ".resblocks.")
            if ".layer_norm" in key:
                key = key.replace(".layer_norm", ".ln_")
            elif ".mlp." in key:
                key = key.replace(".fc1.", ".c_fc.")
                key = key.replace(".fc2.", ".c_proj.")
            elif ".self_attn.out_proj" in key:
                key = key.replace(".self_attn.out_proj.", ".attn.out_proj.")
            elif ".self_attn." in key:
                key = None  # 特殊なので後で処理する
            else:
                raise ValueError(f"unexpected key in DiffUsers model: {key}")
        elif ".position_embedding" in key:
            key = key.replace("embeddings.position_embedding.weight", "positional_embedding")
        elif ".token_embedding" in key:
            key = key.replace("embeddings.token_embedding.weight", "token_embedding.weight")
        elif "final_layer_norm" in key:
            key = key.replace("final_layer_norm", "ln_final")
        return key

    keys = list(checkpoint.keys())
    new_sd = {}
    for key in keys:
        new_key = convert_key(key)
        if new_key is None:
            continue
        new_sd[new_key] = checkpoint[key]

    # attnの変換
    for key in keys:
        if "layers" in key and "q_proj" in key:
            # 三つを結合
            key_q = key
            key_k = key.replace("q_proj", "k_proj")
            key_v = key.replace("q_proj", "v_proj")

            value_q = checkpoint[key_q]
            value_k = checkpoint[key_k]
            value_v = checkpoint[key_v]
            value = torch.cat([value_q, value_k, value_v])

            new_key = key.replace("text_model.encoder.layers.", "transformer.resblocks.")
            new_key = new_key.replace(".self_attn.q_proj.", ".attn.in_proj_")
            new_sd[new_key] = value

    # 最後の層などを捏造するか
    if make_dummy_weights:
        logger.info("make dummy weights for resblock.23, text_projection and logit scale.")
        keys = list(new_sd.keys())
        for key in keys:
            if key.startswith("transformer.resblocks.22."):
                new_sd[key.replace(".22.", ".23.")] = new_sd[key].clone()  # copyしないとsafetensorsの保存で落ちる

        # Diffusersに含まれない重みを作っておく
        new_sd["text_projection"] = torch.ones((1024, 1024), dtype=new_sd[keys[0]].dtype, device=new_sd[keys[0]].device)
        new_sd["logit_scale"] = torch.tensor(1)

    return new_sd


def save_stable_diffusion_checkpoint(
    v2, output_file, text_encoder, unet, ckpt_path, epochs, steps, metadata, save_dtype=None, vae=None
):
    if ckpt_path is not None:
        # epoch/stepを参照する。またVAEがメモリ上にないときなど、もう一度VAEを含めて読み込む
        checkpoint, state_dict = load_checkpoint_with_text_encoder_conversion(ckpt_path)
        if checkpoint is None:  # safetensors または state_dictのckpt
            checkpoint = {}
            strict = False
        else:
            strict = True
        if "state_dict" in state_dict:
            del state_dict["state_dict"]
    else:
        # 新しく作る
        assert vae is not None, "VAE is required to save a checkpoint without a given checkpoint"
        checkpoint = {}
        state_dict = {}
        strict = False

    def update_sd(prefix, sd):
        for k, v in sd.items():
            key = prefix + k
            assert not strict or key in state_dict, f"Illegal key in save SD: {key}"
            if save_dtype is not None:
                v = v.detach().clone().to("cpu").to(save_dtype)
            state_dict[key] = v

    # Convert the UNet model
    unet_state_dict = convert_unet_state_dict_to_sd(v2, unet.state_dict())
    update_sd("model.diffusion_model.", unet_state_dict)

    # Convert the text encoder model
    if v2:
        make_dummy = ckpt_path is None  # 参照元のcheckpointがない場合は最後の層を前の層から複製して作るなどダミーの重みを入れる
        text_enc_dict = convert_text_encoder_state_dict_to_sd_v2(text_encoder.state_dict(), make_dummy)
        update_sd("cond_stage_model.model.", text_enc_dict)
    else:
        text_enc_dict = text_encoder.state_dict()
        update_sd("cond_stage_model.transformer.", text_enc_dict)

    # Convert the VAE
    if vae is not None:
        vae_dict = convert_vae_state_dict(vae.state_dict())
        update_sd("first_stage_model.", vae_dict)

    # Put together new checkpoint
    key_count = len(state_dict.keys())
    new_ckpt = {"state_dict": state_dict}

    # epoch and global_step are sometimes not int
    try:
        if "epoch" in checkpoint:
            epochs += checkpoint["epoch"]
        if "global_step" in checkpoint:
            steps += checkpoint["global_step"]
    except:
        pass

    new_ckpt["epoch"] = epochs
    new_ckpt["global_step"] = steps

    if is_safetensors(output_file):
        # TODO Tensor以外のdictの値を削除したほうがいいか
        save_file(state_dict, output_file, metadata)
    else:
        torch.save(new_ckpt, output_file)

    return key_count


def save_diffusers_checkpoint(v2, output_dir, text_encoder, unet, pretrained_model_name_or_path, vae=None, use_safetensors=False):
    if pretrained_model_name_or_path is None:
        # load default settings for v1/v2
        if v2:
            pretrained_model_name_or_path = DIFFUSERS_REF_MODEL_ID_V2
        else:
            pretrained_model_name_or_path = DIFFUSERS_REF_MODEL_ID_V1

    scheduler = DDIMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer")
    if vae is None:
        vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae")

    # original U-Net cannot be saved, so we need to convert it to the Diffusers version
    # TODO this consumes a lot of memory
    diffusers_unet = diffusers.UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path, subfolder="unet")
    diffusers_unet.load_state_dict(unet.state_dict())

    pipeline = StableDiffusionPipeline(
        unet=diffusers_unet,
        text_encoder=text_encoder,
        vae=vae,
        scheduler=scheduler,
        tokenizer=tokenizer,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=None,
    )
    pipeline.save_pretrained(output_dir, safe_serialization=use_safetensors)


VAE_PREFIX = "first_stage_model."


def load_vae(vae_id, dtype):
    logger.info(f"load VAE: {vae_id}")
    if os.path.isdir(vae_id) or not os.path.isfile(vae_id):
        # Diffusers local/remote
        try:
            vae = AutoencoderKL.from_pretrained(vae_id, subfolder=None, torch_dtype=dtype)
        except EnvironmentError as e:
            logger.error(f"exception occurs in loading vae: {e}")
            logger.error("retry with subfolder='vae'")
            vae = AutoencoderKL.from_pretrained(vae_id, subfolder="vae", torch_dtype=dtype)
        return vae

    # local
    vae_config = create_vae_diffusers_config()

    if vae_id.endswith(".bin"):
        # SD 1.5 VAE on Huggingface
        converted_vae_checkpoint = torch.load(vae_id, map_location="cpu")
    else:
        # StableDiffusion
        vae_model = load_file(vae_id, "cpu") if is_safetensors(vae_id) else torch.load(vae_id, map_location="cpu")
        vae_sd = vae_model["state_dict"] if "state_dict" in vae_model else vae_model

        # vae only or full model
        full_model = False
        for vae_key in vae_sd:
            if vae_key.startswith(VAE_PREFIX):
                full_model = True
                break
        if not full_model:
            sd = {}
            for key, value in vae_sd.items():
                sd[VAE_PREFIX + key] = value
            vae_sd = sd
            del sd

        # Convert the VAE model.
        converted_vae_checkpoint = convert_ldm_vae_checkpoint(vae_sd, vae_config)

    vae = AutoencoderKL(**vae_config)
    vae.load_state_dict(converted_vae_checkpoint)
    return vae


# endregion


def make_bucket_resolutions(max_reso, min_size=256, max_size=1024, divisible=64):
    max_width, max_height = max_reso
    max_area = max_width * max_height

    resos = set()

    width = int(math.sqrt(max_area) // divisible) * divisible
    resos.add((width, width))

    width = min_size
    while width <= max_size:
        height = min(max_size, int((max_area // width) // divisible) * divisible)
        if height >= min_size:
            resos.add((width, height))
            resos.add((height, width))

        # # make additional resos
        # if width >= height and width - divisible >= min_size:
        #   resos.add((width - divisible, height))
        #   resos.add((height, width - divisible))
        # if height >= width and height - divisible >= min_size:
        #   resos.add((width, height - divisible))
        #   resos.add((height - divisible, width))

        width += divisible

    resos = list(resos)
    resos.sort()
    return resos


if __name__ == "__main__":
    resos = make_bucket_resolutions((512, 768))
    logger.info(f"{len(resos)}")
    logger.info(f"{resos}")
    aspect_ratios = [w / h for w, h in resos]
    logger.info(f"{aspect_ratios}")

    ars = set()
    for ar in aspect_ratios:
        if ar in ars:
            logger.error(f"error! duplicate ar: {ar}")
        ars.add(ar)





# Diffusersのコードをベースとした sd_xl_baseのU-Net
# state dictの形式をSDXLに合わせてある

"""
      target: sgm.modules.diffusionmodules.openaimodel.UNetModel
      params:
        adm_in_channels: 2816
        num_classes: sequential
        use_checkpoint: True
        in_channels: 4
        out_channels: 4
        model_channels: 320
        attention_resolutions: [4, 2]
        num_res_blocks: 2
        channel_mult: [1, 2, 4]
        num_head_channels: 64
        use_spatial_transformer: True
        use_linear_in_transformer: True
        transformer_depth: [1, 2, 10]  # note: the first is unused (due to attn_res starting at 2) 32, 16, 8 --> 64, 32, 16
        context_dim: 2048
        spatial_transformer_attn_type: softmax-xformers
        legacy: False
"""

import math
from types import SimpleNamespace
from typing import Any, Optional
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import functional as F
from einops import rearrange
import logging

logger = logging.getLogger(__name__)

IN_CHANNELS: int = 4
OUT_CHANNELS: int = 4
ADM_IN_CHANNELS: int = 2816
CONTEXT_DIM: int = 2048
MODEL_CHANNELS: int = 320
TIME_EMBED_DIM = 320 * 4

USE_REENTRANT = True

# region memory efficient attention

# FlashAttentionを使うCrossAttention
# based on https://github.com/lucidrains/memory-efficient-attention-pytorch/blob/main/memory_efficient_attention_pytorch/flash_attention.py
# LICENSE MIT https://github.com/lucidrains/memory-efficient-attention-pytorch/blob/main/LICENSE

# constants

EPSILON = 1e-6

# helper functions


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


# flash attention forwards and backwards

# https://arxiv.org/abs/2205.14135


class FlashAttentionFunction(torch.autograd.Function):
    @staticmethod
    @torch.no_grad()
    def forward(ctx, q, k, v, mask, causal, q_bucket_size, k_bucket_size):
        """Algorithm 2 in the paper"""

        device = q.device
        dtype = q.dtype
        max_neg_value = -torch.finfo(q.dtype).max
        qk_len_diff = max(k.shape[-2] - q.shape[-2], 0)

        o = torch.zeros_like(q)
        all_row_sums = torch.zeros((*q.shape[:-1], 1), dtype=dtype, device=device)
        all_row_maxes = torch.full((*q.shape[:-1], 1), max_neg_value, dtype=dtype, device=device)

        scale = q.shape[-1] ** -0.5

        if not exists(mask):
            mask = (None,) * math.ceil(q.shape[-2] / q_bucket_size)
        else:
            mask = rearrange(mask, "b n -> b 1 1 n")
            mask = mask.split(q_bucket_size, dim=-1)

        row_splits = zip(
            q.split(q_bucket_size, dim=-2),
            o.split(q_bucket_size, dim=-2),
            mask,
            all_row_sums.split(q_bucket_size, dim=-2),
            all_row_maxes.split(q_bucket_size, dim=-2),
        )

        for ind, (qc, oc, row_mask, row_sums, row_maxes) in enumerate(row_splits):
            q_start_index = ind * q_bucket_size - qk_len_diff

            col_splits = zip(
                k.split(k_bucket_size, dim=-2),
                v.split(k_bucket_size, dim=-2),
            )

            for k_ind, (kc, vc) in enumerate(col_splits):
                k_start_index = k_ind * k_bucket_size

                attn_weights = torch.einsum("... i d, ... j d -> ... i j", qc, kc) * scale

                if exists(row_mask):
                    attn_weights.masked_fill_(~row_mask, max_neg_value)

                if causal and q_start_index < (k_start_index + k_bucket_size - 1):
                    causal_mask = torch.ones((qc.shape[-2], kc.shape[-2]), dtype=torch.bool, device=device).triu(
                        q_start_index - k_start_index + 1
                    )
                    attn_weights.masked_fill_(causal_mask, max_neg_value)

                block_row_maxes = attn_weights.amax(dim=-1, keepdims=True)
                attn_weights -= block_row_maxes
                exp_weights = torch.exp(attn_weights)

                if exists(row_mask):
                    exp_weights.masked_fill_(~row_mask, 0.0)

                block_row_sums = exp_weights.sum(dim=-1, keepdims=True).clamp(min=EPSILON)

                new_row_maxes = torch.maximum(block_row_maxes, row_maxes)

                exp_values = torch.einsum("... i j, ... j d -> ... i d", exp_weights, vc)

                exp_row_max_diff = torch.exp(row_maxes - new_row_maxes)
                exp_block_row_max_diff = torch.exp(block_row_maxes - new_row_maxes)

                new_row_sums = exp_row_max_diff * row_sums + exp_block_row_max_diff * block_row_sums

                oc.mul_((row_sums / new_row_sums) * exp_row_max_diff).add_((exp_block_row_max_diff / new_row_sums) * exp_values)

                row_maxes.copy_(new_row_maxes)
                row_sums.copy_(new_row_sums)

        ctx.args = (causal, scale, mask, q_bucket_size, k_bucket_size)
        ctx.save_for_backward(q, k, v, o, all_row_sums, all_row_maxes)

        return o

    @staticmethod
    @torch.no_grad()
    def backward(ctx, do):
        """Algorithm 4 in the paper"""

        causal, scale, mask, q_bucket_size, k_bucket_size = ctx.args
        q, k, v, o, l, m = ctx.saved_tensors

        device = q.device

        max_neg_value = -torch.finfo(q.dtype).max
        qk_len_diff = max(k.shape[-2] - q.shape[-2], 0)

        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)

        row_splits = zip(
            q.split(q_bucket_size, dim=-2),
            o.split(q_bucket_size, dim=-2),
            do.split(q_bucket_size, dim=-2),
            mask,
            l.split(q_bucket_size, dim=-2),
            m.split(q_bucket_size, dim=-2),
            dq.split(q_bucket_size, dim=-2),
        )

        for ind, (qc, oc, doc, row_mask, lc, mc, dqc) in enumerate(row_splits):
            q_start_index = ind * q_bucket_size - qk_len_diff

            col_splits = zip(
                k.split(k_bucket_size, dim=-2),
                v.split(k_bucket_size, dim=-2),
                dk.split(k_bucket_size, dim=-2),
                dv.split(k_bucket_size, dim=-2),
            )

            for k_ind, (kc, vc, dkc, dvc) in enumerate(col_splits):
                k_start_index = k_ind * k_bucket_size

                attn_weights = torch.einsum("... i d, ... j d -> ... i j", qc, kc) * scale

                if causal and q_start_index < (k_start_index + k_bucket_size - 1):
                    causal_mask = torch.ones((qc.shape[-2], kc.shape[-2]), dtype=torch.bool, device=device).triu(
                        q_start_index - k_start_index + 1
                    )
                    attn_weights.masked_fill_(causal_mask, max_neg_value)

                exp_attn_weights = torch.exp(attn_weights - mc)

                if exists(row_mask):
                    exp_attn_weights.masked_fill_(~row_mask, 0.0)

                p = exp_attn_weights / lc

                dv_chunk = torch.einsum("... i j, ... i d -> ... j d", p, doc)
                dp = torch.einsum("... i d, ... j d -> ... i j", doc, vc)

                D = (doc * oc).sum(dim=-1, keepdims=True)
                ds = p * scale * (dp - D)

                dq_chunk = torch.einsum("... i j, ... j d -> ... i d", ds, kc)
                dk_chunk = torch.einsum("... i j, ... i d -> ... j d", ds, qc)

                dqc.add_(dq_chunk)
                dkc.add_(dk_chunk)
                dvc.add_(dv_chunk)

        return dq, dk, dv, None, None, None, None


# endregion


def get_parameter_dtype(parameter: torch.nn.Module):
    return next(parameter.parameters()).dtype


def get_parameter_device(parameter: torch.nn.Module):
    return next(parameter.parameters()).device


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models: Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param embedding_dim: the dimension of the output. :param max_period: controls the minimum frequency of the
    embeddings. :return: an [N x dim] Tensor of positional embeddings.
    """
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(start=0, end=half_dim, dtype=torch.float32, device=timesteps.device)
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]

    # scale embeddings
    emb = scale * emb

    # concat sine and cosine embeddings: flipped from Diffusers original ver because always flip_sin_to_cos=True
    emb = torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1)

    # zero pad
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


# Deep Shrink: We do not common this function, because minimize dependencies.
def resize_like(x, target, mode="bicubic", align_corners=False):
    org_dtype = x.dtype
    if org_dtype == torch.bfloat16:
        x = x.to(torch.float32)

    if x.shape[-2:] != target.shape[-2:]:
        if mode == "nearest":
            x = F.interpolate(x, size=target.shape[-2:], mode=mode)
        else:
            x = F.interpolate(x, size=target.shape[-2:], mode=mode, align_corners=align_corners)

    if org_dtype == torch.bfloat16:
        x = x.to(org_dtype)
    return x


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        if self.weight.dtype != torch.float32:
            return super().forward(x)
        return super().forward(x.float()).type(x.dtype)


class ResnetBlock2D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.in_layers = nn.Sequential(
            GroupNorm32(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        )

        self.emb_layers = nn.Sequential(nn.SiLU(), nn.Linear(TIME_EMBED_DIM, out_channels))

        self.out_layers = nn.Sequential(
            GroupNorm32(32, out_channels),
            nn.SiLU(),
            nn.Identity(),  # to make state_dict compatible with original model
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        )

        if in_channels != out_channels:
            self.skip_connection = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.skip_connection = nn.Identity()

        self.gradient_checkpointing = False

    def forward_body(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        h = h + emb_out[:, :, None, None]
        h = self.out_layers(h)
        x = self.skip_connection(x)
        return x + h

    def forward(self, x, emb):
        if self.training and self.gradient_checkpointing:
            # logger.info("ResnetBlock2D: gradient_checkpointing")

            def create_custom_forward(func):
                def custom_forward(*inputs):
                    return func(*inputs)

                return custom_forward

            x = torch.utils.checkpoint.checkpoint(create_custom_forward(self.forward_body), x, emb, use_reentrant=USE_REENTRANT)
        else:
            x = self.forward_body(x, emb)

        return x


class Downsample2D(nn.Module):
    def __init__(self, channels, out_channels):
        super().__init__()

        self.channels = channels
        self.out_channels = out_channels

        self.op = nn.Conv2d(self.channels, self.out_channels, 3, stride=2, padding=1)

        self.gradient_checkpointing = False

    def forward_body(self, hidden_states):
        assert hidden_states.shape[1] == self.channels
        hidden_states = self.op(hidden_states)

        return hidden_states

    def forward(self, hidden_states):
        if self.training and self.gradient_checkpointing:
            # logger.info("Downsample2D: gradient_checkpointing")

            def create_custom_forward(func):
                def custom_forward(*inputs):
                    return func(*inputs)

                return custom_forward

            hidden_states = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.forward_body), hidden_states, use_reentrant=USE_REENTRANT
            )
        else:
            hidden_states = self.forward_body(hidden_states)

        return hidden_states


class CrossAttention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        upcast_attention: bool = False,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
        self.upcast_attention = upcast_attention

        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(cross_attention_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(cross_attention_dim, inner_dim, bias=False)

        self.to_out = nn.ModuleList([])
        self.to_out.append(nn.Linear(inner_dim, query_dim))
        # no dropout here

        self.use_memory_efficient_attention_xformers = False
        self.use_memory_efficient_attention_mem_eff = False
        self.use_sdpa = False

    def set_use_memory_efficient_attention(self, xformers, mem_eff):
        self.use_memory_efficient_attention_xformers = xformers
        self.use_memory_efficient_attention_mem_eff = mem_eff

    def set_use_sdpa(self, sdpa):
        self.use_sdpa = sdpa

    def reshape_heads_to_batch_dim(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size * head_size, seq_len, dim // head_size)
        return tensor

    def reshape_batch_dim_to_heads(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // head_size, seq_len, dim * head_size)
        return tensor

    def forward(self, hidden_states, context=None, mask=None):
        if self.use_memory_efficient_attention_xformers:
            return self.forward_memory_efficient_xformers(hidden_states, context, mask)
        if self.use_memory_efficient_attention_mem_eff:
            return self.forward_memory_efficient_mem_eff(hidden_states, context, mask)
        if self.use_sdpa:
            return self.forward_sdpa(hidden_states, context, mask)

        query = self.to_q(hidden_states)
        context = context if context is not None else hidden_states
        key = self.to_k(context)
        value = self.to_v(context)

        query = self.reshape_heads_to_batch_dim(query)
        key = self.reshape_heads_to_batch_dim(key)
        value = self.reshape_heads_to_batch_dim(value)

        hidden_states = self._attention(query, key, value)

        # linear proj
        hidden_states = self.to_out[0](hidden_states)
        # hidden_states = self.to_out[1](hidden_states)     # no dropout
        return hidden_states

    def _attention(self, query, key, value):
        if self.upcast_attention:
            query = query.float()
            key = key.float()

        attention_scores = torch.baddbmm(
            torch.empty(query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device),
            query,
            key.transpose(-1, -2),
            beta=0,
            alpha=self.scale,
        )
        attention_probs = attention_scores.softmax(dim=-1)

        # cast back to the original dtype
        attention_probs = attention_probs.to(value.dtype)

        # compute attention output
        hidden_states = torch.bmm(attention_probs, value)

        # reshape hidden_states
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        return hidden_states

    # TODO support Hypernetworks
    def forward_memory_efficient_xformers(self, x, context=None, mask=None):


        h = self.heads
        q_in = self.to_q(x)
        context = context if context is not None else x
        context = context.to(x.dtype)
        k_in = self.to_k(context)
        v_in = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b n h d", h=h), (q_in, k_in, v_in))
        del q_in, k_in, v_in

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None)  # 最適なのを選んでくれる
        del q, k, v

        out = rearrange(out, "b n h d -> b n (h d)", h=h)

        out = self.to_out[0](out)
        return out

    def forward_memory_efficient_mem_eff(self, x, context=None, mask=None):
        flash_func = FlashAttentionFunction

        q_bucket_size = 512
        k_bucket_size = 1024

        h = self.heads
        q = self.to_q(x)
        context = context if context is not None else x
        context = context.to(x.dtype)
        k = self.to_k(context)
        v = self.to_v(context)
        del context, x

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

        out = flash_func.apply(q, k, v, mask, False, q_bucket_size, k_bucket_size)

        out = rearrange(out, "b h n d -> b n (h d)")

        out = self.to_out[0](out)
        return out

    def forward_sdpa(self, x, context=None, mask=None):
        h = self.heads
        q_in = self.to_q(x)
        context = context if context is not None else x
        context = context.to(x.dtype)
        k_in = self.to_k(context)
        v_in = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q_in, k_in, v_in))
        del q_in, k_in, v_in

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)

        out = rearrange(out, "b h n d -> b n (h d)", h=h)

        out = self.to_out[0](out)
        return out


# feedforward
class GEGLU(nn.Module):
    r"""
    A variant of the gated linear unit activation function from https://arxiv.org/abs/2002.05202.

    Parameters:
        dim_in (`int`): The number of channels in the input.
        dim_out (`int`): The number of channels in the output.
    """

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def gelu(self, gate):
        if gate.device.type != "mps":
            return F.gelu(gate)
        # mps: gelu is not implemented for float16
        return F.gelu(gate.to(dtype=torch.float32)).to(dtype=gate.dtype)

    def forward(self, hidden_states):
        hidden_states, gate = self.proj(hidden_states).chunk(2, dim=-1)
        return hidden_states * self.gelu(gate)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
    ):
        super().__init__()
        inner_dim = int(dim * 4)  # mult is always 4

        self.net = nn.ModuleList([])
        # project in
        self.net.append(GEGLU(dim, inner_dim))
        # project dropout
        self.net.append(nn.Identity())  # nn.Dropout(0)) # dummy for dropout with 0
        # project out
        self.net.append(nn.Linear(inner_dim, dim))

    def forward(self, hidden_states):
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


class BasicTransformerBlock(nn.Module):
    def __init__(
        self, dim: int, num_attention_heads: int, attention_head_dim: int, cross_attention_dim: int, upcast_attention: bool = False
    ):
        super().__init__()

        self.gradient_checkpointing = False

        # 1. Self-Attn
        self.attn1 = CrossAttention(
            query_dim=dim,
            cross_attention_dim=None,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            upcast_attention=upcast_attention,
        )
        self.ff = FeedForward(dim)

        # 2. Cross-Attn
        self.attn2 = CrossAttention(
            query_dim=dim,
            cross_attention_dim=cross_attention_dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            upcast_attention=upcast_attention,
        )

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # 3. Feed-forward
        self.norm3 = nn.LayerNorm(dim)

    def set_use_memory_efficient_attention(self, xformers: bool, mem_eff: bool):
        self.attn1.set_use_memory_efficient_attention(xformers, mem_eff)
        self.attn2.set_use_memory_efficient_attention(xformers, mem_eff)

    def set_use_sdpa(self, sdpa: bool):
        self.attn1.set_use_sdpa(sdpa)
        self.attn2.set_use_sdpa(sdpa)

    def forward_body(self, hidden_states, context=None, timestep=None):
        # 1. Self-Attention
        norm_hidden_states = self.norm1(hidden_states)

        hidden_states = self.attn1(norm_hidden_states) + hidden_states

        # 2. Cross-Attention
        norm_hidden_states = self.norm2(hidden_states)
        hidden_states = self.attn2(norm_hidden_states, context=context) + hidden_states

        # 3. Feed-forward
        hidden_states = self.ff(self.norm3(hidden_states)) + hidden_states

        return hidden_states

    def forward(self, hidden_states, context=None, timestep=None):
        if self.training and self.gradient_checkpointing:
            # logger.info("BasicTransformerBlock: checkpointing")

            def create_custom_forward(func):
                def custom_forward(*inputs):
                    return func(*inputs)

                return custom_forward

            output = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.forward_body), hidden_states, context, timestep, use_reentrant=USE_REENTRANT
            )
        else:
            output = self.forward_body(hidden_states, context, timestep)

        return output


class Transformer2DModel(nn.Module):
    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        cross_attention_dim: Optional[int] = None,
        use_linear_projection: bool = False,
        upcast_attention: bool = False,
        num_transformer_layers: int = 1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim
        self.use_linear_projection = use_linear_projection

        self.norm = torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        # self.norm = GroupNorm32(32, in_channels, eps=1e-6, affine=True)

        if use_linear_projection:
            self.proj_in = nn.Linear(in_channels, inner_dim)
        else:
            self.proj_in = nn.Conv2d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)

        blocks = []
        for _ in range(num_transformer_layers):
            blocks.append(
                BasicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    cross_attention_dim=cross_attention_dim,
                    upcast_attention=upcast_attention,
                )
            )

        self.transformer_blocks = nn.ModuleList(blocks)

        if use_linear_projection:
            self.proj_out = nn.Linear(in_channels, inner_dim)
        else:
            self.proj_out = nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)

        self.gradient_checkpointing = False

    def set_use_memory_efficient_attention(self, xformers, mem_eff):
        for transformer in self.transformer_blocks:
            transformer.set_use_memory_efficient_attention(xformers, mem_eff)

    def set_use_sdpa(self, sdpa):
        for transformer in self.transformer_blocks:
            transformer.set_use_sdpa(sdpa)

    def forward(self, hidden_states, encoder_hidden_states=None, timestep=None):
        # 1. Input
        batch, _, height, weight = hidden_states.shape
        residual = hidden_states

        hidden_states = self.norm(hidden_states)
        if not self.use_linear_projection:
            hidden_states = self.proj_in(hidden_states)
            inner_dim = hidden_states.shape[1]
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * weight, inner_dim)
        else:
            inner_dim = hidden_states.shape[1]
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * weight, inner_dim)
            hidden_states = self.proj_in(hidden_states)

        # 2. Blocks
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, context=encoder_hidden_states, timestep=timestep)

        # 3. Output
        if not self.use_linear_projection:
            hidden_states = hidden_states.reshape(batch, height, weight, inner_dim).permute(0, 3, 1, 2).contiguous()
            hidden_states = self.proj_out(hidden_states)
        else:
            hidden_states = self.proj_out(hidden_states)
            hidden_states = hidden_states.reshape(batch, height, weight, inner_dim).permute(0, 3, 1, 2).contiguous()

        output = hidden_states + residual

        return output


class Upsample2D(nn.Module):
    def __init__(self, channels, out_channels):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(self.channels, self.out_channels, 3, padding=1)

        self.gradient_checkpointing = False

    def forward_body(self, hidden_states, output_size=None):
        assert hidden_states.shape[1] == self.channels

        # Cast to float32 to as 'upsample_nearest2d_out_frame' op does not support bfloat16
        # TODO(Suraj): Remove this cast once the issue is fixed in PyTorch
        # https://github.com/pytorch/pytorch/issues/86679
        dtype = hidden_states.dtype
        if dtype == torch.bfloat16:
            hidden_states = hidden_states.to(torch.float32)

        # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
        if hidden_states.shape[0] >= 64:
            hidden_states = hidden_states.contiguous()

        # if `output_size` is passed we force the interpolation output size and do not make use of `scale_factor=2`
        if output_size is None:
            hidden_states = F.interpolate(hidden_states, scale_factor=2.0, mode="nearest")
        else:
            hidden_states = F.interpolate(hidden_states, size=output_size, mode="nearest")

        # If the input is bfloat16, we cast back to bfloat16
        if dtype == torch.bfloat16:
            hidden_states = hidden_states.to(dtype)

        hidden_states = self.conv(hidden_states)

        return hidden_states

    def forward(self, hidden_states, output_size=None):
        if self.training and self.gradient_checkpointing:
            # logger.info("Upsample2D: gradient_checkpointing")

            def create_custom_forward(func):
                def custom_forward(*inputs):
                    return func(*inputs)

                return custom_forward

            hidden_states = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.forward_body), hidden_states, output_size, use_reentrant=USE_REENTRANT
            )
        else:
            hidden_states = self.forward_body(hidden_states, output_size)

        return hidden_states


class SdxlUNet2DConditionModel(nn.Module):
    _supports_gradient_checkpointing = True

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__()

        self.in_channels = IN_CHANNELS
        self.out_channels = OUT_CHANNELS
        self.model_channels = MODEL_CHANNELS
        self.time_embed_dim = TIME_EMBED_DIM
        self.adm_in_channels = ADM_IN_CHANNELS

        self.gradient_checkpointing = False
        # self.sample_size = sample_size

        # time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(self.model_channels, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
        )

        # label embedding
        self.label_emb = nn.Sequential(
            nn.Sequential(
                nn.Linear(self.adm_in_channels, self.time_embed_dim),
                nn.SiLU(),
                nn.Linear(self.time_embed_dim, self.time_embed_dim),
            )
        )

        # input
        self.input_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(self.in_channels, self.model_channels, kernel_size=3, padding=(1, 1)),
                )
            ]
        )

        # level 0
        for i in range(2):
            layers = [
                ResnetBlock2D(
                    in_channels=1 * self.model_channels,
                    out_channels=1 * self.model_channels,
                ),
            ]
            self.input_blocks.append(nn.ModuleList(layers))

        self.input_blocks.append(
            nn.Sequential(
                Downsample2D(
                    channels=1 * self.model_channels,
                    out_channels=1 * self.model_channels,
                ),
            )
        )

        # level 1
        for i in range(2):
            layers = [
                ResnetBlock2D(
                    in_channels=(1 if i == 0 else 2) * self.model_channels,
                    out_channels=2 * self.model_channels,
                ),
                Transformer2DModel(
                    num_attention_heads=2 * self.model_channels // 64,
                    attention_head_dim=64,
                    in_channels=2 * self.model_channels,
                    num_transformer_layers=2,
                    use_linear_projection=True,
                    cross_attention_dim=2048,
                ),
            ]
            self.input_blocks.append(nn.ModuleList(layers))

        self.input_blocks.append(
            nn.Sequential(
                Downsample2D(
                    channels=2 * self.model_channels,
                    out_channels=2 * self.model_channels,
                ),
            )
        )

        # level 2
        for i in range(2):
            layers = [
                ResnetBlock2D(
                    in_channels=(2 if i == 0 else 4) * self.model_channels,
                    out_channels=4 * self.model_channels,
                ),
                Transformer2DModel(
                    num_attention_heads=4 * self.model_channels // 64,
                    attention_head_dim=64,
                    in_channels=4 * self.model_channels,
                    num_transformer_layers=10,
                    use_linear_projection=True,
                    cross_attention_dim=2048,
                ),
            ]
            self.input_blocks.append(nn.ModuleList(layers))

        # mid
        self.middle_block = nn.ModuleList(
            [
                ResnetBlock2D(
                    in_channels=4 * self.model_channels,
                    out_channels=4 * self.model_channels,
                ),
                Transformer2DModel(
                    num_attention_heads=4 * self.model_channels // 64,
                    attention_head_dim=64,
                    in_channels=4 * self.model_channels,
                    num_transformer_layers=10,
                    use_linear_projection=True,
                    cross_attention_dim=2048,
                ),
                ResnetBlock2D(
                    in_channels=4 * self.model_channels,
                    out_channels=4 * self.model_channels,
                ),
            ]
        )

        # output
        self.output_blocks = nn.ModuleList([])

        # level 2
        for i in range(3):
            layers = [
                ResnetBlock2D(
                    in_channels=4 * self.model_channels + (4 if i <= 1 else 2) * self.model_channels,
                    out_channels=4 * self.model_channels,
                ),
                Transformer2DModel(
                    num_attention_heads=4 * self.model_channels // 64,
                    attention_head_dim=64,
                    in_channels=4 * self.model_channels,
                    num_transformer_layers=10,
                    use_linear_projection=True,
                    cross_attention_dim=2048,
                ),
            ]
            if i == 2:
                layers.append(
                    Upsample2D(
                        channels=4 * self.model_channels,
                        out_channels=4 * self.model_channels,
                    )
                )

            self.output_blocks.append(nn.ModuleList(layers))

        # level 1
        for i in range(3):
            layers = [
                ResnetBlock2D(
                    in_channels=2 * self.model_channels + (4 if i == 0 else (2 if i == 1 else 1)) * self.model_channels,
                    out_channels=2 * self.model_channels,
                ),
                Transformer2DModel(
                    num_attention_heads=2 * self.model_channels // 64,
                    attention_head_dim=64,
                    in_channels=2 * self.model_channels,
                    num_transformer_layers=2,
                    use_linear_projection=True,
                    cross_attention_dim=2048,
                ),
            ]
            if i == 2:
                layers.append(
                    Upsample2D(
                        channels=2 * self.model_channels,
                        out_channels=2 * self.model_channels,
                    )
                )

            self.output_blocks.append(nn.ModuleList(layers))

        # level 0
        for i in range(3):
            layers = [
                ResnetBlock2D(
                    in_channels=1 * self.model_channels + (2 if i == 0 else 1) * self.model_channels,
                    out_channels=1 * self.model_channels,
                ),
            ]

            self.output_blocks.append(nn.ModuleList(layers))

        # output
        self.out = nn.ModuleList(
            [GroupNorm32(32, self.model_channels), nn.SiLU(), nn.Conv2d(self.model_channels, self.out_channels, 3, padding=1)]
        )

    # region diffusers compatibility
    def prepare_config(self):
        self.config = SimpleNamespace()

    @property
    def dtype(self) -> torch.dtype:
        # `torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        return get_parameter_dtype(self)

    @property
    def device(self) -> torch.device:
        # `torch.device`: The device on which the module is (assuming that all the module parameters are on the same device).
        return get_parameter_device(self)

    def set_attention_slice(self, slice_size):
        raise NotImplementedError("Attention slicing is not supported for this model.")

    def is_gradient_checkpointing(self) -> bool:
        return any(hasattr(m, "gradient_checkpointing") and m.gradient_checkpointing for m in self.modules())

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True
        self.set_gradient_checkpointing(value=True)

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False
        self.set_gradient_checkpointing(value=False)

    def set_use_memory_efficient_attention(self, xformers: bool, mem_eff: bool) -> None:
        blocks = self.input_blocks + [self.middle_block] + self.output_blocks
        for block in blocks:
            for module in block:
                if hasattr(module, "set_use_memory_efficient_attention"):
                    # logger.info(module.__class__.__name__)
                    module.set_use_memory_efficient_attention(xformers, mem_eff)

    def set_use_sdpa(self, sdpa: bool) -> None:
        blocks = self.input_blocks + [self.middle_block] + self.output_blocks
        for block in blocks:
            for module in block:
                if hasattr(module, "set_use_sdpa"):
                    module.set_use_sdpa(sdpa)

    def set_gradient_checkpointing(self, value=False):
        blocks = self.input_blocks + [self.middle_block] + self.output_blocks
        for block in blocks:
            for module in block.modules():
                if hasattr(module, "gradient_checkpointing"):
                    # logger.info(f{module.__class__.__name__} {module.gradient_checkpointing} -> {value}")
                    module.gradient_checkpointing = value

    # endregion

    def forward(self, x, timesteps=None, context=None, y=None, **kwargs):
        # broadcast timesteps to batch dimension
        timesteps = timesteps.expand(x.shape[0])

        hs = []
        t_emb = get_timestep_embedding(timesteps, self.model_channels, downscale_freq_shift=0)  # , repeat_only=False)
        t_emb = t_emb.to(x.dtype)
        emb = self.time_embed(t_emb)

        assert x.shape[0] == y.shape[0], f"batch size mismatch: {x.shape[0]} != {y.shape[0]}"
        assert x.dtype == y.dtype, f"dtype mismatch: {x.dtype} != {y.dtype}"
        # assert x.dtype == self.dtype
        emb = emb + self.label_emb(y)

        def call_module(module, h, emb, context):
            x = h
            for layer in module:
                # logger.info(layer.__class__.__name__, x.dtype, emb.dtype, context.dtype if context is not None else None)
                if isinstance(layer, ResnetBlock2D):
                    x = layer(x, emb)
                elif isinstance(layer, Transformer2DModel):
                    x = layer(x, context)
                else:
                    x = layer(x)
            return x

        # h = x.type(self.dtype)
        h = x

        for module in self.input_blocks:
            h = call_module(module, h, emb, context)
            hs.append(h)

        h = call_module(self.middle_block, h, emb, context)

        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = call_module(module, h, emb, context)

        h = h.type(x.dtype)
        h = call_module(self.out, h, emb, context)

        return h


class InferSdxlUNet2DConditionModel:
    def __init__(self, original_unet: SdxlUNet2DConditionModel, **kwargs):
        self.delegate = original_unet

        # override original model's forward method: because forward is not called by `__call__`
        # overriding `__call__` is not enough, because nn.Module.forward has a special handling
        self.delegate.forward = self.forward

        # Deep Shrink
        self.ds_depth_1 = None
        self.ds_depth_2 = None
        self.ds_timesteps_1 = None
        self.ds_timesteps_2 = None
        self.ds_ratio = None

    # call original model's methods
    def __getattr__(self, name):
        return getattr(self.delegate, name)

    def __call__(self, *args, **kwargs):
        return self.delegate(*args, **kwargs)

    def set_deep_shrink(self, ds_depth_1, ds_timesteps_1=650, ds_depth_2=None, ds_timesteps_2=None, ds_ratio=0.5):
        if ds_depth_1 is None:
            logger.info("Deep Shrink is disabled.")
            self.ds_depth_1 = None
            self.ds_timesteps_1 = None
            self.ds_depth_2 = None
            self.ds_timesteps_2 = None
            self.ds_ratio = None
        else:
            logger.info(
                f"Deep Shrink is enabled: [depth={ds_depth_1}/{ds_depth_2}, timesteps={ds_timesteps_1}/{ds_timesteps_2}, ratio={ds_ratio}]"
            )
            self.ds_depth_1 = ds_depth_1
            self.ds_timesteps_1 = ds_timesteps_1
            self.ds_depth_2 = ds_depth_2 if ds_depth_2 is not None else -1
            self.ds_timesteps_2 = ds_timesteps_2 if ds_timesteps_2 is not None else 1000
            self.ds_ratio = ds_ratio

    def forward(self, x, timesteps=None, context=None, y=None, **kwargs):
        r"""
        current implementation is a copy of `SdxlUNet2DConditionModel.forward()` with Deep Shrink.
        """
        _self = self.delegate

        # broadcast timesteps to batch dimension
        timesteps = timesteps.expand(x.shape[0])

        hs = []
        t_emb = get_timestep_embedding(timesteps, _self.model_channels, downscale_freq_shift=0)  # , repeat_only=False)
        t_emb = t_emb.to(x.dtype)
        emb = _self.time_embed(t_emb)

        assert x.shape[0] == y.shape[0], f"batch size mismatch: {x.shape[0]} != {y.shape[0]}"
        assert x.dtype == y.dtype, f"dtype mismatch: {x.dtype} != {y.dtype}"
        # assert x.dtype == _self.dtype
        emb = emb + _self.label_emb(y)

        def call_module(module, h, emb, context):
            x = h
            for layer in module:
                # print(layer.__class__.__name__, x.dtype, emb.dtype, context.dtype if context is not None else None)
                if isinstance(layer, ResnetBlock2D):
                    x = layer(x, emb)
                elif isinstance(layer, Transformer2DModel):
                    x = layer(x, context)
                else:
                    x = layer(x)
            return x

        # h = x.type(self.dtype)
        h = x

        for depth, module in enumerate(_self.input_blocks):
            # Deep Shrink
            if self.ds_depth_1 is not None:
                if (depth == self.ds_depth_1 and timesteps[0] >= self.ds_timesteps_1) or (
                    self.ds_depth_2 is not None
                    and depth == self.ds_depth_2
                    and timesteps[0] < self.ds_timesteps_1
                    and timesteps[0] >= self.ds_timesteps_2
                ):
                    # print("downsample", h.shape, self.ds_ratio)
                    org_dtype = h.dtype
                    if org_dtype == torch.bfloat16:
                        h = h.to(torch.float32)
                    h = F.interpolate(h, scale_factor=self.ds_ratio, mode="bicubic", align_corners=False).to(org_dtype)

            h = call_module(module, h, emb, context)
            hs.append(h)

        h = call_module(_self.middle_block, h, emb, context)

        for module in _self.output_blocks:
            # Deep Shrink
            if self.ds_depth_1 is not None:
                if hs[-1].shape[-2:] != h.shape[-2:]:
                    # print("upsample", h.shape, hs[-1].shape)
                    h = resize_like(h, hs[-1])

            h = torch.cat([h, hs.pop()], dim=1)
            h = call_module(module, h, emb, context)

        # Deep Shrink: in case of depth 0
        if self.ds_depth_1 == 0 and h.shape[-2:] != x.shape[-2:]:
            # print("upsample", h.shape, x.shape)
            h = resize_like(h, x)

        h = h.type(x.dtype)
        h = call_module(_self.out, h, emb, context)

        return h


if __name__ == "__main__":
    import time

    logger.info("create unet")
    unet = SdxlUNet2DConditionModel()

    unet.to("cuda")
    unet.set_use_memory_efficient_attention(True, False)
    unet.set_gradient_checkpointing(True)
    unet.train()

    # 使用メモリ量確認用の疑似学習ループ
    logger.info("preparing optimizer")

    # optimizer = torch.optim.SGD(unet.parameters(), lr=1e-3, nesterov=True, momentum=0.9) # not working

    # import bitsandbytes
    # optimizer = bitsandbytes.adam.Adam8bit(unet.parameters(), lr=1e-3)        # not working
    # optimizer = bitsandbytes.optim.RMSprop8bit(unet.parameters(), lr=1e-3)  # working at 23.5 GB with torch2
    # optimizer=bitsandbytes.optim.Adagrad8bit(unet.parameters(), lr=1e-3)  # working at 23.5 GB with torch2

    import transformers

    optimizer = transformers.optimization.Adafactor(unet.parameters(), relative_step=True)  # working at 22.2GB with torch2

    scaler = torch.cuda.amp.GradScaler(enabled=True)

    logger.info("start training")
    steps = 10
    batch_size = 1

    for step in range(steps):
        logger.info(f"step {step}")
        if step == 1:
            time_start = time.perf_counter()

        x = torch.randn(batch_size, 4, 128, 128).cuda()  # 1024x1024
        t = torch.randint(low=0, high=10, size=(batch_size,), device="cuda")
        ctx = torch.randn(batch_size, 77, 2048).cuda()
        y = torch.randn(batch_size, ADM_IN_CHANNELS).cuda()

        with torch.cuda.amp.autocast(enabled=True):
            output = unet(x, t, ctx, y)
            target = torch.randn_like(output)
            loss = torch.nn.functional.mse_loss(output, target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    time_end = time.perf_counter()
    logger.info(f"elapsed time: {time_end - time_start} [sec] for last {steps - 1} steps")

def convert_sdxl_text_encoder_2_checkpoint(checkpoint, max_length):
    SDXL_KEY_PREFIX = "conditioner.embedders.1.model."

    # SD2のと、基本的には同じ。logit_scaleを後で使うので、それを追加で返す
    # logit_scaleはcheckpointの保存時に使用する
    def convert_key(key):
        # common conversion
        key = key.replace(SDXL_KEY_PREFIX + "transformer.", "text_model.encoder.")
        key = key.replace(SDXL_KEY_PREFIX, "text_model.")

        if "resblocks" in key:
            # resblocks conversion
            key = key.replace(".resblocks.", ".layers.")
            if ".ln_" in key:
                key = key.replace(".ln_", ".layer_norm")
            elif ".mlp." in key:
                key = key.replace(".c_fc.", ".fc1.")
                key = key.replace(".c_proj.", ".fc2.")
            elif ".attn.out_proj" in key:
                key = key.replace(".attn.out_proj.", ".self_attn.out_proj.")
            elif ".attn.in_proj" in key:
                key = None  # 特殊なので後で処理する
            else:
                raise ValueError(f"unexpected key in SD: {key}")
        elif ".positional_embedding" in key:
            key = key.replace(".positional_embedding", ".embeddings.position_embedding.weight")
        elif ".text_projection" in key:
            key = key.replace("text_model.text_projection", "text_projection.weight")
        elif ".logit_scale" in key:
            key = None  # 後で処理する
        elif ".token_embedding" in key:
            key = key.replace(".token_embedding.weight", ".embeddings.token_embedding.weight")
        elif ".ln_final" in key:
            key = key.replace(".ln_final", ".final_layer_norm")
        # ckpt from comfy has this key: text_model.encoder.text_model.embeddings.position_ids
        elif ".embeddings.position_ids" in key:
            key = None  # remove this key: position_ids is not used in newer transformers
        return key

    keys = list(checkpoint.keys())
    new_sd = {}
    for key in keys:
        new_key = convert_key(key)
        if new_key is None:
            continue
        new_sd[new_key] = checkpoint[key]

    # attnの変換
    for key in keys:
        if ".resblocks" in key and ".attn.in_proj_" in key:
            # 三つに分割
            values = torch.chunk(checkpoint[key], 3)

            key_suffix = ".weight" if "weight" in key else ".bias"
            key_pfx = key.replace(SDXL_KEY_PREFIX + "transformer.resblocks.", "text_model.encoder.layers.")
            key_pfx = key_pfx.replace("_weight", "")
            key_pfx = key_pfx.replace("_bias", "")
            key_pfx = key_pfx.replace(".attn.in_proj", ".self_attn.")
            new_sd[key_pfx + "q_proj" + key_suffix] = values[0]
            new_sd[key_pfx + "k_proj" + key_suffix] = values[1]
            new_sd[key_pfx + "v_proj" + key_suffix] = values[2]

    # logit_scale はDiffusersには含まれないが、保存時に戻したいので別途返す
    logit_scale = checkpoint.get(SDXL_KEY_PREFIX + "logit_scale", None)

    # temporary workaround for text_projection.weight.weight for Playground-v2
    if "text_projection.weight.weight" in new_sd:
        logger.info("convert_sdxl_text_encoder_2_checkpoint: convert text_projection.weight.weight to text_projection.weight")
        new_sd["text_projection.weight"] = new_sd["text_projection.weight.weight"]
        del new_sd["text_projection.weight.weight"]

    return new_sd, logit_scale


# load state_dict without allocating new tensors
def _load_state_dict_on_device(model, state_dict, device, dtype=None):
    # dtype will use fp32 as default
    missing_keys = list(model.state_dict().keys() - state_dict.keys())
    unexpected_keys = list(state_dict.keys() - model.state_dict().keys())

    # similar to model.load_state_dict()
    if not missing_keys and not unexpected_keys:
        for k in list(state_dict.keys()):
            set_module_tensor_to_device(model, k, device, value=state_dict.pop(k), dtype=dtype)
        return "<All keys matched successfully>"

    # error_msgs
    error_msgs: List[str] = []
    if missing_keys:
        error_msgs.insert(0, "Missing key(s) in state_dict: {}. ".format(", ".join('"{}"'.format(k) for k in missing_keys)))
    if unexpected_keys:
        error_msgs.insert(0, "Unexpected key(s) in state_dict: {}. ".format(", ".join('"{}"'.format(k) for k in unexpected_keys)))

    raise RuntimeError("Error(s) in loading state_dict for {}:\n\t{}".format(model.__class__.__name__, "\n\t".join(error_msgs)))


def load_models_from_sdxl_checkpoint(model_version, ckpt_path, map_location, dtype=None):
    # model_version is reserved for future use
    # dtype is used for full_fp16/bf16 integration. Text Encoder will remain fp32, because it runs on CPU when caching

    # Load the state dict
    if is_safetensors(ckpt_path):
        checkpoint = None
        try:
            state_dict = load_file(ckpt_path, device=map_location)
        except:
            state_dict = load_file(ckpt_path)  # prevent device invalid Error
        epoch = None
        global_step = None
    else:
        checkpoint = torch.load(ckpt_path, map_location=map_location)
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint.get("epoch", 0)
            global_step = checkpoint.get("global_step", 0)
        else:
            state_dict = checkpoint
            epoch = 0
            global_step = 0
        checkpoint = None

    # U-Net
    logger.info("building U-Net")
    with init_empty_weights():
        unet = SdxlUNet2DConditionModel()

    logger.info("loading U-Net from checkpoint")
    unet_sd = {}
    for k in list(state_dict.keys()):
        if k.startswith("model.diffusion_model."):
            unet_sd[k.replace("model.diffusion_model.", "")] = state_dict.pop(k)
    info = _load_state_dict_on_device(unet, unet_sd, device=map_location, dtype=dtype)
    logger.info(f"U-Net: {info}")

    # Text Encoders
    logger.info("building text encoders")

    # Text Encoder 1 is same to Stability AI's SDXL
    text_model1_cfg = CLIPTextConfig(
        vocab_size=49408,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        max_position_embeddings=77,
        hidden_act="quick_gelu",
        layer_norm_eps=1e-05,
        dropout=0.0,
        attention_dropout=0.0,
        initializer_range=0.02,
        initializer_factor=1.0,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        model_type="clip_text_model",
        projection_dim=768,
        # torch_dtype="float32",
        # transformers_version="4.25.0.dev0",
    )
    with init_empty_weights():
        text_model1 = CLIPTextModel._from_config(text_model1_cfg)

    # Text Encoder 2 is different from Stability AI's SDXL. SDXL uses open clip, but we use the model from HuggingFace.
    # Note: Tokenizer from HuggingFace is different from SDXL. We must use open clip's tokenizer.
    text_model2_cfg = CLIPTextConfig(
        vocab_size=49408,
        hidden_size=1280,
        intermediate_size=5120,
        num_hidden_layers=32,
        num_attention_heads=20,
        max_position_embeddings=77,
        hidden_act="gelu",
        layer_norm_eps=1e-05,
        dropout=0.0,
        attention_dropout=0.0,
        initializer_range=0.02,
        initializer_factor=1.0,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        model_type="clip_text_model",
        projection_dim=1280,
        # torch_dtype="float32",
        # transformers_version="4.25.0.dev0",
    )
    with init_empty_weights():
        text_model2 = CLIPTextModelWithProjection(text_model2_cfg)

    logger.info("loading text encoders from checkpoint")
    te1_sd = {}
    te2_sd = {}
    for k in list(state_dict.keys()):
        if k.startswith("conditioner.embedders.0.transformer."):
            te1_sd[k.replace("conditioner.embedders.0.transformer.", "")] = state_dict.pop(k)
        elif k.startswith("conditioner.embedders.1.model."):
            te2_sd[k] = state_dict.pop(k)

    # 最新の transformers では position_ids を含むとエラーになるので削除 / remove position_ids for latest transformers
    if "text_model.embeddings.position_ids" in te1_sd:
        te1_sd.pop("text_model.embeddings.position_ids")

    info1 = _load_state_dict_on_device(text_model1, te1_sd, device=map_location)  # remain fp32
    logger.info(f"text encoder 1: {info1}")

    converted_sd, logit_scale = convert_sdxl_text_encoder_2_checkpoint(te2_sd, max_length=77)
    info2 = _load_state_dict_on_device(text_model2, converted_sd, device=map_location)  # remain fp32
    logger.info(f"text encoder 2: {info2}")

    # prepare vae
    logger.info("building VAE")
    vae_config = create_vae_diffusers_config()
    with init_empty_weights():
        vae = AutoencoderKL(**vae_config)

    logger.info("loading VAE from checkpoint")
    converted_vae_checkpoint = convert_ldm_vae_checkpoint(state_dict, vae_config)
    info = _load_state_dict_on_device(vae, converted_vae_checkpoint, device=map_location, dtype=dtype)
    logger.info(f"VAE: {info}")

    ckpt_info = (epoch, global_step) if epoch is not None else None
    return text_model1, text_model2, vae, unet, logit_scale, ckpt_info


def make_unet_conversion_map():
    unet_conversion_map_layer = []

    for i in range(3):  # num_blocks is 3 in sdxl
        # loop over downblocks/upblocks
        for j in range(2):
            # loop over resnets/attentions for downblocks
            hf_down_res_prefix = f"down_blocks.{i}.resnets.{j}."
            sd_down_res_prefix = f"input_blocks.{3*i + j + 1}.0."
            unet_conversion_map_layer.append((sd_down_res_prefix, hf_down_res_prefix))

            if i < 3:
                # no attention layers in down_blocks.3
                hf_down_atn_prefix = f"down_blocks.{i}.attentions.{j}."
                sd_down_atn_prefix = f"input_blocks.{3*i + j + 1}.1."
                unet_conversion_map_layer.append((sd_down_atn_prefix, hf_down_atn_prefix))

        for j in range(3):
            # loop over resnets/attentions for upblocks
            hf_up_res_prefix = f"up_blocks.{i}.resnets.{j}."
            sd_up_res_prefix = f"output_blocks.{3*i + j}.0."
            unet_conversion_map_layer.append((sd_up_res_prefix, hf_up_res_prefix))

            # if i > 0: commentout for sdxl
            # no attention layers in up_blocks.0
            hf_up_atn_prefix = f"up_blocks.{i}.attentions.{j}."
            sd_up_atn_prefix = f"output_blocks.{3*i + j}.1."
            unet_conversion_map_layer.append((sd_up_atn_prefix, hf_up_atn_prefix))

        if i < 3:
            # no downsample in down_blocks.3
            hf_downsample_prefix = f"down_blocks.{i}.downsamplers.0.conv."
            sd_downsample_prefix = f"input_blocks.{3*(i+1)}.0.op."
            unet_conversion_map_layer.append((sd_downsample_prefix, hf_downsample_prefix))

            # no upsample in up_blocks.3
            hf_upsample_prefix = f"up_blocks.{i}.upsamplers.0."
            sd_upsample_prefix = f"output_blocks.{3*i + 2}.{2}."  # change for sdxl
            unet_conversion_map_layer.append((sd_upsample_prefix, hf_upsample_prefix))

    hf_mid_atn_prefix = "mid_block.attentions.0."
    sd_mid_atn_prefix = "middle_block.1."
    unet_conversion_map_layer.append((sd_mid_atn_prefix, hf_mid_atn_prefix))

    for j in range(2):
        hf_mid_res_prefix = f"mid_block.resnets.{j}."
        sd_mid_res_prefix = f"middle_block.{2*j}."
        unet_conversion_map_layer.append((sd_mid_res_prefix, hf_mid_res_prefix))

    unet_conversion_map_resnet = [
        # (stable-diffusion, HF Diffusers)
        ("in_layers.0.", "norm1."),
        ("in_layers.2.", "conv1."),
        ("out_layers.0.", "norm2."),
        ("out_layers.3.", "conv2."),
        ("emb_layers.1.", "time_emb_proj."),
        ("skip_connection.", "conv_shortcut."),
    ]

    unet_conversion_map = []
    for sd, hf in unet_conversion_map_layer:
        if "resnets" in hf:
            for sd_res, hf_res in unet_conversion_map_resnet:
                unet_conversion_map.append((sd + sd_res, hf + hf_res))
        else:
            unet_conversion_map.append((sd, hf))

    for j in range(2):
        hf_time_embed_prefix = f"time_embedding.linear_{j+1}."
        sd_time_embed_prefix = f"time_embed.{j*2}."
        unet_conversion_map.append((sd_time_embed_prefix, hf_time_embed_prefix))

    for j in range(2):
        hf_label_embed_prefix = f"add_embedding.linear_{j+1}."
        sd_label_embed_prefix = f"label_emb.0.{j*2}."
        unet_conversion_map.append((sd_label_embed_prefix, hf_label_embed_prefix))

    unet_conversion_map.append(("input_blocks.0.0.", "conv_in."))
    unet_conversion_map.append(("out.0.", "conv_norm_out."))
    unet_conversion_map.append(("out.2.", "conv_out."))

    return unet_conversion_map


def convert_diffusers_unet_state_dict_to_sdxl(du_sd):
    unet_conversion_map = make_unet_conversion_map()

    conversion_map = {hf: sd for sd, hf in unet_conversion_map}
    return convert_unet_state_dict(du_sd, conversion_map)


def convert_unet_state_dict(src_sd, conversion_map):
    converted_sd = {}
    for src_key, value in src_sd.items():
        # さすがに全部回すのは時間がかかるので右から要素を削りつつprefixを探す
        src_key_fragments = src_key.split(".")[:-1]  # remove weight/bias
        while len(src_key_fragments) > 0:
            src_key_prefix = ".".join(src_key_fragments) + "."
            if src_key_prefix in conversion_map:
                converted_prefix = conversion_map[src_key_prefix]
                converted_key = converted_prefix + src_key[len(src_key_prefix) :]
                converted_sd[converted_key] = value
                break
            src_key_fragments.pop(-1)
        assert len(src_key_fragments) > 0, f"key {src_key} not found in conversion map"

    return converted_sd


def convert_sdxl_unet_state_dict_to_diffusers(sd):
    unet_conversion_map = make_unet_conversion_map()

    conversion_dict = {sd: hf for sd, hf in unet_conversion_map}
    return convert_unet_state_dict(sd, conversion_dict)


def convert_text_encoder_2_state_dict_to_sdxl(checkpoint, logit_scale):
    def convert_key(key):
        # position_idsの除去
        if ".position_ids" in key:
            return None

        # common
        key = key.replace("text_model.encoder.", "transformer.")
        key = key.replace("text_model.", "")
        if "layers" in key:
            # resblocks conversion
            key = key.replace(".layers.", ".resblocks.")
            if ".layer_norm" in key:
                key = key.replace(".layer_norm", ".ln_")
            elif ".mlp." in key:
                key = key.replace(".fc1.", ".c_fc.")
                key = key.replace(".fc2.", ".c_proj.")
            elif ".self_attn.out_proj" in key:
                key = key.replace(".self_attn.out_proj.", ".attn.out_proj.")
            elif ".self_attn." in key:
                key = None  # 特殊なので後で処理する
            else:
                raise ValueError(f"unexpected key in DiffUsers model: {key}")
        elif ".position_embedding" in key:
            key = key.replace("embeddings.position_embedding.weight", "positional_embedding")
        elif ".token_embedding" in key:
            key = key.replace("embeddings.token_embedding.weight", "token_embedding.weight")
        elif "text_projection" in key:  # no dot in key
            key = key.replace("text_projection.weight", "text_projection")
        elif "final_layer_norm" in key:
            key = key.replace("final_layer_norm", "ln_final")
        return key

    keys = list(checkpoint.keys())
    new_sd = {}
    for key in keys:
        new_key = convert_key(key)
        if new_key is None:
            continue
        new_sd[new_key] = checkpoint[key]

    # attnの変換
    for key in keys:
        if "layers" in key and "q_proj" in key:
            # 三つを結合
            key_q = key
            key_k = key.replace("q_proj", "k_proj")
            key_v = key.replace("q_proj", "v_proj")

            value_q = checkpoint[key_q]
            value_k = checkpoint[key_k]
            value_v = checkpoint[key_v]
            value = torch.cat([value_q, value_k, value_v])

            new_key = key.replace("text_model.encoder.layers.", "transformer.resblocks.")
            new_key = new_key.replace(".self_attn.q_proj.", ".attn.in_proj_")
            new_sd[new_key] = value

    if logit_scale is not None:
        new_sd["logit_scale"] = logit_scale

    return new_sd


def save_stable_diffusion_checkpoint(
    output_file,
    text_encoder1,
    text_encoder2,
    unet,
    epochs,
    steps,
    ckpt_info,
    vae,
    logit_scale,
    metadata,
    save_dtype=None,
):
    state_dict = {}

    def update_sd(prefix, sd):
        for k, v in sd.items():
            key = prefix + k
            if save_dtype is not None:
                v = v.detach().clone().to("cpu").to(save_dtype)
            state_dict[key] = v

    # Convert the UNet model
    update_sd("model.diffusion_model.", unet.state_dict())

    # Convert the text encoders
    update_sd("conditioner.embedders.0.transformer.", text_encoder1.state_dict())

    text_enc2_dict = convert_text_encoder_2_state_dict_to_sdxl(text_encoder2.state_dict(), logit_scale)
    update_sd("conditioner.embedders.1.model.", text_enc2_dict)

    # Convert the VAE
    vae_dict = convert_vae_state_dict(vae.state_dict())
    update_sd("first_stage_model.", vae_dict)

    # Put together new checkpoint
    key_count = len(state_dict.keys())
    new_ckpt = {"state_dict": state_dict}

    # epoch and global_step are sometimes not int
    if ckpt_info is not None:
        epochs += ckpt_info[0]
        steps += ckpt_info[1]

    new_ckpt["epoch"] = epochs
    new_ckpt["global_step"] = steps

    if is_safetensors(output_file):
        save_file(state_dict, output_file, metadata)
    else:
        torch.save(new_ckpt, output_file)

    return key_count


def save_diffusers_checkpoint(
    output_dir, text_encoder1, text_encoder2, unet, pretrained_model_name_or_path, vae=None, use_safetensors=False, save_dtype=None
):
    from diffusers import StableDiffusionXLPipeline

    # convert U-Net
    unet_sd = unet.state_dict()
    du_unet_sd = convert_sdxl_unet_state_dict_to_diffusers(unet_sd)

    diffusers_unet = UNet2DConditionModel(**DIFFUSERS_SDXL_UNET_CONFIG)
    if save_dtype is not None:
        diffusers_unet.to(save_dtype)
    diffusers_unet.load_state_dict(du_unet_sd)

    # create pipeline to save
    if pretrained_model_name_or_path is None:
        pretrained_model_name_or_path = DIFFUSERS_REF_MODEL_ID_SDXL

    scheduler = EulerDiscreteScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer1 = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer")
    tokenizer2 = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer_2")
    if vae is None:
        vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae")

    # prevent local path from being saved
    def remove_name_or_path(model):
        if hasattr(model, "config"):
            model.config._name_or_path = None
            model.config._name_or_path = None

    remove_name_or_path(diffusers_unet)
    remove_name_or_path(text_encoder1)
    remove_name_or_path(text_encoder2)
    remove_name_or_path(scheduler)
    remove_name_or_path(tokenizer1)
    remove_name_or_path(tokenizer2)
    remove_name_or_path(vae)

    pipeline = StableDiffusionXLPipeline(
        unet=diffusers_unet,
        text_encoder=text_encoder1,
        text_encoder_2=text_encoder2,
        vae=vae,
        scheduler=scheduler,
        tokenizer=tokenizer1,
        tokenizer_2=tokenizer2,
    )
    if save_dtype is not None:
        pipeline.to(None, save_dtype)
    pipeline.save_pretrained(output_dir, safe_serialization=use_safetensors)

def finalsave(input_ckpt_path, model_output_dir):
    # Assuming the model version and map location are predefined or can be inferred
    model_version = MODEL_VERSION_SDXL_BASE_V1_0
    map_location = torch.device('cpu')  # or 'cuda' for GPU

    # Load models from the checkpoint
    text_model1, text_model2, vae, unet, logit_scale, ckpt_info = load_models_from_sdxl_checkpoint(
        model_version, input_ckpt_path, map_location
    )

    # Save the converted models into the Diffusers format
    save_diffusers_checkpoint(
        model_output_dir, text_model1, text_model2, unet, DIFFUSERS_REF_MODEL_ID_SDXL, vae, use_safetensors=True
    )

    print(f"Models have been converted and saved to {model_output_dir}.")
