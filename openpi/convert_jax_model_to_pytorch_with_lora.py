#!/usr/bin/env python3
"""
Convert a JAX OpenPI checkpoint to PyTorch, with explicit LoRA merge support.

This script is intended for LoRA fine-tuned checkpoints where the original
`convert_jax_model_to_pytorch.py` would otherwise ignore `lora_a/lora_b`
weights and export only the base weights.
"""

import json
import os
import pathlib
import shutil
from typing import Literal

from flax.nnx import traversals
import numpy as np
import orbax.checkpoint as ocp
import safetensors
import torch
import tyro

import openpi.models.gemma
import openpi.models.model
import openpi.models.pi0_config
import openpi.models_pytorch.pi0_pytorch
from openpi.training import utils
import openpi.training.config as _config


def _to_numpy(x):
    return np.asarray(x)


def _merge_last_two_dims(base: np.ndarray, lora_a: np.ndarray | None, lora_b: np.ndarray | None, scaling: float) -> np.ndarray:
    base = _to_numpy(base).astype(np.float32, copy=False)
    if lora_a is None or lora_b is None:
        return base
    delta = np.matmul(_to_numpy(lora_a).astype(np.float32, copy=False), _to_numpy(lora_b).astype(np.float32, copy=False))
    return base + delta * float(scaling)


def _lora_scaling(config, key: str) -> float:
    if key == "attn":
        lora_cfg = config.lora_configs.get("attn")
    elif key == "ffn":
        lora_cfg = config.lora_configs.get("ffn")
    else:
        raise ValueError(f"Unknown LoRA config key: {key}")
    return 1.0 if lora_cfg is None else float(lora_cfg.scaling_value)


def _pop_merged_einsum(state_dict: dict, key_prefix: str, suffix: str, scaling: float) -> np.ndarray:
    base = state_dict.pop(f"{key_prefix}/w{suffix}")
    lora_a = state_dict.pop(f"{key_prefix}/lora_a{suffix}", None)
    lora_b = state_dict.pop(f"{key_prefix}/lora_b{suffix}", None)
    return _merge_last_two_dims(base, lora_a, lora_b, scaling)


def _pop_merged_ffn_tensor(state_dict: dict, key_prefix: str, suffix: str, scaling: float) -> np.ndarray:
    base = state_dict.pop(f"{key_prefix}{suffix}")
    lora_a = state_dict.pop(f"{key_prefix}_lora_a{suffix}", None)
    lora_b = state_dict.pop(f"{key_prefix}_lora_b{suffix}", None)
    return _merge_last_two_dims(base, lora_a, lora_b, scaling)


def slice_paligemma_state_dict(state_dict, config, *, paligemma_lora_config, pi05: bool):
    """Convert PaliGemma JAX parameters to PyTorch format, merging LoRA weights when present."""
    suffix = "/value" if "img/embedding/kernel/value" in state_dict else ""

    # patch embeddings
    jax_key = f"img/embedding/kernel{suffix}"
    pytorch_key = "paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.patch_embedding.weight"
    state_dict[pytorch_key] = state_dict.pop(jax_key).transpose(3, 2, 0, 1)

    jax_key = f"img/embedding/bias{suffix}"
    pytorch_key = "paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.patch_embedding.bias"
    state_dict[pytorch_key] = state_dict.pop(jax_key)

    # positional embeddings
    jax_key = f"img/pos_embedding{suffix}"
    pytorch_key = "paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.position_embedding.weight"
    state_dict[pytorch_key] = state_dict.pop(jax_key).reshape(-1, config.vision_config.hidden_size)

    # vision layers
    encoderblock_layernorm0_scale = state_dict.pop(f"img/Transformer/encoderblock/LayerNorm_0/scale{suffix}")
    encoderblock_layernorm0_bias = state_dict.pop(f"img/Transformer/encoderblock/LayerNorm_0/bias{suffix}")
    encoderblock_layernorm1_scale = state_dict.pop(f"img/Transformer/encoderblock/LayerNorm_1/scale{suffix}")
    encoderblock_layernorm1_bias = state_dict.pop(f"img/Transformer/encoderblock/LayerNorm_1/bias{suffix}")

    encoderblock_mlp_dense0_kernel = state_dict.pop(f"img/Transformer/encoderblock/MlpBlock_0/Dense_0/kernel{suffix}")
    encoderblock_mlp_dense0_bias = state_dict.pop(f"img/Transformer/encoderblock/MlpBlock_0/Dense_0/bias{suffix}")
    encoderblock_mlp_dense1_kernel = state_dict.pop(f"img/Transformer/encoderblock/MlpBlock_0/Dense_1/kernel{suffix}")
    encoderblock_mlp_dense1_bias = state_dict.pop(f"img/Transformer/encoderblock/MlpBlock_0/Dense_1/bias{suffix}")

    encoderblock_attention_0_key_kernel = state_dict.pop(
        f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/key/kernel{suffix}"
    )
    encoderblock_attention_0_key_bias = state_dict.pop(
        f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/key/bias{suffix}"
    )
    encoderblock_attention_0_value_kernel = state_dict.pop(
        f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/value/kernel{suffix}"
    )
    encoderblock_attention_0_value_bias = state_dict.pop(
        f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/value/bias{suffix}"
    )
    encoderblock_attention_0_query_kernel = state_dict.pop(
        f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/query/kernel{suffix}"
    )
    encoderblock_attention_0_query_bias = state_dict.pop(
        f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/query/bias{suffix}"
    )
    encoderblock_attention_0_out_kernel = state_dict.pop(
        f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/out/kernel{suffix}"
    )
    encoderblock_attention_0_out_bias = state_dict.pop(
        f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/out/bias{suffix}"
    )

    for i in range(config.vision_config.num_hidden_layers):
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.layer_norm1.weight"
        ] = encoderblock_layernorm0_scale[i].transpose()
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.layer_norm1.bias"
        ] = encoderblock_layernorm0_bias[i]
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.layer_norm2.weight"
        ] = encoderblock_layernorm1_scale[i].transpose()
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.layer_norm2.bias"
        ] = encoderblock_layernorm1_bias[i]
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.mlp.fc1.weight"
        ] = encoderblock_mlp_dense0_kernel[i].transpose()
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.mlp.fc1.bias"
        ] = encoderblock_mlp_dense0_bias[i]
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.mlp.fc2.weight"
        ] = encoderblock_mlp_dense1_kernel[i].transpose()
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.mlp.fc2.bias"
        ] = encoderblock_mlp_dense1_bias[i]
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.self_attn.k_proj.weight"
        ] = encoderblock_attention_0_key_kernel[i].reshape(-1, config.vision_config.hidden_size).transpose()
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.self_attn.k_proj.bias"
        ] = encoderblock_attention_0_key_bias[i].reshape(-1, config.vision_config.hidden_size).reshape(-1)
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.self_attn.v_proj.weight"
        ] = encoderblock_attention_0_value_kernel[i].reshape(-1, config.vision_config.hidden_size).transpose()
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.self_attn.v_proj.bias"
        ] = encoderblock_attention_0_value_bias[i].reshape(-1, config.vision_config.hidden_size).reshape(-1)
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.self_attn.q_proj.weight"
        ] = encoderblock_attention_0_query_kernel[i].reshape(-1, config.vision_config.hidden_size).transpose()
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.self_attn.q_proj.bias"
        ] = encoderblock_attention_0_query_bias[i].reshape(-1, config.vision_config.hidden_size).reshape(-1)
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.self_attn.out_proj.weight"
        ] = encoderblock_attention_0_out_kernel[i].reshape(-1, config.vision_config.hidden_size).transpose()
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.self_attn.out_proj.bias"
        ] = encoderblock_attention_0_out_bias[i].reshape(-1, config.vision_config.hidden_size).reshape(-1)

    jax_key = f"img/Transformer/encoder_norm/scale{suffix}"
    pytorch_key = "paligemma_with_expert.paligemma.model.vision_tower.vision_model.post_layernorm.weight"
    state_dict[pytorch_key] = state_dict.pop(jax_key).transpose()

    jax_key = f"img/Transformer/encoder_norm/bias{suffix}"
    pytorch_key = "paligemma_with_expert.paligemma.model.vision_tower.vision_model.post_layernorm.bias"
    state_dict[pytorch_key] = state_dict.pop(jax_key)

    jax_key = f"img/head/kernel{suffix}"
    pytorch_key = "paligemma_with_expert.paligemma.model.multi_modal_projector.linear.weight"
    state_dict[pytorch_key] = state_dict.pop(jax_key).transpose()

    jax_key = f"img/head/bias{suffix}"
    pytorch_key = "paligemma_with_expert.paligemma.model.multi_modal_projector.linear.bias"
    state_dict[pytorch_key] = state_dict.pop(jax_key)

    jax_key = f"llm/embedder/input_embedding{suffix}"
    pytorch_key = "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
    state_dict[pytorch_key] = state_dict.pop(jax_key)

    attn_scaling = _lora_scaling(paligemma_lora_config, "attn")
    ffn_scaling = _lora_scaling(paligemma_lora_config, "ffn")
    llm_attention_attn_vec_einsum = _pop_merged_einsum(
        state_dict, "llm/layers/attn/attn_vec_einsum", suffix, attn_scaling
    )
    llm_attention_kv_einsum = _pop_merged_einsum(state_dict, "llm/layers/attn/kv_einsum", suffix, attn_scaling)
    llm_attention_q_einsum = _pop_merged_einsum(state_dict, "llm/layers/attn/q_einsum", suffix, attn_scaling)

    llm_mlp_gating_einsum = _pop_merged_ffn_tensor(state_dict, "llm/layers/mlp/gating_einsum", suffix, ffn_scaling)
    llm_mlp_linear = _pop_merged_ffn_tensor(state_dict, "llm/layers/mlp/linear", suffix, ffn_scaling)

    llm_input_layernorm = state_dict.pop(f"llm/layers/pre_attention_norm/scale{suffix}")
    llm_post_attention_layernorm = state_dict.pop(f"llm/layers/pre_ffw_norm/scale{suffix}")

    for i in range(config.text_config.num_hidden_layers):
        q_proj_weight_reshaped = (
            llm_attention_q_einsum[i]
            .transpose(0, 2, 1)
            .reshape(
                config.text_config.num_attention_heads * config.text_config.head_dim, config.text_config.hidden_size
            )
        )
        state_dict[f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.self_attn.q_proj.weight"] = (
            q_proj_weight_reshaped
        )

        k_proj_weight_reshaped = llm_attention_kv_einsum[i, 0, 0].transpose()
        state_dict[f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.self_attn.k_proj.weight"] = (
            k_proj_weight_reshaped
        )
        v_proj_weight_reshaped = llm_attention_kv_einsum[i, 1, 0].transpose()
        state_dict[f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.self_attn.v_proj.weight"] = (
            v_proj_weight_reshaped
        )

        o_proj_weight_reshaped = (
            llm_attention_attn_vec_einsum[i]
            .transpose(2, 0, 1)
            .reshape(
                config.text_config.num_attention_heads * config.text_config.head_dim, config.text_config.hidden_size
            )
        )
        state_dict[f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.self_attn.o_proj.weight"] = (
            o_proj_weight_reshaped
        )

        gate_proj_weight = llm_mlp_gating_einsum[i, 0]
        state_dict[f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.mlp.gate_proj.weight"] = (
            gate_proj_weight.transpose()
        )
        up_proj_weight = llm_mlp_gating_einsum[i, 1]
        state_dict[f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.mlp.up_proj.weight"] = (
            up_proj_weight.transpose()
        )
        state_dict[f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.mlp.down_proj.weight"] = (
            llm_mlp_linear[i].transpose()
        )
        state_dict[f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.input_layernorm.weight"] = (
            llm_input_layernorm[i]
        )
        state_dict[
            f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.post_attention_layernorm.weight"
        ] = llm_post_attention_layernorm[i]

    jax_key = f"llm/final_norm/scale{suffix}"
    pytorch_key = "paligemma_with_expert.paligemma.model.language_model.norm.weight"
    state_dict[pytorch_key] = state_dict.pop(jax_key)

    expert_dict = {}
    final_state_dict = {}
    expert_keys = [
        f"llm/final_norm_1/scale{suffix}",
        f"llm/final_norm_1/Dense_0/bias{suffix}",
        f"llm/final_norm_1/Dense_0/kernel{suffix}",
        f"llm/layers/attn/attn_vec_einsum_1/w{suffix}",
        f"llm/layers/attn/attn_vec_einsum_1/lora_a{suffix}",
        f"llm/layers/attn/attn_vec_einsum_1/lora_b{suffix}",
        f"llm/layers/attn/kv_einsum_1/w{suffix}",
        f"llm/layers/attn/kv_einsum_1/lora_a{suffix}",
        f"llm/layers/attn/kv_einsum_1/lora_b{suffix}",
        f"llm/layers/attn/q_einsum_1/w{suffix}",
        f"llm/layers/attn/q_einsum_1/lora_a{suffix}",
        f"llm/layers/attn/q_einsum_1/lora_b{suffix}",
        f"llm/layers/mlp_1/gating_einsum{suffix}",
        f"llm/layers/mlp_1/gating_einsum_lora_a{suffix}",
        f"llm/layers/mlp_1/gating_einsum_lora_b{suffix}",
        f"llm/layers/mlp_1/linear{suffix}",
        f"llm/layers/mlp_1/linear_lora_a{suffix}",
        f"llm/layers/mlp_1/linear_lora_b{suffix}",
        f"llm/layers/pre_attention_norm_1/scale{suffix}",
        f"llm/layers/pre_attention_norm_1/Dense_0/bias{suffix}",
        f"llm/layers/pre_attention_norm_1/Dense_0/kernel{suffix}",
        f"llm/layers/pre_ffw_norm_1/scale{suffix}",
        f"llm/layers/pre_ffw_norm_1/Dense_0/bias{suffix}",
        f"llm/layers/pre_ffw_norm_1/Dense_0/kernel{suffix}",
    ]

    for key, value in state_dict.items():
        if key not in expert_keys:
            final_state_dict[key] = torch.from_numpy(_to_numpy(value))
        else:
            expert_dict[key] = value

    return final_state_dict, expert_dict


def slice_gemma_state_dict(state_dict, config, *, num_expert, pi05: bool):
    """Convert Gemma JAX expert parameters to PyTorch format, merging LoRA weights when present."""
    if not hasattr(config, "vocab_size"):
        config.vocab_size = 257152
    if not hasattr(config, "hidden_size"):
        config.hidden_size = config.width
    if not hasattr(config, "num_hidden_layers"):
        config.num_hidden_layers = config.depth
    if not hasattr(config, "num_attention_heads"):
        config.num_attention_heads = config.num_heads

    suffix = "/value" if f"llm/layers/attn/attn_vec_einsum_{num_expert}/w/value" in state_dict else ""

    attn_scaling = _lora_scaling(config, "attn")
    ffn_scaling = _lora_scaling(config, "ffn")
    llm_attention_attn_vec_einsum = _pop_merged_einsum(
        state_dict, f"llm/layers/attn/attn_vec_einsum_{num_expert}", suffix, attn_scaling
    )
    llm_attention_kv_einsum = _pop_merged_einsum(
        state_dict, f"llm/layers/attn/kv_einsum_{num_expert}", suffix, attn_scaling
    )
    llm_attention_q_einsum = _pop_merged_einsum(
        state_dict, f"llm/layers/attn/q_einsum_{num_expert}", suffix, attn_scaling
    )

    llm_mlp_gating_einsum = _pop_merged_ffn_tensor(
        state_dict, f"llm/layers/mlp_{num_expert}/gating_einsum", suffix, ffn_scaling
    )
    llm_mlp_linear = _pop_merged_ffn_tensor(state_dict, f"llm/layers/mlp_{num_expert}/linear", suffix, ffn_scaling)

    if pi05:
        llm_input_layernorm_bias = state_dict.pop(f"llm/layers/pre_attention_norm_{num_expert}/Dense_0/bias{suffix}")
        llm_post_attention_layernorm_bias = state_dict.pop(f"llm/layers/pre_ffw_norm_{num_expert}/Dense_0/bias{suffix}")
        llm_input_layernorm_kernel = state_dict.pop(
            f"llm/layers/pre_attention_norm_{num_expert}/Dense_0/kernel{suffix}"
        )
        llm_post_attention_layernorm_kernel = state_dict.pop(
            f"llm/layers/pre_ffw_norm_{num_expert}/Dense_0/kernel{suffix}"
        )
    else:
        llm_input_layernorm = state_dict.pop(f"llm/layers/pre_attention_norm_{num_expert}/scale{suffix}")
        llm_post_attention_layernorm = state_dict.pop(f"llm/layers/pre_ffw_norm_{num_expert}/scale{suffix}")

    for i in range(config.num_hidden_layers):
        q_proj_weight_reshaped = (
            llm_attention_q_einsum[i]
            .transpose(0, 2, 1)
            .reshape(config.num_attention_heads * config.head_dim, config.hidden_size)
        )
        state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.self_attn.q_proj.weight"] = (
            q_proj_weight_reshaped
        )

        k_proj_weight_reshaped = llm_attention_kv_einsum[i, 0, 0].transpose()
        state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.self_attn.k_proj.weight"] = (
            k_proj_weight_reshaped
        )
        v_proj_weight_reshaped = llm_attention_kv_einsum[i, 1, 0].transpose()
        state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.self_attn.v_proj.weight"] = (
            v_proj_weight_reshaped
        )

        o_proj_weight_reshaped = (
            llm_attention_attn_vec_einsum[i]
            .reshape(config.num_attention_heads * config.head_dim, config.hidden_size)
            .transpose(1, 0)
        )
        state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.self_attn.o_proj.weight"] = (
            o_proj_weight_reshaped
        )

        gate_proj_weight = llm_mlp_gating_einsum[i, 0]
        state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.mlp.gate_proj.weight"] = (
            gate_proj_weight.transpose()
        )
        up_proj_weight = llm_mlp_gating_einsum[i, 1]
        state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.mlp.up_proj.weight"] = (
            up_proj_weight.transpose()
        )
        state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.mlp.down_proj.weight"] = (
            llm_mlp_linear[i].transpose()
        )

        if pi05:
            state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.input_layernorm.dense.bias"] = (
                llm_input_layernorm_bias[i]
            )
            state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.post_attention_layernorm.dense.bias"] = (
                llm_post_attention_layernorm_bias[i]
            )
            state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.input_layernorm.dense.weight"] = (
                llm_input_layernorm_kernel[i].transpose()
            )
            state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.post_attention_layernorm.dense.weight"] = (
                llm_post_attention_layernorm_kernel[i].transpose()
            )
        else:
            state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.input_layernorm.weight"] = (
                llm_input_layernorm[i]
            )
            state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.post_attention_layernorm.weight"] = (
                llm_post_attention_layernorm[i]
            )

    if pi05:
        final_norm_bias = state_dict.pop(f"llm/final_norm_{num_expert}/Dense_0/bias{suffix}")
        final_norm_kernel = state_dict.pop(f"llm/final_norm_{num_expert}/Dense_0/kernel{suffix}")
        state_dict["paligemma_with_expert.gemma_expert.model.norm.dense.bias"] = final_norm_bias
        state_dict["paligemma_with_expert.gemma_expert.model.norm.dense.weight"] = final_norm_kernel.transpose()
    else:
        state_dict["paligemma_with_expert.gemma_expert.model.norm.weight"] = state_dict.pop(
            f"llm/final_norm_{num_expert}/scale{suffix}"
        )

    final_state_dict = {}
    for key, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            final_state_dict[key] = torch.from_numpy(_to_numpy(value))
        else:
            final_state_dict[key] = value

    return final_state_dict


def slice_initial_orbax_checkpoint(checkpoint_dir: str, restore_precision: str | None = None):
    params = openpi.models.model.restore_params(
        f"{checkpoint_dir}/params/", restore_type=np.ndarray, dtype=restore_precision
    )
    return {"paligemma_params": traversals.flatten_mapping(params["PaliGemma"], sep="/"), "projection_params": params}


def load_jax_model_and_print_keys(checkpoint_dir: str):
    checkpoint_dir = os.path.abspath(checkpoint_dir) if not checkpoint_dir.startswith("gs://") else checkpoint_dir
    checkpointer = ocp.PyTreeCheckpointer()
    metadata = checkpointer.metadata(f"{checkpoint_dir}/params")
    print(utils.array_tree_to_info(metadata))


def convert_pi0_checkpoint(
    checkpoint_dir: str, precision: str, output_path: str, model_config: openpi.models.pi0_config.Pi0Config
):
    print(f"Converting PI0 checkpoint from {checkpoint_dir} to {output_path}")
    print(f"Model config: {model_config}")

    initial_params = slice_initial_orbax_checkpoint(checkpoint_dir=checkpoint_dir, restore_precision="float32")

    if model_config.pi05:
        keys = [
            "action_in_proj",
            "action_out_proj",
            "time_mlp_in",
            "time_mlp_out",
        ]
    else:
        keys = [
            "state_proj",
            "action_in_proj",
            "action_out_proj",
            "action_time_mlp_in",
            "action_time_mlp_out",
        ]

    projection_params = {}
    for key in keys:
        kernel_params = initial_params["projection_params"][key]["kernel"]
        bias_params = initial_params["projection_params"][key]["bias"]
        if isinstance(kernel_params, dict):
            weight = kernel_params["value"]
            bias = bias_params["value"]
        else:
            weight = kernel_params
            bias = bias_params

        projection_params[f"{key}.weight"] = torch.from_numpy(np.array(weight)).T
        projection_params[f"{key}.bias"] = torch.from_numpy(np.array(bias))

    class PaliGemmaConfig:
        def __init__(self, gemma_cfg):
            self.vision_config = type(
                "obj",
                (object,),
                {
                    "hidden_size": 1152,
                    "num_hidden_layers": 27,
                    "num_attention_heads": 16,
                    "intermediate_size": 4304,
                    "patch_size": 14,
                    "projection_dim": 2048,
                },
            )()
            self.text_config = type(
                "obj",
                (object,),
                {
                    "hidden_size": gemma_cfg.width,
                    "num_hidden_layers": gemma_cfg.depth,
                    "num_attention_heads": gemma_cfg.num_heads,
                    "head_dim": gemma_cfg.head_dim,
                    "intermediate_size": gemma_cfg.mlp_dim,
                },
            )()

    paligemma_lora_config = openpi.models.gemma.get_config(model_config.paligemma_variant)
    action_expert_config = openpi.models.gemma.get_config(model_config.action_expert_variant)
    paligemma_config = PaliGemmaConfig(paligemma_lora_config)

    paligemma_params, expert_params = slice_paligemma_state_dict(
        initial_params["paligemma_params"],
        paligemma_config,
        paligemma_lora_config=paligemma_lora_config,
        pi05=model_config.pi05,
    )

    gemma_params = slice_gemma_state_dict(
        expert_params,
        action_expert_config,
        num_expert=1,
        pi05=model_config.pi05,
    )

    pi0_model = openpi.models_pytorch.pi0_pytorch.PI0Pytorch(model_config)
    all_params = {**paligemma_params, **gemma_params, **projection_params}
    incompatible = pi0_model.load_state_dict(all_params, strict=False)

    if incompatible.missing_keys:
        print("Missing keys:")
        for key in incompatible.missing_keys:
            print(f"  - {key}")
    if incompatible.unexpected_keys:
        print("Unexpected keys:")
        for key in incompatible.unexpected_keys:
            print(f"  - {key}")

    if precision == "float32":
        pi0_model = pi0_model.to(torch.float32)
    elif precision == "bfloat16":
        pi0_model = pi0_model.to(torch.bfloat16)
    else:
        raise ValueError(f"Invalid precision: {precision}")

    os.makedirs(output_path, exist_ok=True)
    safetensors.torch.save_model(pi0_model, os.path.join(output_path, "model.safetensors"))

    assets_source = pathlib.Path(checkpoint_dir).parent / "assets"
    if assets_source.exists():
        assets_dest = pathlib.Path(output_path) / "assets"
        if assets_dest.exists():
            shutil.rmtree(assets_dest)
        shutil.copytree(assets_source, assets_dest)

    config_dict = {
        "action_dim": model_config.action_dim,
        "action_horizon": model_config.action_horizon,
        "paligemma_variant": model_config.paligemma_variant,
        "action_expert_variant": model_config.action_expert_variant,
        "precision": precision,
        "lora_merged": True,
    }
    with open(os.path.join(output_path, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)

    print("Model conversion completed successfully!")
    print(f"Model saved to {output_path}")


def main(
    checkpoint_dir: str,
    config_name: str,
    output_path: str | None = None,
    precision: Literal["float32", "bfloat16", "float16"] = "bfloat16",
    *,
    inspect_only: bool = False,
):
    model_config = _config.get_config(config_name).model
    if not isinstance(model_config, openpi.models.pi0_config.Pi0Config):
        raise ValueError(f"Config {config_name} is not a Pi0Config")
    if inspect_only:
        load_jax_model_and_print_keys(checkpoint_dir)
    else:
        if not output_path:
            print("Error: --output_path is required for conversion. Use --inspect_only to only view keys.")
            return
        convert_pi0_checkpoint(checkpoint_dir, precision, output_path, model_config)


if __name__ == "__main__":
    tyro.cli(main)
