#!/usr/bin/env python3
"""Launch the upstream LingBot-VA ``wan_va_server.py`` with a prompt-padding override.

Upstream ``VA_Server._get_t5_prompt_embeds`` tokenizes with
``padding="max_length"`` (512 tokens) and runs the 11 GB T5 text encoder on CPU
when offload is enabled.  Embeddings are then truncated to the real token count
and zero padded, so padding only changes compute: on 4090-hexa-node2 one reset
took ~6.4 min with max_length vs ~1.3 s with ``longest``.

This shim leaves the upstream file untouched.  It imports it as a module, wraps
``VA_Server.__init__`` so that ``self.tokenizer`` rewrites ``padding="max_length"``
to ``SIMUGUARD_PROMPT_PADDING`` (default ``longest``), and calls upstream ``main()``.
Numerical agreement of the two paddings is measured separately by
``scripts/robotwin/check_prompt_padding.py``.

With offload enabled upstream also keeps the VAE on CPU and encodes every
observation chunk there (server at ~400% CPU, GPU idle, >14 min for the first
chunk on node2).  ``SIMUGUARD_VAE_DEVICE=gpu_staged`` stages the VAE onto the
policy GPU around ``_encode_obs`` (parking the KV cache on CPU to fit 24 GB),
mirroring the patch used by the LingBot reproduction runs.  Both overrides change
only *where/how much* is computed, never the model or its weights, but CPU and
GPU kernels are not bit-identical.

Environment:
    SIMUGUARD_WAN_VA_SERVER   path to upstream wan_va/wan_va_server.py (required)
    SIMUGUARD_PROMPT_PADDING  replacement padding mode (default "longest"); "max_length" keeps upstream
    SIMUGUARD_VAE_DEVICE      "gpu_staged" (default) or "upstream"
    SIMUGUARD_ATTN_WINDOW     override the config's temporal attention window (optional)
    SIMUGUARD_MODEL_PATH      checkpoint dir for configs whose path is an upstream placeholder
    SIMUGUARD_ENABLE_OFFLOAD  "1"/"0" to force VAE+text-encoder offload (24 GB cards need "1")

``SIMUGUARD_ATTN_WINDOW`` is *not* a performance-only override: it shortens the
temporal context the policy attends to and can change its actions.  Upstream
``robotwin30_train`` uses 72, whose KV cache is 10.1 GiB and does not fit a 24 GB
card next to the 11 GB transformer; 48 gives a 6.7 GiB cache.  Any run that sets
it must report it.

Run exactly like the upstream script, e.g. through ``torch.distributed.run``.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from typing import Any


class PaddingOverrideTokenizer:
    def __init__(self, tokenizer: Any, padding: str) -> None:
        object.__setattr__(self, "_tokenizer", tokenizer)
        object.__setattr__(self, "_padding", padding)
        object.__setattr__(self, "override_count", 0)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if kwargs.get("padding") == "max_length":
            kwargs["padding"] = self._padding
            object.__setattr__(self, "override_count", self.override_count + 1)
        return self._tokenizer(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._tokenizer, name)


def move_kv_cache(server: Any, device: Any) -> int:
    """Move the transformer's attention KV cache; pure data movement, values unchanged."""

    import torch

    target = torch.device(device)
    moved = 0
    for block in getattr(server.transformer, "blocks", []):
        caches = getattr(getattr(block, "attn1", None), "attn_caches", None) or {}
        cache = caches.get(server.cache_name)
        if not cache:
            continue
        for key, value in cache.items():
            if torch.is_tensor(value) and value.device != target:
                moved += value.numel() * value.element_size()
                cache[key] = value.to(target)
    return moved


def stage_vaes(server: Any, device: Any) -> list[Any]:
    """Return the VAE modules used for observation encoding."""

    vaes = []
    for name in ("streaming_vae", "streaming_vae_half"):
        wrapper = getattr(server, name, None)
        vae = getattr(wrapper, "vae", None)
        if vae is not None:
            vaes.append(vae)
    return vaes


def install_staged_vae(module: Any, server_cls: Any) -> None:
    """Run ``_encode_obs`` with the VAE on the policy GPU and the KV cache parked on CPU.

    Upstream reads ``next(vae.parameters()).device`` *inside* ``_encode_obs`` and
    moves the observation tensors there, so the VAE must already be on the GPU
    when that method starts -- moving it inside ``encode_chunk`` makes the input
    stay on CPU and conv3d fails with mixed devices.
    """

    original_encode_obs = server_cls._encode_obs

    def _encode_obs(self: Any, obs: Any) -> Any:
        import torch

        if not getattr(self, "enable_offload", False):
            return original_encode_obs(self, obs)
        parked = move_kv_cache(self, "cpu")
        torch.cuda.empty_cache()
        vaes = stage_vaes(self, self.device)
        for vae in vaes:
            vae.to(self.device)
        if not getattr(self, "_simuguard_logged_staging", False):
            module.logger.info(
                f"[simuguard] staged {len(vaes)} VAE module(s) onto {self.device}; "
                f"KV cache parked on CPU: {parked / 1024 ** 3:.2f} GiB"
            )
            self._simuguard_logged_staging = True
        try:
            return original_encode_obs(self, obs)
        finally:
            for vae in vaes:
                vae.to("cpu")
            torch.cuda.empty_cache()
            if parked:
                move_kv_cache(self, self.device)

    server_cls._encode_obs = _encode_obs


def apply_config_overrides(module: Any, config_name: str) -> dict[str, Any]:
    """Apply runtime config overrides that do not change model behaviour.

    ``robotwin`` (the config matching the RoboTwin posttrain checkpoint) ships an
    upstream placeholder checkpoint path and ``enable_offload=False``, which does
    not fit a 24 GB card.  Both are environment facts, not model settings.
    """

    config = module.VA_CONFIGS[config_name]
    applied: dict[str, Any] = {"config": config_name}
    model_path = os.environ.get("SIMUGUARD_MODEL_PATH")
    if model_path and not os.path.isdir(str(config.wan22_pretrained_model_name_or_path)):
        applied["model_path"] = {"before": config.wan22_pretrained_model_name_or_path, "after": model_path}
        config.wan22_pretrained_model_name_or_path = model_path
    offload = os.environ.get("SIMUGUARD_ENABLE_OFFLOAD")
    if offload is not None:
        value = offload not in ("0", "false", "False", "")
        applied["enable_offload"] = {"before": getattr(config, "enable_offload", None), "after": value}
        config.enable_offload = value
    return applied


def set_attention_window(module: Any, config_name: str, window: int) -> dict[str, Any]:
    config = module.VA_CONFIGS[config_name]
    before = int(config.attn_window)
    config.attn_window = int(window)
    return {"config": config_name, "attn_window_before": before, "attn_window_after": int(window)}


def config_name_from_argv(argv: list[str], default: str = "robotwin") -> str:
    for index, item in enumerate(argv):
        if item == "--config-name" and index + 1 < len(argv):
            return argv[index + 1]
        if item.startswith("--config-name="):
            return item.split("=", 1)[1]
    return default


def load_upstream(server_path: str) -> Any:
    server_path = os.path.abspath(server_path)
    server_dir = os.path.dirname(server_path)
    if server_dir not in sys.path:
        sys.path.insert(0, server_dir)
    spec = importlib.util.spec_from_file_location("wan_va_server_upstream", server_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)  # defines classes; upstream main() only runs under __main__
    return module


def install_overrides(module: Any, padding: str, vae_mode: str) -> None:
    server_cls = module.VA_Server
    if getattr(server_cls, "_simuguard_padding", None) is not None:
        return
    original_init = server_cls.__init__

    def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        if padding != "max_length":
            self.tokenizer = PaddingOverrideTokenizer(self.tokenizer, padding)
            module.logger.info(f"[simuguard] prompt tokenizer padding override: max_length -> {padding}")

    server_cls.__init__ = __init__
    server_cls._simuguard_padding = padding
    server_cls._simuguard_vae_mode = vae_mode
    if vae_mode == "gpu_staged":
        install_staged_vae(module, server_cls)


def main() -> None:
    server_path = os.environ.get("SIMUGUARD_WAN_VA_SERVER")
    if not server_path:
        raise SystemExit("SIMUGUARD_WAN_VA_SERVER must point to upstream wan_va/wan_va_server.py")
    padding = os.environ.get("SIMUGUARD_PROMPT_PADDING", "longest")
    vae_mode = os.environ.get("SIMUGUARD_VAE_DEVICE", "gpu_staged")
    if vae_mode not in ("gpu_staged", "upstream"):
        raise SystemExit(f"SIMUGUARD_VAE_DEVICE must be gpu_staged or upstream, got {vae_mode!r}")
    module = load_upstream(server_path)
    install_overrides(module, padding, vae_mode)
    module.init_logger()
    config_name = config_name_from_argv(sys.argv)
    overrides = apply_config_overrides(module, config_name)
    if len(overrides) > 1:
        module.logger.info(f"[simuguard] config overrides (environment only): {overrides}")
    window = os.environ.get("SIMUGUARD_ATTN_WINDOW")
    if window:
        change = set_attention_window(module, config_name, int(window))
        module.logger.info(f"[simuguard] attention window override (changes policy behaviour): {change}")
    module.main()


if __name__ == "__main__":
    main()
