"""Import-only compatibility shim for LingBot-VA on environments without flash-attn.

Upstream wan_va/modules/model.py imports flash_attn unconditionally, while the
inference server constructs the transformer with attn_mode="torch".  Any real
call fails loudly so a silent fallback can never change model behaviour.
"""


def flash_attn_func(*args, **kwargs):
    raise RuntimeError("flash_attn is not installed; SimuGuard compat shim was called (attn_mode must be torch)")
