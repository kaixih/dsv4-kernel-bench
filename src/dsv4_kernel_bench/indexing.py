from __future__ import annotations

import torch


def local_topk_to_global_flat(topk_idxs: torch.Tensor, seqlen_kv: int) -> torch.Tensor:
    """Convert batch-local top-k indices to NVIDIA DSA flat-global indices.

    Canonical top-k indices are shaped ``[B, S, TopK]`` and store local KV row
    indices in ``[0, S_kv)``. ``-1`` marks invalid positions.

    NVIDIA DSA/FlashMLA uses flattened SB row order:
    ``row = s * B + b`` and ``global_kv = local_kv * B + b``.
    The returned tensor is shaped ``[S * B, TopK]``.
    """

    if topk_idxs.dim() != 3:
        raise ValueError(f"topk_idxs must have shape [B, S, TopK], got {tuple(topk_idxs.shape)}")
    if not torch.is_floating_point(topk_idxs) and topk_idxs.dtype not in (
        torch.int16,
        torch.int32,
        torch.int64,
    ):
        raise TypeError(f"topk_idxs must be integer typed, got {topk_idxs.dtype}")

    batch, seqlen_q, topk = topk_idxs.shape
    idxs_sb = topk_idxs.permute(1, 0, 2).reshape(seqlen_q * batch, topk)
    valid = idxs_sb >= 0

    if valid.any():
        invalid_high = idxs_sb[valid] >= seqlen_kv
        if invalid_high.any():
            bad = idxs_sb[valid][invalid_high][0].item()
            raise ValueError(f"topk index {bad} exceeds S_kv={seqlen_kv}")

    batch_ids = torch.arange(seqlen_q * batch, device=topk_idxs.device, dtype=idxs_sb.dtype) % batch
    batch_ids = batch_ids.unsqueeze(1).expand_as(idxs_sb)
    global_idxs = torch.where(valid, idxs_sb * batch + batch_ids, idxs_sb)
    return global_idxs.to(torch.int32).contiguous()


def flatten_q_sbhd(q: torch.Tensor) -> torch.Tensor:
    """Convert canonical ``[B, S, H, D]`` to DSA ``[S * B, H, D]``."""

    if q.dim() != 4:
        raise ValueError(f"q must have shape [B, S, H, D], got {tuple(q.shape)}")
    batch, seqlen_q, heads, dim = q.shape
    return q.permute(1, 0, 2, 3).reshape(seqlen_q * batch, heads, dim).contiguous()


def to_dsa_query_sbhd(q: torch.Tensor) -> torch.Tensor:
    """Convert canonical ``[B, S, H, D]`` to DSA public ``[S, B, H, D]``."""

    if q.dim() != 4:
        raise ValueError(f"q must have shape [B, S, H, D], got {tuple(q.shape)}")
    return q.permute(1, 0, 2, 3).contiguous()


def flatten_kv_sbd(kv: torch.Tensor) -> torch.Tensor:
    """Convert canonical ``[B, S_kv, D]`` to DSA ``[S_kv * B, D]``."""

    if kv.dim() != 3:
        raise ValueError(f"kv must have shape [B, S_kv, D], got {tuple(kv.shape)}")
    batch, seqlen_kv, dim = kv.shape
    return kv.permute(1, 0, 2).reshape(seqlen_kv * batch, dim).contiguous()


def to_dsa_kv_sbd(kv: torch.Tensor) -> torch.Tensor:
    """Convert canonical ``[B, S_kv, D]`` to DSA public ``[S_kv, B, D]``."""

    if kv.dim() != 3:
        raise ValueError(f"kv must have shape [B, S_kv, D], got {tuple(kv.shape)}")
    return kv.permute(1, 0, 2).contiguous()


def unflatten_output_bshd(out_flat: torch.Tensor, batch: int, seqlen_q: int) -> torch.Tensor:
    """Convert DSA ``[S * B, H, D]`` output back to ``[B, S, H, D]``."""

    if out_flat.dim() != 3:
        raise ValueError(f"out_flat must have shape [S * B, H, D], got {tuple(out_flat.shape)}")
    _, heads, dim = out_flat.shape
    return out_flat.reshape(seqlen_q, batch, heads, dim).permute(1, 0, 2, 3).contiguous()


def from_dsa_output_sbhd(out: torch.Tensor, batch: int, heads: int) -> torch.Tensor:
    """Convert DSA public output ``[S, B, H * D]`` back to ``[B, S, H, D]``."""

    if out.dim() != 3:
        raise ValueError(f"out must have shape [S, B, H * D], got {tuple(out.shape)}")
    seqlen_q, out_batch, hidden = out.shape
    if out_batch != batch:
        raise ValueError(f"output batch {out_batch} does not match input batch {batch}")
    if hidden % heads != 0:
        raise ValueError(f"output hidden size {hidden} is not divisible by heads={heads}")
    dim = hidden // heads
    return out.reshape(seqlen_q, batch, heads, dim).permute(1, 0, 2, 3).contiguous()
