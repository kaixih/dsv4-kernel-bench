import torch

from dsv4_kernel_bench.indexing import (
    flatten_kv_sbd,
    flatten_q_sbhd,
    local_topk_to_global_flat,
    unflatten_output_bshd,
)


def test_local_topk_to_global_flat_preserves_invalid_indices():
    topk = torch.tensor(
        [
            [[0, 2, -1], [1, -1, 3]],
            [[1, 0, -1], [3, 2, -1]],
        ],
        dtype=torch.int32,
    )

    out = local_topk_to_global_flat(topk, seqlen_kv=4)

    expected = torch.tensor(
        [
            [0, 4, -1],
            [3, 1, -1],
            [2, -1, 6],
            [7, 5, -1],
        ],
        dtype=torch.int32,
    )
    torch.testing.assert_close(out, expected)


def test_flatten_and_unflatten_roundtrip():
    q = torch.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5)
    flat = flatten_q_sbhd(q)
    restored = unflatten_output_bshd(flat, batch=2, seqlen_q=3)
    torch.testing.assert_close(restored, q)


def test_flatten_kv_sbd_order():
    kv = torch.arange(2 * 3 * 5).reshape(2, 3, 5)
    flat = flatten_kv_sbd(kv)
    expected = torch.stack([kv[0, 0], kv[1, 0], kv[0, 1], kv[1, 1], kv[0, 2], kv[1, 2]])
    torch.testing.assert_close(flat, expected)
