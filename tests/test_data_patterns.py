import torch

from dsv4_kernel_bench.data import make_sparse_attention_inputs


def test_swa_pattern_uses_raw_window_ids_only():
    inputs = make_sparse_attention_inputs(
        batch=1,
        seqlen_q=5,
        seqlen_kv=5,
        heads=1,
        dim=8,
        topk=3,
        selection_pattern="swa",
        device="cpu",
    )

    expected = torch.tensor(
        [[[0, -1, -1], [0, 1, -1], [0, 1, 2], [1, 2, 3], [2, 3, 4]]],
        dtype=torch.int32,
    )
    torch.testing.assert_close(inputs.topk_idxs.cpu(), expected)
    assert inputs.kv.shape[1] == 5
    assert inputs.metadata["n_compressed"] == 0


def test_csa_pattern_appends_shifted_compressed_ids():
    inputs = make_sparse_attention_inputs(
        batch=1,
        seqlen_q=8,
        seqlen_kv=8,
        heads=1,
        dim=8,
        topk=4,
        selection_pattern="csa",
        window_size=2,
        compressed_topk=2,
        device="cpu",
    )

    # Raw KV length is 8, so compressed C0/C1 live at logical ids 8/9.
    assert inputs.kv.shape[1] == 10
    assert inputs.metadata["n_compressed"] == 2
    torch.testing.assert_close(inputs.topk_idxs[0, 3], torch.tensor([2, 3, 8, -1]))
    torch.testing.assert_close(inputs.topk_idxs[0, 7], torch.tensor([6, 7, 8, 9]))


def test_hca_pattern_uses_all_visible_compressed_ids():
    inputs = make_sparse_attention_inputs(
        batch=1,
        seqlen_q=256,
        seqlen_kv=256,
        heads=1,
        dim=8,
        topk=1,
        selection_pattern="hca",
        window_size=1,
        device="cpu",
    )

    assert inputs.kv.shape[1] == 258
    assert inputs.metadata["n_compressed"] == 2
    torch.testing.assert_close(inputs.topk_idxs[0, 127], torch.tensor([127, 256, -1]))
    torch.testing.assert_close(inputs.topk_idxs[0, 255], torch.tensor([255, 256, 257]))
