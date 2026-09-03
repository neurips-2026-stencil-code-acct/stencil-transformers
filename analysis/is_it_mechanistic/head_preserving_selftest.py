"""Deterministic checks for head_preserving_substitution.py."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from head_preserving_substitution import (
    CheckedAttentionControl,
    assert_distinct_split_paths,
    load_split_pairs,
    validate_mean_attention,
    verify_patched_noop,
)


def test_per_head_substitution_is_not_head_averaged():
    import torch

    # Head 0 attends to the same token; head 1 attends to the other token.
    per_head = np.array(
        [
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[0.0, 1.0], [1.0, 0.0]],
            ]
        ],
        dtype=np.float32,
    )
    values = torch.tensor(
        [[[[1.0], [3.0]], [[10.0], [30.0]]]], dtype=torch.float32
    )
    unused_attention = torch.full((1, 2, 2, 2), 0.5)

    control = CheckedAttentionControl(substitute=per_head)
    actual = control._apply(0, unused_attention, values)
    expected = torch.tensor(
        [[[[1.0], [3.0]], [[30.0], [10.0]]]], dtype=torch.float32
    )
    layer_average = torch.tensor(per_head.mean(axis=1))
    accidentally_averaged = layer_average.expand(1, 2, 2, 2) @ values

    assert torch.equal(actual, expected)
    assert not torch.equal(actual, accidentally_averaged)
    assert control.n_substituted == 1


def test_validation_and_test_cannot_be_the_same_file():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        validation = root / "val.npy"
        test = root / "test.npy"
        array = np.zeros((2, 3, 4), dtype=np.float32)
        np.save(validation, array)
        np.save(test, array + 1)

        assert_distinct_split_paths(validation, test)
        try:
            assert_distinct_split_paths(validation, validation)
        except ValueError:
            pass
        else:
            raise AssertionError("same-file validation/test reuse was not rejected")


def test_split_sampler_is_deterministic_and_keeps_splits_distinct():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        validation = np.arange(3 * 5 * 4, dtype=np.float32).reshape(3, 5, 4)
        test = validation + 10_000
        validation_path = root / "val.npy"
        test_path = root / "test.npy"
        np.save(validation_path, validation)
        np.save(test_path, test)

        x1, y1 = load_split_pairs(validation_path, 6, seed=9)
        x2, y2 = load_split_pairs(validation_path, 6, seed=9)
        test_x, test_y = load_split_pairs(test_path, 6, seed=9)
        assert np.array_equal(x1, x2)
        assert np.array_equal(y1, y2)
        assert not np.array_equal(x1, test_x)
        assert not np.array_equal(y1, test_y)


def test_mean_shape_and_rows_are_checked():
    valid = np.full((2, 3, 4, 4), 0.25, dtype=np.float64)
    validate_mean_attention(valid, n_space=4)
    invalid = valid.copy()
    invalid[0, 0, 0, 0] = 0.5
    try:
        validate_mean_attention(invalid, n_space=4)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid attention row sums were not rejected")


def test_patched_noop_matches_a_small_transformer():
    import torch
    import torch.nn as nn

    torch.manual_seed(7)

    class SmallModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.input = nn.Linear(1, 8)
            layer = nn.TransformerEncoderLayer(
                d_model=8,
                nhead=2,
                dim_feedforward=16,
                dropout=0.0,
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=2)
            self.output = nn.Linear(8, 1)

        def forward(self, x):
            hidden = self.input(x.unsqueeze(-1))
            return self.output(self.encoder(hidden)).squeeze(-1)

    model = SmallModel().eval()
    x = np.random.default_rng(4).normal(size=(6, 1, 5)).astype(np.float32)
    error = verify_patched_noop(model, x, "cpu", batch_size=6)
    assert error <= 1e-5


def main():
    tests = [
        test_per_head_substitution_is_not_head_averaged,
        test_validation_and_test_cannot_be_the_same_file,
        test_split_sampler_is_deterministic_and_keeps_splits_distinct,
        test_mean_shape_and_rows_are_checked,
        test_patched_noop_matches_a_small_transformer,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"PASS all {len(tests)} head-preserving substitution checks")


if __name__ == "__main__":
    main()
