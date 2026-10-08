import numpy as np

from code.auc_metrics import decompose_pooled_auc, nested_within_auc, verify_identity
from code.nonidentification import restratify


def test_exact_decomposition():
    score = np.array([0.1, 0.8, 0.2, 0.7, 0.3, 0.9])
    y = np.array([0, 1, 0, 1, 1, 0])
    learner = np.array(["a", "a", "b", "b", "c", "c"])
    d = decompose_pooled_auc(score, y, learner)
    assert verify_identity(d)
    assert d.total_pairs == 9
    assert d.within_pairs == 3


def test_nonidentification_preserves_within_auc():
    score = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7])
    y = np.array([0, 1, 1, 0, 0, 1])
    learner = np.array([0, 0, 1, 1, 2, 2])
    before = decompose_pooled_auc(score, y, learner)
    shifted = restratify(score, learner, {0: 10.0, 1: -10.0, 2: 0.0})
    after = decompose_pooled_auc(shifted, y, learner)
    assert np.isclose(before.auc_within, after.auc_within)
    assert not np.isclose(before.auc_pooled, after.auc_pooled)


def test_nested_pair_weight():
    score = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7])
    y = np.array([0, 1, 0, 1, 0, 1])
    learner = np.array([0, 0, 1, 1, 2, 2])
    kc = np.array(["x", "x", "x", "y", "z", "z"])
    out = nested_within_auc(score, y, learner, kc)
    assert out["pairs"] == 2
    assert 0.0 < out["pair_weight"] < 1.0


if __name__ == "__main__":
    test_exact_decomposition()
    test_nonidentification_preserves_within_auc()
    test_nested_pair_weight()
    print("ALL CORE CHECKS PASSED")
