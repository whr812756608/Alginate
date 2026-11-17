"""Smoke tests for the publication structure; not scientific reproduction tests."""
import numpy as np
import pytest
from alginate_ms2.config import load_models
from alginate_ms2.validation import grouped_loo_splits, mae_percentage_points

def test_models():
    models = load_models()["models"]
    assert len(models) == 5
    assert all(set(x["loo_mae_pp_by_nce"]) == {"18", "22", "26"} for x in models.values())
    assert len(models["3"]["ratios"]) == 8
    assert len(models["4"]["ratios"]) == 8

def test_grouped_loo_never_splits_composition():
    groups = np.repeat([0,20,40,50,60,80,100],3)
    seen = set()
    for held, train, test in grouped_loo_splits(groups):
        assert not (set(train) & set(test))
        assert set(groups[test]) == {held}
        assert held not in set(groups[train])
        seen.update(test.tolist())
    assert seen == set(range(21))

def test_bad_groups_rejected():
    with pytest.raises(ValueError):
        list(grouped_loo_splits([0,20,40]))

def test_mae():
    assert mae_percentage_points([10,20],[9,22]) == 1.5
