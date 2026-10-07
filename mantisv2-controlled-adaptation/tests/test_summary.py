import numpy as np
import pandas as pd

from mantis_adaptation.summarize import (
    bootstrap_mean_ci,
    make_main_table,
    make_paired_effects,
    wtl,
)


def sample_table():
    return pd.DataFrame(
        {
            "dataset": ["a", "b", "c"],
            "frozen": [0.70, 0.80, 0.90],
            "full": [0.72, 0.79, 0.90],
            "scratch": [0.60, 0.82, 0.88],
            "delta_full_frozen": [0.02, -0.01, 0.0],
            "delta_full_scratch": [0.12, -0.03, 0.02],
        }
    )


def test_wtl_sign_comparison():
    assert wtl(pd.Series([0.1, 0.0, -0.2])) == (1, 1, 1)


def test_bootstrap_is_reproducible():
    first = bootstrap_mean_ci([1.0, 2.0, 3.0], 1000, 7)
    second = bootstrap_mean_ci([1.0, 2.0, 3.0], 1000, 7)
    assert first == second
    assert np.isclose(first[0], 2.0)


def test_main_table_and_paired_effects():
    table = make_main_table(sample_table())
    full = table.loc[table["mode"] == "full"].iloc[0]
    assert (full["wins_vs_frozen"], full["ties_vs_frozen"], full["losses_vs_frozen"]) == (
        1,
        1,
        1,
    )
    paired = make_paired_effects(sample_table(), repetitions=1000, seed=3)
    assert set(paired["contrast"]) == {"full-frozen", "full-scratch"}

