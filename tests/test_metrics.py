from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import numpy as np
import pytest

from pie.metrics import (
    DE_PROB_THRESHOLD,
    METRIC_KEYS,
    PAIR_METRIC_KEYS,
    ScoringInputs,
    discrimination_score_l1,
    score,
)

FIXTURES = Path(__file__).parent / "fixtures" / "metrics"
DE_KEYS = ("binary_auprc", "sig_jaccard", "direction_match", "spearman_nsig", "spearman_lfc")
ROW_KEYS = ("binary_auprc", "sig_jaccard", "direction_match", "spearman_lfc")
ARRAYS = (
    "p_de",
    "lfc_pred",
    "delta_p_pred",
    "de_true",
    "tested",
    "lfc_true",
    "delta_p_true",
    "ctrl_means",
)
T, F = True, False
NAN = math.nan


def _load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


ORACLE = _load("de_metrics_oracle.json")
L1_PARITY = _load("l1_parity.json")


def _same(actual: float, expected: float) -> bool:
    """Exact equality, with NaN equal to NaN."""
    if math.isnan(expected):
        return math.isnan(actual)
    return actual == expected


def _f32(value: float) -> float:
    return float(np.float32(value))


def _make(
    dataset: str,
    genes: list[str],
    contexts: list[str],
    perts: list[str],
    **arrays: object,
) -> ScoringInputs:
    """ScoringInputs with neutral defaults for every array that is not given."""
    shape = (len(perts), len(genes))
    values = {
        "p_de": np.zeros(shape, np.float32),
        "lfc_pred": np.zeros(shape, np.float32),
        "delta_p_pred": np.zeros(shape, np.float32),
        "de_true": np.zeros(shape, bool),
        "tested": np.ones(shape, bool),
        "lfc_true": np.zeros(shape, np.float64),
        "delta_p_true": np.zeros(shape, np.float32),
        "ctrl_means": np.ones(shape, np.float32),
    }
    for name, value in arrays.items():
        values[name] = np.asarray(value, dtype=values[name].dtype).reshape(shape)
    return ScoringInputs(
        dataset=dataset, genes=list(genes), contexts=list(contexts), perts=list(perts), **values
    )


def _take(inp: ScoringInputs, idx: np.ndarray) -> ScoringInputs:
    return ScoringInputs(
        dataset=inp.dataset,
        genes=inp.genes,
        contexts=[inp.contexts[i] for i in idx],
        perts=[inp.perts[i] for i in idx],
        **{name: getattr(inp, name)[idx] for name in ARRAYS},
    )


def _dataset(ds: dict) -> ScoringInputs:
    return _make(
        ds["dataset"],
        ds["genes"],
        ds["contexts"],
        ds["perts"],
        p_de=ds["p_de"],
        lfc_pred=ds["lfc_pred"],
        de_true=ds["de_true"],
        tested=ds["tested"],
        lfc_true=ds["lfc_true"],
    )


def _l1(
    genes: list[str],
    perts: list[str],
    ctrl: list[float],
    true: list[list[float]],
    pred: list[list[float]],
) -> ScoringInputs:
    n_rows = len(perts)
    return _make(
        "alpha",
        genes,
        ["c1"] * n_rows,
        perts,
        delta_p_true=true,
        delta_p_pred=pred,
        ctrl_means=np.tile(np.asarray(ctrl, np.float32), (n_rows, 1)),
    )


def test_metric_keys_and_threshold() -> None:
    assert METRIC_KEYS == (
        "binary_auprc",
        "sig_jaccard",
        "direction_match",
        "spearman_nsig",
        "spearman_lfc",
        "discrimination_score_l1",
    )
    assert PAIR_METRIC_KEYS == (
        "binary_auprc",
        "sig_jaccard",
        "direction_match",
        "spearman_lfc",
        "discrimination_score_l1",
    )
    assert DE_PROB_THRESHOLD == 0.5


# --- parity with the original implementations (fixtures from the original code) ---


@pytest.mark.parametrize("case", ORACLE["oracle_cases"], ids=lambda case: case["name"])
def test_oracle_cases_match_original(case: dict) -> None:
    n_rows, n_genes = len(case["p_de"]), len(case["p_de"][0])
    inputs = _make(
        "alpha",
        [f"g{j}" for j in range(n_genes)],
        ["c1"] * n_rows,
        [f"g{i}" for i in range(n_rows)],
        p_de=case["p_de"],
        lfc_pred=case["lfc_pred"],
        de_true=case["de_true"],
        tested=case["tested"],
        lfc_true=case["lfc_true"],
    )
    result = score([inputs])
    context = result.per_context.iloc[0]
    for key, expected in case["expected"].items():
        assert _same(float(context[key]), expected), key
        assert _same(result.aggregate[key], expected), key


@pytest.mark.parametrize("case", ORACLE["scorer_cases"], ids=lambda case: case["name"])
def test_scorer_cases_match_original(case: dict) -> None:
    result = score([_dataset(ds) for ds in case["datasets"]])
    expected = case["expected"]
    contexts = result.per_context.set_index(["dataset", "context"])
    assert len(contexts) == len(expected["per_context"])
    for row in expected["per_context"]:
        got = contexts.loc[(row["dataset"], row["context"])]
        assert int(got["n_rows"]) == row["n_rows"]
        for key in DE_KEYS:
            assert _same(float(got[key]), row[key]), (row["dataset"], row["context"], key)
    pairs = result.per_pair.set_index(["dataset", "context", "perturbation"])
    assert len(pairs) == len(expected["per_pair"])
    for row in expected["per_pair"]:
        got = pairs.loc[(row["dataset"], row["context"], row["perturbation"])]
        assert int(got["n_rows"]) == row["n_rows"]
        for key in ROW_KEYS:
            assert _same(float(got[key]), row[key]), (row["context"], row["perturbation"], key)
    for key in DE_KEYS:
        assert _same(result.aggregate[key], expected["aggregate"][key]), key
        assert _same(result.context_mean[key], expected["context_mean"][key]), key


@pytest.mark.parametrize("case", L1_PARITY["cases"], ids=lambda case: case["name"])
def test_l1_cases_match_reference(case: dict) -> None:
    n_rows = len(case["perts"])
    inputs = _make(
        "alpha",
        case["genes"],
        ["c1"] * n_rows,
        case["perts"],
        delta_p_pred=case["delta_p_pred"],
        delta_p_true=case["delta_p_true"],
        ctrl_means=np.tile(np.asarray(case["ctrl"], np.float32), (n_rows, 1)),
    )
    result = score([inputs])
    per_pert = result.per_pair.set_index("perturbation")["discrimination_score_l1"]
    expected = case["expected"]["per_pert"]
    for pert in sorted(set(case["perts"])):
        assert _same(float(per_pert[pert]), expected.get(pert, NAN)), pert
    context = float(result.per_context["discrimination_score_l1"].iloc[0])
    assert _same(context, case["expected"]["context_mean"])


# --- discrimination_score_l1 directly ---


def test_discrimination_score_l1_ties_between_other_perts() -> None:
    ctrl = np.full(3, 10.0)
    real = ctrl + np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0], [5, 5, 5]], dtype=float)
    pred = ctrl + np.array([[0, 0, 0], [3, 0, 0], [0, 3, 0], [1, 1, 0]], dtype=float)
    got = discrimination_score_l1(real, pred, ctrl, ["a", "b", "c", "d"], ["x1", "x2", "x3"])
    np.testing.assert_array_equal(got, [1.0, 1.0, 1.0, 0.25])


def test_discrimination_score_l1_excludes_own_gene_by_name() -> None:
    ctrl = np.full(3, 10.0)
    real = ctrl + np.array([[10, 0, 0], [0, 1, 0]], dtype=float)
    pred = ctrl + np.array([[10, 1, 0], [0, 1, 0]], dtype=float)
    excluded = discrimination_score_l1(real, pred, ctrl, ["a", "b"], ["a", "x1", "x2"])
    np.testing.assert_array_equal(excluded, [0.5, 1.0])
    kept = discrimination_score_l1(real, pred, ctrl, ["a", "b"], ["y", "x1", "x2"])
    np.testing.assert_array_equal(kept, [1.0, 1.0])


def test_discrimination_score_l1_tie_with_own_pert_follows_default_argsort() -> None:
    ctrl = np.zeros(2)
    real = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 3.0]])
    pred = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 3.0]])
    got = discrimination_score_l1(real, pred, ctrl, ["a", "b", "c"], ["x1", "x2"])
    for i in range(3):
        distances = np.abs(real - pred[i]).sum(axis=1)
        rank = int(np.flatnonzero(np.argsort(distances) == i)[0])
        assert got[i] == 1 - rank / 3
    assert got[2] == 1.0


# --- row-level edge-case policies ---


def test_binary_auprc_policies() -> None:
    perts = ["p0", "p1", "p2", "p3", "p4", "p5"]
    p_de = [
        [0.9, 0.1, 0.2],
        [0.9, 0.1, 0.2],
        [0.9, 0.1, 0.2],
        [0.9, 0.1, 0.2],
        [0.9, 0.95, 0.1],
        [0.2, 0.9, 0.1],
    ]
    de_true = [[T, F, F], [F, F, F], [T, T, T], [T, F, F], [T, F, F], [T, T, F]]
    tested = [[T, T, T], [T, T, T], [T, T, T], [F, F, F], [T, F, T], [T, T, T]]
    lfc_true = [[1, 0, 0], [0, 0, 0], [1, 1, 1], [1, 0, 0], [1, 0, 0], [1, NAN, 0]]
    inputs = _make(
        "alpha",
        ["x0", "x1", "x2"],
        ["c1"] * 6,
        perts,
        p_de=p_de,
        de_true=de_true,
        tested=tested,
        lfc_true=lfc_true,
    )
    got = score([inputs]).per_pair.set_index("perturbation")["binary_auprc"]
    assert got["p0"] == 1.0
    assert got["p1"] == 0.0  # no positive: 0.0, row kept
    assert got["p2"] == 0.0  # only positives: 0.0, row kept
    assert got["p3"] == 0.0  # no tested gene: 0.0, row kept
    assert got["p4"] == 1.0  # the untested high score is ignored
    assert got["p5"] == 0.5  # a DE gene with NaN truth counts as a negative


def test_sig_jaccard_policies() -> None:
    genes = ["g0", "g1", "g2", "g3"]
    perts = ["a_threshold", "b_empty", "g0", "drug_x", "e_masked", "f_untested"]
    p_de = [
        [0.5, 0.49, 0, 0],
        [0, 0, 0, 0],
        [0.9, 0.9, 0, 0],
        [0.9, 0.9, 0, 0],
        [0.9, 0.9, 0.9, 0],
        [0.9, 0.9, 0, 0],
    ]
    de_true = [
        [T, T, F, F],
        [F, F, F, F],
        [T, F, F, F],
        [T, F, F, F],
        [T, T, F, F],
        [T, T, F, F],
    ]
    tested = [
        [T, T, T, T],
        [T, T, T, T],
        [T, T, T, T],
        [T, T, T, T],
        [T, T, T, T],
        [T, F, T, T],
    ]
    lfc_pred = [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, NAN, 0, 0],
        [0, 0, 0, 0],
    ]
    inputs = _make(
        "alpha",
        genes,
        ["c1"] * 6,
        perts,
        p_de=p_de,
        de_true=de_true,
        tested=tested,
        lfc_pred=lfc_pred,
    )
    got = score([inputs]).per_pair.set_index("perturbation")["sig_jaccard"]
    assert got["a_threshold"] == 0.5  # P(DE) == 0.5 is significant (inclusive)
    assert got["b_empty"] == 1.0  # empty union
    assert got["g0"] == 0.0  # the target gene g0 is removed from both sets
    assert got["drug_x"] == 0.5  # not a gene on the axis: nothing removed, no error
    assert got["e_masked"] == _f32(1 / 3)  # NaN LFC prediction forces P(DE) to 0
    assert got["f_untested"] == 1.0  # untested genes are in neither set


def test_direction_match_policies() -> None:
    de_true = [[F, F, F], [T, T, T], [T, T, F], [T, T, F]]
    lfc_true = [[1, 1, 1], [1, -2, 3], [1, 2, 0], [NAN, 2, 0]]
    lfc_pred = [[1, 1, 1], [0.5, -1, NAN], [0, -1, 5], [1, 1, 0]]
    inputs = _make(
        "alpha",
        ["x0", "x1", "x2"],
        ["c1"] * 4,
        ["p0", "p1", "p2", "p3"],
        de_true=de_true,
        lfc_true=lfc_true,
        lfc_pred=lfc_pred,
    )
    got = score([inputs]).per_pair.set_index("perturbation")["direction_match"]
    assert got["p0"] == 0.0  # no true DE gene: 0.0, row kept
    assert got["p1"] == _f32(2 / 3)  # a NaN prediction never matches
    assert got["p2"] == 0.0  # a zero prediction does not match a nonzero truth
    assert got["p3"] == 1.0  # NaN truth is not a true DE gene


def test_nan_probability_is_zero_and_infinite_truth_is_de() -> None:
    inf = math.inf
    inputs = _make(
        "alpha",
        ["x0", "x1", "x2"],
        ["c1"] * 2,
        ["p0", "p1"],
        p_de=[[NAN, 0.9, 0.1], [NAN, 0.9, 0.1]],
        de_true=[[T, T, F], [T, T, F]],
        lfc_true=[[inf, 1, 0], [-inf, 1, 0]],
        lfc_pred=[[1, 1, 0], [1, 1, 0]],
    )
    got = score([inputs]).per_pair.set_index("perturbation")
    # NaN P(DE) ranks as 0: order 0.9 (pos), 0.1 (neg), 0.0 (pos) -> AP = 0.5 + 0.5 * 2/3
    assert math.isclose(got.loc["p0", "binary_auprc"], 5 / 6, abs_tol=1e-6)
    assert math.isclose(got.loc["p1", "binary_auprc"], 5 / 6, abs_tol=1e-6)
    # a +-inf true LFC is a true DE gene; NaN P(DE) is not significant
    assert got.loc["p0", "sig_jaccard"] == 0.5
    assert got.loc["p1", "sig_jaccard"] == 0.5
    assert got.loc["p0", "direction_match"] == 1.0  # sign(+inf) == sign(1)
    assert got.loc["p1", "direction_match"] == 0.5  # sign(-inf) != sign(1)


def test_spearman_lfc_omits_undefined_rows() -> None:
    contexts = ["c1", "c1", "c1", "c1", "c1", "c2"]
    perts = ["p0", "p0", "p1", "p2", "p3", "q0"]
    lfc_true = [[1, 2, 3, 4]] * 6
    lfc_pred = [
        [1, 2, 3, 4],
        [1, 2, 3, 4],
        [4, 3, 2, 1],
        [1, 1, 1, 1],
        [1, 1, 2, 3],
        [2, 2, 2, 2],
    ]
    de_true = [
        [T, T, T, T],
        [T, F, F, F],
        [T, T, T, T],
        [T, T, T, T],
        [T, T, T, T],
        [T, T, T, T],
    ]
    inputs = _make(
        "alpha",
        ["x0", "x1", "x2", "x3"],
        contexts,
        perts,
        lfc_true=lfc_true,
        lfc_pred=lfc_pred,
        de_true=de_true,
    )
    result = score([inputs])
    up = float(np.corrcoef([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0])[0, 1])
    down = float(np.corrcoef([1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0])[0, 1])
    tied = float(np.corrcoef([1.0, 2.0, 3.0, 4.0], [1.5, 1.5, 3.0, 4.0])[0, 1])
    pairs = result.per_pair.set_index("perturbation")["spearman_lfc"]
    assert pairs["p0"] == up  # the single-gene row is undefined and left out
    assert pairs["p1"] == down
    assert math.isnan(pairs["p2"])  # constant prediction ranks
    assert pairs["p3"] == tied  # average ranks for ties
    assert math.isnan(pairs["q0"])
    context = result.per_context.set_index("context")["spearman_lfc"]
    assert context["c1"] == _f32((up + down + tied) / 3)
    assert math.isnan(context["c2"])
    assert result.aggregate["spearman_lfc"] == _f32((up + down + tied) / 3)


def test_spearman_nsig_policies() -> None:
    contexts = ["c1", "c1", "c1", "c2", "c3"]
    perts = ["p0", "p1", "p2", "r0", "s0"]
    p_de = [
        [0.9, 0.9, 0.9, 0.0],
        [0.9, 0.0, 0.0, 0.0],
        [0.9, 0.0, 0.0, 0.0],
        [0.9, 0.0, 0.0, 0.0],
        [0.9, 0.0, 0.0, 0.0],
    ]
    de_true = [
        [T, T, F, F],
        [F, F, F, F],
        [T, F, F, F],
        [T, F, F, F],
        [F, F, F, F],
    ]
    inputs = _make(
        "alpha", ["x0", "x1", "x2", "x3"], contexts, perts, p_de=p_de, de_true=de_true
    )
    result = score([inputs])
    nsig = result.per_context.set_index("context")["spearman_nsig"]
    rho = _f32(float(np.corrcoef([2.0, 1.0], [2.0, 1.0])[0, 1]))
    assert nsig["c1"] == rho  # p1 has no true DE gene and is left out
    assert math.isnan(nsig["c2"])  # a single perturbation: undefined
    assert nsig["c3"] == 1.0  # no perturbation with a true DE gene
    assert result.aggregate["spearman_nsig"] == _f32((rho + 1.0) / 2)


# --- discrimination_score_l1 policies inside the scorer ---


def test_l1_drops_non_finite_truth_and_counts_only_scored_perts() -> None:
    inputs = _l1(
        ["x1", "x2"],
        ["a", "b", "a", "c"],
        [1.0, 1.0],
        true=[[1, 0], [0, 1], [NAN, 0], [NAN, 0]],
        pred=[[0, 1], [0, 1], [NAN, 5], [0, 0]],
    )
    result = score([inputs])
    got = result.per_pair.set_index("perturbation")["discrimination_score_l1"]
    assert got["a"] == 0.5  # P = 2: the perturbation without finite truth is not counted
    assert got["b"] == 1.0
    assert math.isnan(got["c"])
    assert result.per_context["discrimination_score_l1"].iloc[0] == 0.75


def test_l1_all_nan_control_skips_the_context(caplog: pytest.LogCaptureFixture) -> None:
    inputs = _l1(
        ["x1", "x2"], ["a", "b"], [NAN, NAN], true=[[1, 0], [0, 1]], pred=[[1, 0], [0, 1]]
    )
    with caplog.at_level(logging.WARNING, logger="pie.metrics"):
        result = score([inputs])
    assert math.isnan(result.per_context["discrimination_score_l1"].iloc[0])
    assert math.isnan(result.aggregate["discrimination_score_l1"])
    assert "all-NaN control" in caplog.text


def test_l1_rejects_malformed_controls_and_predictions() -> None:
    genes, contexts, perts = ["x1", "x2"], ["c1", "c1"], ["a", "b"]
    with pytest.raises(ValueError, match="conflicting"):
        score([_make("alpha", genes, contexts, perts, ctrl_means=[[1, 1], [2, 2]])])
    with pytest.raises(ValueError, match="non-finite control"):
        score([_make("alpha", genes, contexts, perts, ctrl_means=[[1, NAN], [1, NAN]])])
    with pytest.raises(ValueError, match="non-finite predictions"):
        score([_make("alpha", genes, contexts, perts, delta_p_pred=[[NAN, 0], [0, 0]])])


# --- hierarchy, duplicates, validation ---


def test_hierarchy_weights_pairs_contexts_and_datasets_equally() -> None:
    perfect = ([0.9, 0.1], [T, F])
    zero = ([0.9, 0.1], [F, F])
    rows = [("c1", "p1", perfect), ("c1", "p1", zero), ("c1", "p2", perfect), ("c2", "p3", zero)]
    alpha = _make(
        "alpha",
        ["x0", "x1"],
        [r[0] for r in rows],
        [r[1] for r in rows],
        p_de=[r[2][0] for r in rows],
        de_true=[r[2][1] for r in rows],
    )
    beta = _make("beta", ["x0", "x1"], ["c1"], ["p1"], p_de=[perfect[0]], de_true=[perfect[1]])
    result = score([alpha, beta])
    auprc = result.per_context.set_index(["dataset", "context"])["binary_auprc"]
    assert auprc[("alpha", "c1")] == 0.75  # pairs p1 = 0.5 and p2 = 1.0, not rows (2/3)
    assert auprc[("alpha", "c2")] == 0.0
    assert auprc[("beta", "c1")] == 1.0
    pairs = result.per_pair.set_index(["dataset", "context", "perturbation"])
    assert pairs.loc[("alpha", "c1", "p1"), "binary_auprc"] == 0.5
    assert pairs.loc[("alpha", "c1", "p1"), "n_rows"] == 2
    # alpha = mean(0.75, 0.0) = 0.375; aggregate = mean(0.375, 1.0), not the context mean
    assert result.aggregate["binary_auprc"] == 0.6875
    assert result.context_mean["binary_auprc"] == (0.75 + 0.0 + 1.0) / 3


@pytest.mark.parametrize("layout", ["interleaved", "appended"])
def test_duplicate_rows_do_not_change_any_score(layout: str) -> None:
    case = next(c for c in ORACLE["scorer_cases"] if c["name"] == "one_dataset_three_contexts")
    full = _dataset(case["datasets"][0])
    seen: set[tuple[str, str]] = set()
    first: list[int] = []
    for i, key in enumerate(zip(full.contexts, full.perts, strict=True)):
        if key not in seen:
            seen.add(key)
            first.append(i)
    base = _take(full, np.asarray(first))
    rng = np.random.default_rng(0)
    shape = base.p_de.shape
    base.delta_p_pred = rng.normal(0.0, 1.0, shape).astype(np.float32)
    base.delta_p_true = rng.normal(0.0, 1.0, shape).astype(np.float32)
    base.ctrl_means = np.full(shape, 2.0, dtype=np.float32)
    n_rows = len(base.perts)
    if layout == "interleaved":
        idx = np.repeat(np.arange(n_rows), 2)
    else:
        idx = np.concatenate([np.arange(n_rows), np.arange(n_rows)])
    once, twice = score([base]), score([_take(base, idx)])
    for key in METRIC_KEYS:
        assert _same(twice.aggregate[key], once.aggregate[key]), key
        assert _same(twice.context_mean[key], once.context_mean[key]), key
        for a, b in zip(twice.per_context[key], once.per_context[key], strict=True):
            assert _same(float(a), float(b)), key
    for key in PAIR_METRIC_KEYS:
        for a, b in zip(twice.per_pair[key], once.per_pair[key], strict=True):
            assert _same(float(a), float(b)), key
    assert (twice.per_context["n_rows"] == 2 * once.per_context["n_rows"]).all()
    assert (twice.per_pair["n_rows"] == 2 * once.per_pair["n_rows"]).all()


def test_inputs_are_validated() -> None:
    wide = _make("alpha", ["x0", "x1"], ["c1"], ["p0"])
    wide.tested = np.ones((1, 3), dtype=bool)
    with pytest.raises(ValueError, match="tested has shape"):
        score([wide])
    short = _make("alpha", ["x0", "x1"], ["c1"], ["p0"])
    short.contexts = []
    with pytest.raises(ValueError, match="contexts"):
        score([short])


def test_datasets_without_rows_are_left_out() -> None:
    beta = _make("beta", ["x0", "x1"], ["c1"], ["p1"], p_de=[[0.9, 0.1]], de_true=[[T, F]])
    empty = _make("alpha", ["x0", "x1"], [], [])
    with_empty, alone = score([empty, beta]), score([beta])
    for key in METRIC_KEYS:
        assert _same(with_empty.aggregate[key], alone.aggregate[key]), key
    nothing = score([])
    assert nothing.per_context.empty
    assert nothing.per_pair.empty
    assert list(nothing.per_context.columns) == ["dataset", "context", "n_rows", *METRIC_KEYS]
    assert all(math.isnan(nothing.aggregate[key]) for key in METRIC_KEYS)
    assert all(math.isnan(nothing.context_mean[key]) for key in METRIC_KEYS)
