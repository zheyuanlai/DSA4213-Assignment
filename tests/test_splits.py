"""Split-integrity checks (added after an external code review, NOTES D122/D123). Skipped when data/processed
does not exist (e.g. a fresh clone before scripts/prepare_data.sh).

Hard guarantees: LM splits are review- and text-disjoint and share no text with the labelled IMDb splits;
classifier-train / classifier-val partition IMDb-train (22,500 / 2,500, stratified); the D46 robustness subset
excludes exactly the 123 test reviews whose text occurs in IMDb-train. Known, documented overlap: 26 classifier-val
reviews have a same-label copy in classifier-train (they do not change any chosen C — see
results/part3/val_dedup_sensitivity.csv); the committed results/part3/split_overlap.json must match a recomputation.
"""
import json

import pytest

from src.common import ROOT

pytestmark = pytest.mark.skipif(not (ROOT / "data/processed/lm_split.json").exists(), reason="processed data not built")


@pytest.fixture(scope="module")
def ov():
    from src.analysis.split_checks import overlaps
    return overlaps()


def test_lm_splits_disjoint(ov):
    lm = ov["lm_splits"]
    assert lm["sizes"] == {"lm_val": 2000, "lm_test": 2000, "lm_train": 44885}
    assert set(lm["index_overlaps"].values()) == {0}
    assert set(lm["text_overlaps"].values()) == {0}
    assert set(lm["text_overlap_with_labelled_train_or_test"].values()) == {0}


def test_classifier_split_partition(ov):
    c = ov["classifier_split"]
    assert c["sizes"] == {"clf_train": 22500, "clf_val": 2500}
    assert c["index_overlap"] == 0 and c["union"] == 25000
    assert c["labels_agree_within_split_groups"]


def test_robust_subset_matches_train_test_duplicates(ov):
    assert ov["labelled"]["imdb_test_reviews_with_text_in_imdb_train"] == 123
    meta = json.load(open(ROOT / "results/part3/classify_meta.json"))
    assert meta["robust_excluded_n"] == 123


def test_split_overlap_record_is_current(ov):
    path = ROOT / "results/part3/split_overlap.json"
    if not path.exists():
        pytest.skip("split_overlap.json not written yet")
    assert json.load(open(path)) == json.loads(json.dumps(ov))
