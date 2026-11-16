import json
import os
import socket

import pytest

from se import cli, gates
from se.scorers import finbert, finbert_tone, hf_classifier, lm_lexicon

CSV = "Word,Negative,Positive,Other\nGAINS,0,2009,0\nSTRONG,0,2009,0\nLOSS,2009,0,0\nWEAK,2009,0,0\nRETIRED,0,-2020,0\nPLAIN,0,0,0\n"


@pytest.fixture
def lex(tmp_path):
    p = tmp_path / "lm.csv"
    p.write_text(CSV)
    return lm_lexicon.LMLexicon(p)


def _fake_model(tmp_path, id2label):
    d = tmp_path / "model"
    d.mkdir(exist_ok=True)
    (d / "config.json").write_text(json.dumps({"id2label": id2label}))
    return d


def test_gates_file_facts_and_table(capsys):
    g = gates.load()
    assert g["audit"]["date"] == "2026-11-14" and "owner" in g["audit"]["source"]
    st = {k: v["status"] for k, v in g["gates"].items()}
    assert st["data:fnspid"] == "blocked" and st["data:alpaca_benzinga"] == "blocked"
    assert st["data:sec_edgar"] == "not_adopted" and st["data:gdelt"] == "rejected"
    assert st["data:user_import"] == "allowed_with_licence_note"
    assert st["study:confirmatory"] == "blocked" and st["storage:real_data_and_weights"] == "blocked"
    fp, ft, lm = (g["gates"]["scorer:" + k] for k in ("finbert_prosus", "finbert_tone", "lm_lexicon"))
    assert (fp["licence"], fp["contamination"], fp["allowed_use"]) == ("verified_permitted", "unresolved", "exploratory")
    assert (ft["licence"], ft["allowed_use"]) == ("unresolved", "blocked") and lm["allowed_use"] == "exploratory"
    assert all(v.get("unlock") for v in g["gates"].values())
    assert cli.main(["gates"]) == 0
    out = capsys.readouterr().out
    assert all(k in out for k in g["gates"]) and "2026-11-14" in out


def test_lm_buckets_and_counts(lex):
    a = lex.score("Strong gains despite one loss")
    assert (a["lm_pos_count"], a["lm_neg_count"], a["tokens"], a["bucket"]) == (2, 1, 5, "positive")
    assert a["x"] == pytest.approx(0.2) and a["probabilities"] is None and a["matched"] == ["strong", "gains", "loss"]
    assert lex.score("weak quarter")["bucket"] == "negative"
    assert lex.score("gains and loss")["bucket"] == "neutral (zero-net)"
    none = lex.score("plain retired words 2024")
    assert none["bucket"] == "no lexicon coverage" and none["x"] == 0 and none["tokens"] == 3
    assert lex.score("lossless gainsaying")["bucket"] == "no lexicon coverage"       # whole tokens only
    assert lex.score("LOSS-making")["lm_neg_count"] == 1 and "EXPLORATORY ONLY" in a["flag"]
    assert "retired" not in lex.positive and a["version"].startswith("sha256:")


def test_lm_refuses_without_file(tmp_path):
    for path in (None, tmp_path / "missing.csv"):
        with pytest.raises(lm_lexicon.LexiconMissingError, match="does not include or download"):
            lm_lexicon.LMLexicon(path)
    bad = tmp_path / "bad.csv"
    bad.write_text("Term,Score\nx,1\n")
    with pytest.raises(ValueError, match="missing columns"):
        lm_lexicon.LMLexicon(bad)


def test_label_map_rules():
    ok = hf_classifier.resolve_label_map({"0": "positive", "1": "negative", "2": "neutral"})
    assert ok == {"positive": 0, "negative": 1, "neutral": 2}
    assert hf_classifier.resolve_label_map({0: "Neutral", 1: "Positive", 2: "Negative"})["positive"] == 1
    for bad in ({"0": "LABEL_0", "1": "LABEL_1", "2": "LABEL_2"}, {"0": "positive", "1": "negative"},
                {"0": "bullish", "1": "bearish", "2": "neutral"}, None):
        with pytest.raises(hf_classifier.AmbiguousLabelMap):
            hf_classifier.resolve_label_map(bad)
    labels = {"0": "LABEL_0", "1": "LABEL_1", "2": "LABEL_2"}
    assert hf_classifier.resolve_label_map(labels, finbert_tone.OVERRIDE) == {"neutral": 0, "positive": 1, "negative": 2}
    with pytest.raises(hf_classifier.AmbiguousLabelMap, match="source"):
        hf_classifier.resolve_label_map(labels, {"map": finbert_tone.OVERRIDE["map"], "source": ""})


def test_finbert_outputs_carry_exploratory_flag(tmp_path):
    s = finbert.FinBERTScorer(_fake_model(tmp_path, {"0": "positive", "1": "negative", "2": "neutral"}))
    out = s.format_output(hf_classifier.softmax([2.0, 0.0, 1.0]))
    assert out["flag"] == "EXPLORATORY ONLY, training-data dates unverified"
    assert out["label"] == "positive" and out["x"] == pytest.approx(out["output_scores"]["positive"] - out["output_scores"]["negative"])
    assert sum(out["output_scores"].values()) == pytest.approx(1.0) and "price move" in out["note"]
    blocked = {"gates": {"scorer:finbert_prosus": {"allowed_use": "blocked"}}}
    with pytest.raises(gates.ScorerBlockedError):
        finbert.FinBERTScorer(tmp_path / "model", gates=blocked)


def test_finbert_tone_blocked_before_touching_files(tmp_path):
    with pytest.raises(gates.ScorerBlockedError, match="licence"):
        finbert_tone.FinBERTToneScorer(tmp_path / "does-not-exist")          # real gates.yaml: blocked
    unblocked = {"gates": {"scorer:finbert_tone": {"allowed_use": "exploratory", "licence": "verified_permitted"}}}
    with pytest.raises(FileNotFoundError, match="never downloads"):
        finbert_tone.FinBERTToneScorer(tmp_path / "does-not-exist", gates=unblocked)
    s = finbert_tone.FinBERTToneScorer(_fake_model(tmp_path, {"0": "LABEL_0", "1": "LABEL_1", "2": "LABEL_2"}), gates=unblocked)
    assert s.label_map == {"neutral": 0, "positive": 1, "negative": 2}
    assert s.format_output([0.2, 0.7, 0.1])["label"] == "positive" and "EXPLORATORY ONLY" in s.format_output([0.2, 0.7, 0.1])["flag"]


def test_no_code_path_downloads(tmp_path, monkeypatch, lex):
    def no_network(*a, **k):
        raise AssertionError("network access attempted")
    monkeypatch.setattr(socket, "socket", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)
    monkeypatch.setattr(socket, "getaddrinfo", no_network)
    assert lex.score("strong gains")["bucket"] == "positive"
    with pytest.raises(FileNotFoundError):
        hf_classifier.LocalHFClassifier(tmp_path / "ProsusAI" / "finbert")       # a hub-style name is just a missing path
    with pytest.raises(gates.ScorerBlockedError):
        finbert_tone.FinBERTToneScorer("yiyanghkust/finbert-tone")
    s = finbert.FinBERTScorer(_fake_model(tmp_path, {"0": "positive", "1": "negative", "2": "neutral"}))
    with pytest.raises((hf_classifier.MissingDependencyError, OSError)):          # no torch, or no local weights
        s.score("strong gains")
    assert os.environ.get("HF_HUB_OFFLINE") == "1"


@pytest.mark.models
def test_real_local_weights():
    s = finbert.FinBERTScorer(os.environ["SE_MODEL_DIR"])
    assert "EXPLORATORY ONLY" in s.score("Shares rally on strong demand")["flag"]
