"""The injection hook: proof the burst landed, at the right step, and changed something.

THE TEST THAT MATTERS is behavioural, not structural. "The hook was invoked"
proves nothing: a hook can fire at the wrong step, replace a row that never
reaches backward(), splice the wrong text, or fire for twin. So the load-bearing
test trains the same seed twice past the injection step -- once injecting, once
as twin -- and asserts the weights are BIT-IDENTICAL through step 199 and
DIFFERENT from step 200, by the same SHA-256-over-raw-tensor-bytes method
probes/determinism/check.py uses.

That single assertion covers three claims at once: the burst landed (something
changed), it landed at the right step (nothing changed before it), and it
reached the gradient (the change is in the weights, not just in a tensor).

The second load-bearing test is reachability under accumulation. The batch is
split into accumulation steps, and a burst spliced into a sequence that never
reaches backward() is invisible to every other check -- including the first
test, if the row landed outside the batch entirely. So the exact 194 token IDs
are asserted present, contiguously at the configured position, in the tensor
passed to the forward whose backward() actually runs -- captured by wrapping the
real call rather than by re-deriving which micro-batch should hold it.

WHAT THIS SUITE DOES NOT COVER: CUDA. It runs on CPU because this machine has
no GPU, so it establishes the bookkeeping and says nothing about kernel
determinism. Same split as the step 12 acceptance test.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
np = pytest.importorskip("numpy")


def _load(name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


T = _load("train")
INJECT = _load("injection")
SEAM = _load("model_seam")
ORDER = _load("data_order")
SPEC = _load("corpus_spec")

sys.path.insert(0, str(REPO_ROOT))
from burst.config import ARMS, INJECTING_ARMS  # noqa: E402

# THE REAL BURST TEXTS, THE REAL TOKENIZER, THE REAL POSITION. Only the model
# and the run length are shrunk. Using a stub tokenizer and a throwaway burst
# file would have tested the plumbing against text the study does not use --
# and the loader is right to refuse a burst path outside the repo, which is
# what forced this and made the test better.
SEQ_LEN = 1024
BURST_LEN = 194
POSITION = 400
VOCAB = 50257
BATCH = 4
MICRO = 2
STEPS = 6
INJECT_STEP = 3
N_SEQ = BATCH * STEPS


@pytest.fixture(scope="module")
def burst_file():
    """The committed fluent-false text, repo-relative as the loader demands."""
    return Path("bursts/fluent_false.txt")


@pytest.fixture(scope="module")
def tokenizer():
    from burst_match import load_tokenizer

    return load_tokenizer(stream=io.StringIO())


@pytest.fixture
def corpus(tmp_path):
    path = tmp_path / "corpus"
    path.mkdir(parents=True, exist_ok=True)
    tokens = (np.arange(N_SEQ * SEQ_LEN, dtype=np.int64) * 7 % VOCAB
              ).astype("<u2")
    (path / SPEC.SHARD_TEMPLATE.format(index=0)).write_bytes(tokens.tobytes())
    (path / "manifest.json").write_text(json.dumps({"blocks": {}}),
                                        encoding="utf-8")
    return path


def _write_configs(tmp_path, arm, burst_path, **overrides):
    base = yaml.safe_load(
        (REPO_ROOT / "configs" / "base.yaml").read_text(encoding="utf-8"))
    base["model"].update(n_layer=2, n_head=2, n_embd=32, vocab_size=VOCAB,
                         block_size=SEQ_LEN, tie_embeddings=True)
    base["training"].update(batch_size=BATCH, seq_len=SEQ_LEN,
                            total_steps=STEPS, micro_batch=MICRO,
                            dtype="fp32")
    base["corpus"]["expected_token_budget"] = BATCH * SEQ_LEN * STEPS
    base["optimizer"].update(grad_clip=1.0, adamw_impl="foreach")
    base["checkpointing"].update(weights_only_interval=2, full_interval=2)
    base["learning_rate"]["warmup_steps"] = 1
    base["injection"].update(injection_step=INJECT_STEP,
                             burst_length_tokens=BURST_LEN,
                             burst_position=POSITION)
    shipped = yaml.safe_load(
        (REPO_ROOT / "configs" / "base.yaml").read_text(encoding="utf-8"))
    base["injection"]["burst_text_paths"] = dict(
        shipped["injection"]["burst_text_paths"])
    for section, values in overrides.items():
        base[section].update(values)

    from transformers.models.gpt2.modeling_gpt2 import (
        GPT2Config, GPT2LMHeadModel)
    m = base["model"]
    probe = GPT2LMHeadModel(GPT2Config(
        n_layer=m["n_layer"], n_head=m["n_head"], n_embd=m["n_embd"],
        n_positions=m["block_size"], vocab_size=m["vocab_size"],
        resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
        tie_word_embeddings=True))
    base["model"]["expected_param_count"] = sum(
        p.numel() for _, p in probe.named_parameters())

    b = tmp_path / f"base_{arm}.yaml"
    b.write_text(yaml.safe_dump(base, sort_keys=False), encoding="utf-8")
    r = tmp_path / f"run_{arm}.yaml"
    r.write_text(f"seed: 3\narm: {arm}\n", encoding="utf-8")
    return b, r


def _cfg(tmp_path, arm, burst_path, **overrides):
    from burst.config import load_config

    b, r = _write_configs(tmp_path, arm, burst_path, **overrides)
    return load_config(b, r, outdir=tmp_path / f"out_{arm}",
                       require_complete=True, stream=io.StringIO())


def _run(tmp_path, corpus, arm, burst_path, outdir, steps=None):
    cfg = _cfg(tmp_path, arm, burst_path)
    return T.train(cfg, family=SEAM.FAMILY_HF_GPT2, corpus_dir=corpus,
                   outdir=outdir, stream=io.StringIO(),
                   strict_determinism=False, n_sequences=N_SEQ, steps=steps,
                   expected_order_digest=ORDER.seed_digest(3, N_SEQ))


# ---------------------------------------------------------------------------
# THE TEST THAT MATTERS
# ---------------------------------------------------------------------------


def test_identical_before_the_injection_step_and_different_after(
        tmp_path, corpus, burst_file):
    """The load-bearing behavioural test.

    Same seed, two arms. Bit-identical through the step before injection;
    different from the injection step onward. Proves the burst landed, landed
    at the right step, and reached the gradient.
    """

    before = {}
    after = {}
    for arm in ("fluent-false", "twin"):
        # Up to and INCLUDING the step before injection.
        r = _run(tmp_path, corpus, arm, burst_file,
                 tmp_path / f"pre_{arm}", steps=INJECT_STEP)
        before[arm] = r["final_state_digest"]
        # Through the injection step.
        r = _run(tmp_path, corpus, arm, burst_file,
                 tmp_path / f"post_{arm}", steps=INJECT_STEP + 1)
        after[arm] = r["final_state_digest"]

    assert before["fluent-false"] == before["twin"], (
        "the injecting arm and its twin diverged BEFORE the injection step -- "
        "something in the injection path is perturbing the run early: an RNG "
        "draw, a consumed sampler index, or a hook firing at the wrong step")
    assert after["fluent-false"] != after["twin"], (
        "the injecting arm and its twin are still identical AFTER the "
        "injection step -- the burst did not reach the gradient, which reads "
        "exactly like a negative result and is not one")


def test_the_burst_tokens_reach_the_tensor_that_backward_runs_on(
        tmp_path, corpus, burst_file, tokenizer, monkeypatch):
    """Reachability under accumulation.

    A burst spliced into a sequence that never reaches backward() is invisible
    to every other check. Captured by wrapping the REAL loss call, not by
    re-deriving which micro-batch ought to hold it.
    """
    cfg = _cfg(tmp_path, "fluent-false", burst_file)
    plan = INJECT.build_plan(cfg, tokenizer=tokenizer)

    seen = []
    real = SEAM.compute_loss

    def capture(model, inputs, targets):
        seen.append(inputs.detach().clone())
        return real(model, inputs, targets)

    monkeypatch.setattr(SEAM, "compute_loss", capture)
    T.train(cfg, family=SEAM.FAMILY_HF_GPT2, corpus_dir=corpus,
            outdir=tmp_path / "reach", stream=io.StringIO(),
            strict_determinism=False, n_sequences=N_SEQ,
            steps=INJECT_STEP + 1,
            expected_order_digest=ORDER.seed_digest(3, N_SEQ))

    burst = torch.tensor(list(plan.burst_ids), dtype=torch.long)
    # inputs are the sequence minus its last token, so the burst region is
    # present in full whenever position + burst_length <= seq_len - 1.
    hits = [
        (t_i, row)
        for t_i, tensor in enumerate(seen)
        for row in range(tensor.shape[0])
        if torch.equal(
            tensor[row, plan.position:plan.position + plan.burst_length],
            burst)
    ]
    assert hits, (
        "the burst's token IDs never appeared, at the configured position, in "
        "any tensor that backward() ran on. The splice may have landed in a "
        "micro-batch that is never forwarded.")
    assert len(hits) == 1, f"the burst appeared {len(hits)} times: {hits}"


# ---------------------------------------------------------------------------
# The five silent breakages
# ---------------------------------------------------------------------------


def test_twin_never_injects(tmp_path, burst_file, tokenizer):
    assert INJECT.build_plan(_cfg(tmp_path, "twin", burst_file),
                             tokenizer=tokenizer) is None


@pytest.mark.parametrize("arm", INJECTING_ARMS)
def test_every_injecting_arm_produces_a_plan(tmp_path, burst_file, arm,
                                             tokenizer):
    """Both directions, because a hook that no-ops reads as a null result."""
    plan = INJECT.build_plan(_cfg(tmp_path, arm, burst_file),
                             tokenizer=tokenizer)
    assert plan is not None
    assert plan.arm == arm
    assert plan.step == INJECT_STEP


def test_scrambled_corpus_is_not_reachable_as_an_arm():
    """It has a committed text and committed measurements and is not a run.

    bursts/provenance.json holds SEVEN arms; the study has six. The hook looks
    up by arm name and never iterates either bursts/ or provenance.
    """
    assert "scrambled-corpus" not in ARMS
    assert "scrambled-corpus" not in INJECTING_ARMS
    provenance = json.loads(
        (REPO_ROOT / "bursts" / "provenance.json").read_text(encoding="utf-8"))
    assert "scrambled-corpus" in provenance["arms"], (
        "if this entry disappears the cut was done by deletion rather than by "
        "descoping, and its measurements stop being reproducible")
    shipped = yaml.safe_load(
        (REPO_ROOT / "configs" / "base.yaml").read_text(encoding="utf-8"))
    assert "scrambled-corpus" not in shipped["injection"]["burst_text_paths"]


def test_the_hook_draws_no_randomness():
    """A single RNG draw here breaks bit-identity with the twin before step 200."""
    import ast

    source = (REPO_ROOT / "scripts" / "injection.py").read_text(
        encoding="utf-8")
    tree = ast.parse(source)
    banned = {"random", "randint", "randn", "rand", "shuffle", "choice",
              "manual_seed", "sample"}
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = getattr(node.func, "attr", getattr(node.func, "id", ""))
            if name in banned:
                calls.append((name, node.lineno))
    assert not calls, f"randomness in the injection path: {calls}"
    assert "derived_seed" in source


def test_the_slot_is_derived_and_stable_across_processes():
    """Same mechanism as data order; no PYTHONHASHSEED dependence."""
    import os
    import subprocess

    code = (
        "import sys; sys.path.insert(0, r'%s');"
        "from injection import batch_slot_for;"
        "print(batch_slot_for(3, 256))" % (REPO_ROOT / "scripts")
    )
    env = dict(os.environ, PYTHONHASHSEED="12345")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, env=env, check=True)
    assert int(out.stdout.strip()) == INJECT.batch_slot_for(3, 256)


def test_a_burst_text_that_does_not_match_provenance_is_refused(tmp_path, tokenizer):
    """Refuses, not warns: a warning at step 200 is a warning nobody reads."""
    path = tmp_path / "wrong.txt"
    path.write_text("not the committed text", encoding="utf-8")
    provenance = tmp_path / "provenance.json"
    provenance.write_text(json.dumps(
        {"arms": {"fluent-false": {"sha256": "0" * 64, "tokens": 194}}}),
        encoding="utf-8")
    import unittest.mock as mock

    with mock.patch.object(INJECT, "BURSTS_PROVENANCE", provenance):
        with pytest.raises(INJECT.InjectionError, match="DOES NOT MATCH ITS PROVENANCE"):
            INJECT.load_burst_ids("fluent-false", path, 194, tokenizer)


def test_a_burst_of_the_wrong_length_is_refused(tmp_path, tokenizer, monkeypatch):
    path = tmp_path / "short.txt"
    path.write_text("abc", encoding="utf-8")
    monkeypatch.setattr(INJECT, "BURSTS_PROVENANCE", Path("does-not-exist"))
    with pytest.raises(INJECT.InjectionError, match="tokenizes to 1 tokens"):
        INJECT.load_burst_ids("fluent-false", path, 194, tokenizer)


# ---------------------------------------------------------------------------
# The splice is not reimplemented, and the position is the measured one
# ---------------------------------------------------------------------------


def test_the_hook_uses_assemble_sequence_and_does_not_splice_itself():
    source = (REPO_ROOT / "scripts" / "injection.py").read_text(
        encoding="utf-8")
    assert "from burst_match import assemble_sequence" in source
    assert "assemble_sequence(filler_ids, burst_ids, position)" in source
    # A hand-rolled splice would look like list concatenation around a slice.
    assert "filler[:position]" not in source
    assert "+ list(burst" not in source


def test_the_configured_position_is_what_step_8_measured():
    """Drift breaks the suite rather than the study."""
    shipped = yaml.safe_load(
        (REPO_ROOT / "configs" / "base.yaml").read_text(encoding="utf-8"))
    configured = shipped["injection"]["burst_position"]
    measured = json.loads(
        (REPO_ROOT / "docs" / "measurements" /
         "8b-i-in-context-match.json").read_text(encoding="utf-8"))
    assert configured == measured["burst_position"] == 400
    for arm, record in measured["arms"].items():
        assert record["position"] == configured, arm


def test_the_shipped_config_burst_length_matches_every_arm_text():
    shipped = yaml.safe_load(
        (REPO_ROOT / "configs" / "base.yaml").read_text(encoding="utf-8"))
    length = shipped["injection"]["burst_length_tokens"]
    provenance = json.loads(
        (REPO_ROOT / "bursts" / "provenance.json").read_text(encoding="utf-8"))
    for arm in INJECTING_ARMS:
        assert provenance["arms"][arm]["tokens"] == length, arm


def test_the_injection_step_is_the_step_after_the_anchor_checkpoint(tmp_path):
    """Step 199 holds the state immediately before injection, by (199+1)%50==0."""
    from burst.config import load_config

    cfg = load_config(REPO_ROOT / "configs" / "base.yaml",
                      REPO_ROOT / "configs" / "runs" / "seed03_twin.yaml",
                      outdir=tmp_path, require_complete=False,
                      stream=io.StringIO())
    assert cfg.injection.injection_step == 200
    assert cfg.checkpoint_kind_at(199) == "weights_only"
    assert cfg.checkpoint_kind_at(200) is None


# ---------------------------------------------------------------------------
# The refactor is inert except at the injection step
# ---------------------------------------------------------------------------


def test_rows_then_shift_equals_the_old_batch(tmp_path, corpus):
    """The reader split must change nothing on a non-injecting step."""
    reader = T.ShardReader(corpus, seq_len=SEQ_LEN)
    indices = [0, 1, 2, 3]
    inputs_a, targets_a = reader.batch(indices)
    inputs_b, targets_b = reader.shift(reader.rows(indices))
    assert torch.equal(inputs_a, inputs_b)
    assert torch.equal(targets_a, targets_b)


def test_a_non_injecting_step_leaves_the_rows_untouched(tmp_path, corpus,
                                                        burst_file, tokenizer):
    plan = INJECT.build_plan(_cfg(tmp_path, "fluent-false", burst_file),
                             tokenizer=tokenizer)
    reader = T.ShardReader(corpus, seq_len=SEQ_LEN)
    raw = reader.rows([0, 1])
    out, fired = INJECT.apply(plan, plan.step - 1, plan.micro_index, raw)
    assert fired is False
    assert torch.equal(out, raw)
    out, fired = INJECT.apply(plan, plan.step, plan.micro_index + 99, raw)
    assert fired is False
    assert torch.equal(out, raw)


def test_injection_replaces_exactly_one_row_and_consumes_no_index(
        tmp_path, corpus, burst_file, tokenizer):
    plan = INJECT.build_plan(_cfg(tmp_path, "fluent-false", burst_file),
                             tokenizer=tokenizer)
    reader = T.ShardReader(corpus, seq_len=SEQ_LEN)
    raw = reader.rows(list(range(MICRO)))
    out, fired = INJECT.apply(plan, plan.step, plan.micro_index, raw)
    assert fired is True
    assert out.shape == raw.shape, "the batch changed size"
    changed = [i for i in range(raw.shape[0])
               if not torch.equal(out[i], raw[i])]
    assert changed == [plan.row]
    assert torch.equal(
        out[plan.row],
        torch.tensor(list(plan.sequence), dtype=raw.dtype))


def test_the_record_carries_what_cannot_be_recovered_later(
        tmp_path, corpus, burst_file):
    """The burst-region losses exist only at the moment of injection."""
    record = _run(tmp_path, corpus, "fluent-false", burst_file,
                  tmp_path / "rec", steps=INJECT_STEP + 1)
    fired = record["injection_fired"]
    assert fired is not None, "injection never fired"
    for key in ("arm", "step", "batch_slot", "micro_index", "row",
                "burst_token_sha256", "burst_file_sha256", "position"):
        assert key in fired, key
    assert fired["step"] == INJECT_STEP
    region = fired["burst_region"]
    assert region["n_predictions"] == BURST_LEN
    assert len(region["per_token_losses"]) == BURST_LEN
    assert all(v >= 0 for v in region["per_token_losses"])


def test_twin_records_no_injection(tmp_path, corpus, burst_file):
    record = _run(tmp_path, corpus, "twin", burst_file, tmp_path / "twinrec",
                  steps=INJECT_STEP + 1)
    assert record["injection_plan"] is None
    assert record["injection_fired"] is None
