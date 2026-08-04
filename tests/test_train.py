"""The training loop, and the resume test the whole study rests on.

THE RISK THIS FILE EXISTS FOR.

A resumed run that restores weights and optimizer state but not RNG state
diverges from an uninterrupted one. It does not crash, it does not warn, and
the loss curve looks entirely normal. The study's headline is a burst arm
against its seed-matched twin, so two runs being bit-identical apart from the
burst IS the experiment. Asa named resume as the largest untested gap in the
determinism work.

The acceptance test is therefore: train N steps clean; train N steps with a
kill and resume in the middle; assert the two final states are BIT-IDENTICAL by
SHA-256 over raw tensor bytes -- the same method probes/determinism/check.py
uses. Not close. Identical.

WHAT THIS SUITE PROVES, AND WHAT IT CANNOT

It runs on CPU, because this machine has no GPU. It genuinely exercises RNG
capture and restore, checkpoint round-trip, the permutation contract,
accumulation arithmetic, and the checkpoint schedule.

It says NOTHING about CUDA kernel determinism, cuDNN flags,
CUBLAS_WORKSPACE_CONFIG, or TF32, because none of those exist without a GPU. A
green run here is not a determinism result, and `configure_determinism`
returns a COVERAGE_WARNING saying so on a CPU-only process precisely so that
nobody can read one as the other.
"""

from __future__ import annotations

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
SEAM = _load("model_seam")
ORDER = _load("data_order")
SPEC = _load("corpus_spec")
RNG = _load("rng_state")

SEQ_LEN = 16
BATCH = 4
MICRO = 2
STEPS = 4
VOCAB = 64
N_SEQ = BATCH * STEPS


def _write_corpus(path: Path) -> None:
    """A miniature corpus with the real file layout.

    Only shard 0 is needed: sequence(i) does divmod(i, SEQUENCES_PER_SHARD),
    and every index here is far below that, so the file only has to be long
    enough for the offsets actually read.
    """
    path.mkdir(parents=True, exist_ok=True)
    tokens = (np.arange(N_SEQ * SEQ_LEN, dtype=np.int64) * 7 % VOCAB
              ).astype("<u2")
    (path / SPEC.SHARD_TEMPLATE.format(index=0)).write_bytes(tokens.tobytes())
    (path / "manifest.json").write_text(
        json.dumps({"spec": {"revision": SPEC.REVISION}, "blocks": {}}),
        encoding="utf-8")


def _write_configs(tmp_path: Path, **overrides) -> tuple:
    """A small but structurally real config the loader will accept."""
    base = yaml.safe_load(
        (REPO_ROOT / "configs" / "base.yaml").read_text(encoding="utf-8"))
    base["model"].update(n_layer=2, n_head=2, n_embd=32, vocab_size=VOCAB,
                         block_size=SEQ_LEN, tie_embeddings=True)
    base["training"].update(batch_size=BATCH, seq_len=SEQ_LEN,
                            total_steps=STEPS, micro_batch=MICRO,
                            dtype="fp32")
    base["corpus"]["expected_token_budget"] = BATCH * SEQ_LEN * STEPS
    base["optimizer"]["grad_clip"] = 1.0
    base["optimizer"]["adamw_impl"] = "foreach"
    base["checkpointing"].update(weights_only_interval=2, full_interval=2)
    base["learning_rate"]["warmup_steps"] = 1
    # burst_position must fit THIS geometry: the shipped 400 is for a
    # 1024-token sequence and the loader correctly refuses it here.
    base["injection"].update(injection_step=1, burst_length_tokens=4,
                             burst_position=4)
    # Keyed from INJECTING_ARMS so cutting or adding an arm cannot leave this
    # fixture describing a study that no longer exists.
    from burst.config import INJECTING_ARMS
    base["injection"]["burst_text_paths"] = {
        arm: "README.md" for arm in INJECTING_ARMS}
    for dotted, value in overrides.items():
        section, key = dotted.split("__")
        base[section][key] = value

    # expected_param_count must match what this shape actually builds, and the
    # only honest way to know is to build it.
    probe = base["model"].copy()
    base["model"]["expected_param_count"] = _count_for(probe)

    base_path = tmp_path / "base.yaml"
    base_path.write_text(yaml.safe_dump(base, sort_keys=False),
                         encoding="utf-8")
    run_path = tmp_path / "run.yaml"
    run_path.write_text("seed: 3\narm: twin\n", encoding="utf-8")
    return base_path, run_path


def _count_for(model_section) -> int:
    from transformers.models.gpt2.modeling_gpt2 import (
        GPT2Config, GPT2LMHeadModel)

    cfg = GPT2Config(
        n_layer=model_section["n_layer"], n_head=model_section["n_head"],
        n_embd=model_section["n_embd"], n_positions=model_section["block_size"],
        vocab_size=model_section["vocab_size"],
        resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
        tie_word_embeddings=True)
    model = GPT2LMHeadModel(cfg)
    return sum(p.numel() for _, p in model.named_parameters())


def _load_cfg(tmp_path, base_path, run_path):
    sys.path.insert(0, str(REPO_ROOT))
    from burst.config import load_config

    return load_config(base_path, run_path, outdir=tmp_path / "out",
                       require_complete=True, family=SEAM.FAMILY_HF_GPT2,
                       stream=io.StringIO())


def _run(tmp_path, corpus, outdir, **kwargs):
    base_path, run_path = _write_configs(tmp_path)
    cfg = _load_cfg(tmp_path, base_path, run_path)
    return T.train(cfg, family=SEAM.FAMILY_HF_GPT2, corpus_dir=corpus,
                   outdir=outdir, stream=io.StringIO(),
                   strict_determinism=False, n_sequences=N_SEQ,
                   expected_order_digest=ORDER.seed_digest(3, N_SEQ), **kwargs)


@pytest.fixture
def corpus(tmp_path):
    path = tmp_path / "corpus"
    _write_corpus(path)
    return path


# ---------------------------------------------------------------------------
# THE ACCEPTANCE TEST
# ---------------------------------------------------------------------------


def test_a_killed_and_resumed_run_is_bit_identical_to_an_uninterrupted_one(
        tmp_path, corpus):
    """THE test. Everything else in this file supports it.

    Clean run to step 3. Then a run that stops at step 1, dies, and resumes
    from the step-1 full checkpoint. The final states must be identical byte
    for byte, not merely close.
    """
    clean = _run(tmp_path, corpus, tmp_path / "clean")

    # The interrupted run: steps 0..1, then nothing. Step 1 writes a full
    # checkpoint because both intervals are 2 and full takes precedence.
    _run(tmp_path, corpus, tmp_path / "part", steps=2)
    checkpoint = tmp_path / "part" / "step000001_full.pt"
    assert checkpoint.is_file(), "no full checkpoint to resume from"

    resumed = _run(tmp_path, corpus, tmp_path / "resumed", resume=checkpoint)

    assert resumed["final_state_digest"] == clean["final_state_digest"], (
        "a resumed run diverged from an uninterrupted one -- the failure this "
        "test exists for, and the one that would silently break every twin "
        "comparison in the study")


def test_the_resume_actually_resumed_rather_than_restarting(tmp_path, corpus):
    """A resume that quietly restarted would pass the digest test by accident.

    If `resume` were ignored and the run simply trained 0..3 again, the final
    state would match the clean run perfectly. So the test above needs this
    one beside it: the resumed run must have started at step 2.
    """
    _run(tmp_path, corpus, tmp_path / "part", steps=2)
    resumed = _run(tmp_path, corpus, tmp_path / "resumed",
                   resume=tmp_path / "part" / "step000001_full.pt")
    assert resumed["steps_run"] == [2, STEPS]
    assert resumed["resume"]["step"] == 1
    assert resumed["resume"]["rng_restored"]["torch_cpu"] is True


def test_rng_restore_is_not_yet_load_bearing_and_this_is_asserted(
        tmp_path, corpus, monkeypatch):
    """WHAT THE ACCEPTANCE TEST DOES NOT PROVE, pinned as a fact.

    Bit-identity on resume currently survives even with RNG restore removed
    entirely, because in this configuration NOTHING CONSUMES RNG AFTER
    INITIALIZATION: dropout is off in model_seam, and data order is a pure
    function of (seed, step) rather than of a random stream.

    So the acceptance test above proves the resume path is correct, and it does
    NOT prove the RNG half of it is doing anything. That is worth an assertion
    rather than a skip: a skip would quietly stop reporting the day the
    assumption breaks, whereas this FAILS -- and a failure here is good news
    that needs acting on, not a regression.

    If this test starts failing, something now consumes RNG per step (dropout
    enabled, a sampling model, a stochastic augmentation). At that moment RNG
    restore becomes load-bearing for the study's central claim, and this test
    should be inverted into the strict form it is named after. The mechanism
    itself is tested directly in tests/test_rng_state.py, so it is already
    correct when that day arrives.
    """
    clean = _run(tmp_path, corpus, tmp_path / "clean")
    _run(tmp_path, corpus, tmp_path / "part", steps=2)

    monkeypatch.setattr(RNG, "restore", lambda state: {"skipped": True})
    broken = _run(tmp_path, corpus, tmp_path / "broken",
                  resume=tmp_path / "part" / "step000001_full.pt")

    assert broken["final_state_digest"] == clean["final_state_digest"], (
        "dropping RNG restore now CHANGES the result, which means something "
        "began consuming RNG per step. That is a real change to the loop: RNG "
        "restore is now load-bearing, this test should be inverted into its "
        "strict form, and the resume guarantee genuinely depends on it.")


# ---------------------------------------------------------------------------
# The permutation contract is a precondition, not a log line
# ---------------------------------------------------------------------------


def test_a_wrong_order_digest_refuses_before_any_batch(tmp_path, corpus):
    base_path, run_path = _write_configs(tmp_path)
    cfg = _load_cfg(tmp_path, base_path, run_path)
    with pytest.raises(Exception, match="DOES NOT MATCH THE MANIFEST"):
        T.train(cfg, family=SEAM.FAMILY_HF_GPT2, corpus_dir=corpus,
                outdir=tmp_path / "out2", stream=io.StringIO(),
                strict_determinism=False, n_sequences=N_SEQ,
                expected_order_digest="0" * 64)


def test_the_loop_verifies_against_the_recorded_digest_not_its_own(tmp_path):
    """A check against a self-derived value proves only determinism."""
    with pytest.raises(T.TrainError, match="computes itself"):
        T._expected_order_digest(999)


def test_recorded_digests_match_what_the_loop_would_compute():
    """The corpus report and data_order must agree for every study seed."""
    report = REPO_ROOT / "docs" / "measurements" / "11-corpus.json"
    if not report.is_file():
        pytest.skip("no corpus report built")
    digests = json.loads(report.read_text(encoding="utf-8"))["data_order"][
        "permutation_digests"]
    assert len(digests) == 10
    for seed in ("0", "3"):
        assert digests[seed] == ORDER.seed_digest(
            int(seed), SPEC.TRAIN_SEQUENCES)


# ---------------------------------------------------------------------------
# micro_batch and family have no defaults
# ---------------------------------------------------------------------------


def test_the_loop_refuses_a_null_micro_batch(tmp_path, corpus):
    import dataclasses

    base_path, run_path = _write_configs(tmp_path)
    cfg = _load_cfg(tmp_path, base_path, run_path)
    undecided = dataclasses.replace(
        cfg, training=dataclasses.replace(cfg.training, micro_batch=None))
    with pytest.raises(T.TrainError, match="NO DEFAULT"):
        T.train(undecided, family=SEAM.FAMILY_HF_GPT2, corpus_dir=corpus,
                outdir=tmp_path / "o", stream=io.StringIO(),
                strict_determinism=False, n_sequences=N_SEQ)


def test_the_cli_requires_an_explicit_family():
    parser = T._build_parser()
    family = {a.dest: a for a in parser._actions}["family"]
    assert family.required is True
    assert family.default is None


# ---------------------------------------------------------------------------
# Accumulation, clipping, and the schedule
# ---------------------------------------------------------------------------


def test_accumulation_divides_the_batch_exactly(tmp_path, corpus):
    record = _run(tmp_path, corpus, tmp_path / "a")
    assert record["micro_batch"] == MICRO
    assert record["accumulation_steps"] == BATCH // MICRO
    assert record["accumulation_steps"] * MICRO == BATCH


def test_the_pre_clip_gradient_norm_is_logged_every_step(tmp_path, corpus):
    """The measurement that discharges the grad-clip obligation."""
    record = _run(tmp_path, corpus, tmp_path / "a")
    assert len(record["grad_norms"]) == STEPS
    for row in record["grad_norms"]:
        assert "pre_clip_grad_norm" in row
        assert row["pre_clip_grad_norm"] >= 0.0


def test_clipping_happens_after_accumulation_not_inside_it():
    """Read from the source: the order is the algorithm.

    Clipping inside the accumulation loop is a different algorithm wearing the
    same config value, and it would look fine.
    """
    source = (REPO_ROOT / "scripts" / "train.py").read_text(encoding="utf-8")
    body = source.split("for micro_index in range(accum):")[1]
    inner, after = body.split("clip_grad_norm_", 1)
    assert "loss.backward()" in inner
    assert "optimizer.step()" in after
    assert "clip_grad_norm_" not in inner


def test_the_normalisation_choice_is_documented_at_the_line():
    """Anyone changing it must know they are changing the study."""
    source = (REPO_ROOT / "scripts" / "train.py").read_text(encoding="utf-8")
    assert "/ accum" in source
    assert "NORMALISATION" in source
    assert "not associative" in source


def test_checkpoints_follow_the_config_schedule(tmp_path, corpus):
    """The loop must use checkpoint_kind_at, not reimplement the rules."""
    record = _run(tmp_path, corpus, tmp_path / "a")
    kinds = [(c["step"], c["kind"]) for c in record["checkpoints"]]
    assert kinds == [(1, "full"), (3, "full")]
    assert all(c["carries_rng"] for c in record["checkpoints"])


# ---------------------------------------------------------------------------
# Checkpoint round-trip
# ---------------------------------------------------------------------------


def test_saving_and_reloading_reproduces_the_state_exactly(tmp_path, corpus):
    record = _run(tmp_path, corpus, tmp_path / "a", steps=2)
    before = record["final_state_digest"]
    model, optimizer = record["model_ref"], record["optimizer_ref"]
    T.load_checkpoint(tmp_path / "a" / "step000001_full.pt", model, optimizer)
    assert T.state_digest(model, optimizer) == before


def test_a_weights_only_checkpoint_cannot_be_resumed_from(tmp_path, corpus):
    """It has no optimizer state and no RNG state."""
    record = _run(tmp_path, corpus, tmp_path / "a", steps=2)
    path = tmp_path / "wo.pt"
    T.save_checkpoint(path, kind="weights_only", step=1,
                      model=record["model_ref"],
                      optimizer=record["optimizer_ref"],
                      cfg=_load_cfg(tmp_path, *_write_configs(tmp_path)),
                      family=SEAM.FAMILY_HF_GPT2)
    with pytest.raises(T.TrainError, match="cannot be resumed from"):
        T.load_checkpoint(path, record["model_ref"], record["optimizer_ref"])


# ---------------------------------------------------------------------------
# Determinism configuration, and what a CPU run does not prove
# ---------------------------------------------------------------------------


def test_determinism_settings_are_returned_not_merely_applied():
    applied = T.configure_determinism(3, strict=False)
    assert applied["torch.use_deterministic_algorithms"] is True
    assert applied["torch.backends.cudnn.deterministic"] is True
    assert applied["torch.backends.cuda.matmul.allow_tf32"] is False


def test_a_cpu_run_says_it_proves_nothing_about_cuda():
    """A green CPU suite must not read as a determinism result."""
    applied = T.configure_determinism(3, strict=False)
    if applied["cuda_available"]:
        pytest.skip("CUDA present; the CPU-only warning does not apply")
    assert "nothing about CUDA kernel determinism" in applied[
        "COVERAGE_WARNING"]


def test_the_device_is_recorded_so_a_pair_can_be_shown_to_share_one():
    """WHICH CARD RAN THIS. An arm and its seed-matched twin trained on
    different card models are not a valid pair, because kernel selection
    depends on the device -- and before this field nothing recorded it.

    The value cannot live in run_provenance.yaml: `burst/` may not import
    torch, and querying the device initialises CUDA, which would happen before
    the CUBLAS_WORKSPACE_CONFIG guard. So it lands in train_record.json. See
    S87.
    """
    applied = T.configure_determinism(3, strict=False)
    for key in ("device_name", "device_capability", "torch_version",
                "cuda_version"):
        assert key in applied, f"{key} must be recorded, even as None"
    # torch is always present in this environment, so this one is never None.
    assert applied["torch_version"]
    if applied["cuda_available"]:
        assert applied["device_name"], "CUDA present but no device recorded"
        assert applied["device_capability"]
    else:
        assert applied["device_name"] is None
        assert applied["device_capability"] is None


def test_the_injection_hook_is_wired_in_not_a_seam():
    """Step 14 replaced the empty seam with a real hook.

    This test previously asserted the seam did NOTHING, which was correct until
    the hook existed. Inverted rather than deleted: the loop must now call
    injection.apply on raw rows BEFORE the input/target shift, because splicing
    into an already-shifted pair would be a second splice. The behavioural
    proof lives in tests/test_injection.py.
    """
    source = (REPO_ROOT / "scripts" / "train.py").read_text(encoding="utf-8")
    assert "_injection_seam" not in source, "the empty seam is still present"
    assert "INJECT.apply(plan, step, micro_index, raw)" in source
    # Order matters: inject on raw rows, then shift.
    inject_at = source.index("INJECT.apply(")
    shift_at = source.index("reader.shift(raw)")
    assert inject_at < shift_at, (
        "the hook runs after the input/target shift, which means it would have "
        "to patch two tensors separately -- a second splice")
