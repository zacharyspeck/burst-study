#!/usr/bin/env python
"""Measure how hard a candidate burst passage shoves the weights.

The headline comparison in this study is a coherent passage against a noise
passage, and that comparison only means anything if both passages deliver the
same size shove. This script measures the shove on public GPT-2 (124M) and
prints the numbers side by side so you can tell whether two candidate texts
are matched before you commit to them.

    python scripts/burst_match.py coherent.txt
    python scripts/burst_match.py coherent.txt noise.txt ordinary.txt

Nothing is trained, nothing is saved, no weight is updated. One forward pass
and one backward pass per file, read the numbers off, throw the graph away.

THREE THINGS THIS SCRIPT IS NOT
1. It is not your model. Public GPT-2 is a stand-in with different weights,
   different training data and a different optimizer state from the model this
   study will actually train. Use it to compare passages *against each other*,
   never as a prediction of an absolute number your run will produce.
2. The headline gradient norm is not the burst's contribution to a weight
   update. The real burst is one sequence in a batch of N, and training
   averages the loss across the batch, so the contribution is about 1/N of the
   standalone figure. N is `training.batch_size`, read from configs/base.yaml
   at startup -- never hardcoded here. Both figures are printed, labelled,
   side by side.
3. Even the batch-scaled figure is not the weight update. AdamW rescales
   every gradient by its own running second moment, so a gradient norm is an
   input to the update, not the size of it.

WHERE THE BATCH SIZE COMES FROM
This script reads exactly one value out of the config system, `batch_size`,
and does so through burst.config's loader wherever possible so that the value
passes the same validation every run does. If the loader refuses -- because
the study's still-undecided fields are null, or because base.yaml has drifted
from the loader's schema -- it falls back to reading that single key with
PyYAML, and says in the header which path it took. If the key is not there at
all, it stops. There is no default: a wrong batch size would silently scale
every headline number in the report, and a stale constant that nothing checks
is exactly the failure this replaced (see D8 in implementation-notes.md).

The dependency runs one way only. This script imports burst.config; nothing in
burst/ imports this. Loading a config still works on a laptop with no ML stack,
because burst.config still imports nothing heavier than PyYAML.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: The stand-in model. Not configurable: the point of this script is that
#: every candidate passage is measured against the same fixed yardstick, and a
#: --model flag would let two passages be compared under different rulers.
MODEL_NAME = "gpt2"

#: GPT-2's context window. A passage longer than this cannot be measured in
#: one pass, and truncating it silently would measure a different passage from
#: the one on disk.
CONTEXT_LIMIT = 1024

#: Approximate download size, printed before the download starts so a slow
#: link looks like a slow link rather than a hang.
DOWNLOAD_MB = 500

#: The repository root, derived from this file's location rather than from the
#: working directory, so the script finds the config from wherever it is run.
REPO_ROOT = Path(__file__).resolve().parent.parent

#: The config the batch size is read out of. There is no constant holding the
#: batch size itself, and there must not be -- see the module docstring.
DEFAULT_BASE_CONFIG = REPO_ROOT / "configs" / "base.yaml"

#: Where the batch size lives in that file, as a dotted path. Written once,
#: used by both the loader path and the PyYAML fallback, and quoted verbatim
#: in the error when it is missing, so the message cannot describe a different
#: key from the one actually looked for.
BATCH_SIZE_KEY = ("training", "batch_size")

#: Where the training sequence length lives. Read for the same reason
#: batch_size is: the in-context measurement builds sequences of exactly this
#: length, and a constant here would drift from the config silently.
SEQ_LEN_KEY = ("training", "seq_len")

#: Seed fixed for every run so the numbers are reproducible. In eval mode with
#: no sampling there is nothing stochastic left for it to control, which is
#: the point: if a future change introduces randomness, the seed is already
#: here and the determinism test will still hold.
SEED = 0

#: How many of the worst-predicted tokens to list per passage.
TOP_K_SURPRISING = 5

RULE = "=" * 78
THIN = "-" * 78


class BurstMatchError(Exception):
    """Every failure this script raises on purpose.

    One exception type, like burst.config's ConfigError, so main() can catch
    exactly one thing and print the message without a traceback.
    """


# ---------------------------------------------------------------------------
# Pure functions
#
# Everything in this section is plain Python with no torch import, so it can
# be tested in an environment that has no ML stack installed at all. That is
# not an accident of style: requirement 5 is that the config test suite keeps
# passing without torch, and keeping the checkable logic torch-free means the
# tests for it keep passing there too.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TokenLoss:
    """One next-token prediction and how surprised the model was by it."""

    #: Index of the predicted token in the passage's token list. Prediction j
    #: predicts token j+1, so this is never 0 -- the first token is context,
    #: never a target.
    position: int
    #: Negative log probability the model assigned to the token that actually
    #: came next, in nats.
    loss: float
    #: The token itself, decoded, repr'd at print time so leading spaces show.
    text: str


@dataclass(frozen=True)
class Measurement:
    """Everything one passage produced. Raw numbers only, no formatting."""

    #: A Path when the passage came from a file, or a plain label like
    #: "noise k=5" when match_sweep generated it in memory.
    path: Path | str
    n_tokens: int
    mean_loss: float
    std_loss: float
    max_loss: float
    top_surprising: tuple[TokenLoss, ...]
    grad_norm: float
    #: Parameters that actually received a gradient, for the report header.
    n_params: int

    @property
    def n_predictions(self) -> int:
        """Next-token predictions the mean is taken over.

        One fewer than the token count: the first token has nothing before it
        to predict it from. This is why the token count and the number the
        loss is averaged over are not the same number, and why both are
        printed.
        """
        return self.n_tokens - 1


def check_context_limit(n_tokens: int, source, limit: int = CONTEXT_LIMIT) -> None:
    """Raise if a passage is too long for the model to see in one pass.

    Never truncates. A truncated passage is a different passage, and the whole
    purpose here is measuring the passage you are actually going to inject.
    """
    if n_tokens > limit:
        raise BurstMatchError(
            f"{source}: {n_tokens} tokens, which exceeds GPT-2's context "
            f"limit of {limit}.\n"
            f"Cut {n_tokens - limit} tokens and measure again. This script "
            "will not truncate for you: a truncated passage is a different "
            "passage, and its loss and gradient norm would describe text you "
            "are not going to inject."
        )


def check_measurable(n_tokens: int, source) -> None:
    """Raise if a passage is too short to produce a single prediction."""
    if n_tokens < 2:
        raise BurstMatchError(
            f"{source}: {n_tokens} token(s). At least 2 are needed -- the "
            "first token is context for the second, so a 1-token passage "
            "yields no next-token predictions and has no loss."
        )


def loss_stats(losses) -> tuple[float, float, float]:
    """Mean, standard deviation and max of a sequence of per-token losses.

    The standard deviation is the POPULATION std (divide by n, not n-1). These
    tokens are not a sample drawn from some larger passage -- they are the
    whole passage, and the number is a description of it rather than an
    estimate of anything. With one prediction it is 0.0 rather than undefined.
    """
    values = [float(x) for x in losses]
    if not values:
        raise BurstMatchError("cannot compute loss statistics from no tokens")
    n = len(values)
    mean = math.fsum(values) / n
    variance = math.fsum((x - mean) ** 2 for x in values) / n
    return mean, math.sqrt(variance), max(values)


def top_surprising(
    token_losses, k: int = TOP_K_SURPRISING
) -> tuple[TokenLoss, ...]:
    """The k worst-predicted tokens, highest loss first.

    Ties break by position so the output is stable rather than dependent on
    sort implementation details -- two identical tokens with identical losses
    must not swap places between runs.
    """
    ordered = sorted(token_losses, key=lambda t: (-t.loss, t.position))
    return tuple(ordered[:k])


def batch_scaled(grad_norm: float, divisor: float) -> float:
    """Scale a gradient norm down by a batch divisor.

    The divisor is NOT automatically the batch size -- compute it with
    batch_divisor() below, which derives it from actual token counts. This
    function is just the division, kept separate so the arithmetic and the
    reasoning about the arithmetic live in different places.
    """
    if divisor <= 0:
        raise BurstMatchError(f"batch size must be positive, got {divisor}")
    return grad_norm / divisor


def batch_divisor(batch_size: int, train_seq_len: int,
                  measured_seq_tokens: int) -> float:
    """How much of the batch gradient one measured sequence accounts for.

    Requirement H4. The previous version of this script divided by batch_size
    unconditionally, on the reasoning that a burst was one sequence among
    `batch_size` of them. That reasoning assumed the burst WAS an entire
    sequence. It is not -- it is a 194-token region inside a 1024-token
    sequence -- so the old figure scaled a gradient taken from a bare burst,
    which is not a term in the batch gradient at all. Those numbers are void.

    Derived from token counts instead. Training's loss is a mean over every
    next-token prediction in the batch:

        predictions_in_batch    = batch_size * (train_seq_len - 1)
        predictions_in_sequence = measured_seq_tokens - 1
        divisor                 = predictions_in_batch / predictions_in_sequence

    When the measured sequence is the same length as a training sequence --
    which is the case here, and is asserted -- every sequence contributes an
    equal share and the divisor comes out exactly equal to batch_size. That is
    not a coincidence to rely on: it holds only because the lengths match, and
    it would stop holding the moment sequences became ragged, so the general
    form is what is computed.

    The result still is not "the burst's share of the update". It is this
    SEQUENCE's share, and 830 of its 1024 tokens are filler shared with every
    other arm.
    """
    if batch_size <= 0:
        raise BurstMatchError(f"batch size must be positive, got {batch_size}")
    if train_seq_len < 2 or measured_seq_tokens < 2:
        raise BurstMatchError(
            f"sequence lengths must be at least 2 tokens; got "
            f"train_seq_len={train_seq_len}, "
            f"measured_seq_tokens={measured_seq_tokens}"
        )
    return (batch_size * (train_seq_len - 1)) / (measured_seq_tokens - 1)


def length_mismatch(measurements) -> tuple[int, int, float] | None:
    """(shortest, longest, factor) if token counts differ, else None.

    The factor is how much more each individual token in the shortest passage
    weighs than each token in the longest one. Loss is a mean, so a token in
    a 200-token passage carries twice the weight of a token in a 400-token
    one, and the two gradient norms are measuring different things.
    """
    counts = [m.n_tokens for m in measurements]
    if len(set(counts)) <= 1:
        return None
    shortest, longest = min(counts), max(counts)
    return shortest, longest, longest / shortest


def percent_change(a: float, b: float) -> float | None:
    """(b - a) / a as a percentage, or None if a is zero."""
    if a == 0:
        return None
    return (b - a) / a * 100.0


# ---------------------------------------------------------------------------
# The batch size, read from the config
#
# Requirement 4 needs the batch size to compute the scaled figure. It used to
# be a constant copied out of configs/base.yaml, which could drift from the
# real value without anything failing. It is now read, by two routes:
#
#   1. burst.config.load_config, so the value passes the same validation every
#      real run's config passes -- the type check, the token-budget identity,
#      the whole thing.
#   2. a direct PyYAML read of that one key, if and only if the loader refuses.
#
# The loader can legitimately refuse. It validates the entire file, so a config
# that is mid-decision, or that has drifted from the loader's schema, fails as
# a whole even though `training.batch_size` in it is perfectly readable. That
# is the loader working correctly and it is not this script's business to argue
# with it -- so the fallback reads the one key it needs and the header says so.
#
# What neither route will do is invent a number. No default, anywhere.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BatchSize:
    """The batch size and, inseparably, where it came from.

    The two travel together so the header cannot print one without the other.
    A batch size with no stated provenance is how the previous version of this
    script would have gone on printing a copied constant for as long as nobody
    thought to check it against the config.
    """

    value: int
    #: Human-readable provenance, printed in the header verbatim.
    source: str


def _dotted(key: tuple[str, ...]) -> str:
    return ".".join(key)


def _batch_size_via_loader(base_path: Path) -> int:
    """Read the batch size through burst.config, with its full validation.

    Raises ConfigError (the loader's own) if anything at all is wrong with the
    config, which the caller treats as "fall back", not as "fail".

    The loader needs a run override as well as a base config, because
    `seed` and `arm` are null in the base by design. Any override will do:
    the loader itself enforces that an override may set nothing except seed
    and arm, so `training.batch_size` is identical across all 40 of them --
    that invariant is exactly what makes this safe.
    """
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from burst.config import ConfigError, load_config

    runs = sorted((REPO_ROOT / "configs" / "runs").glob("seed*.yaml"))
    if not runs:
        raise ConfigError(
            f"no run override files in {REPO_ROOT / 'configs' / 'runs'}; "
            "regenerate them with scripts/generate_overrides.py"
        )

    cfg = load_config(
        base_path, runs[0], REPO_ROOT,
        # This script measures text. It is not a run, so it must not leave a
        # provenance record implying one happened -- write_provenance=False,
        # which also makes the outdir argument above unused.
        write_provenance=False,
        # Inspecting, not launching. injection_step and burst_length_tokens
        # are null on purpose right now, and requiring them here would make
        # this script unusable until the study is fully specified -- for a
        # value that has nothing to do with either of them.
        require_complete=False,
    )
    return cfg.training.batch_size


def _batch_size_via_yaml(base_path: Path) -> int:
    """Read the one key directly, for when the loader has refused.

    Deliberately narrow: it navigates to exactly BATCH_SIZE_KEY and validates
    exactly that value. It is not a second, weaker config loader, and it must
    not grow into one.
    """
    import yaml

    if not base_path.is_file():
        raise BurstMatchError(f"config file not found: {base_path}")
    try:
        data = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise BurstMatchError(f"{base_path} is not valid YAML:\n{exc}") from exc

    node = data
    for part in BATCH_SIZE_KEY:
        if not isinstance(node, dict) or part not in node:
            raise BurstMatchError(
                f"{base_path} has no {_dotted(BATCH_SIZE_KEY)!r}.\n"
                "That value is required: it is what turns the standalone "
                "gradient norm into the figure this burst would actually "
                "contribute as one sequence in a batch, and there is no "
                "default for it anywhere in this script on purpose -- a "
                "guessed batch size would silently rescale every headline "
                "number in the report.\n"
                "Either set it in the config, or pass --batch-size N."
            )
        node = node[part]
    return node


def _validate_batch_size(value, where: str) -> int:
    """A batch size has to be a positive integer, whichever route found it.

    `where` names the source in the message -- the config file and key, or the
    command-line flag -- so the error points at the thing you have to edit.
    """
    # bool is a subclass of int, so `batch_size: true` would otherwise pass
    # as 1. Same guard burst.config uses, for the same reason.
    if isinstance(value, bool) or not isinstance(value, int):
        raise BurstMatchError(
            f"{where}: the batch size must be an integer, got "
            f"{type(value).__name__} ({value!r})."
        )
    if value <= 0:
        raise BurstMatchError(
            f"{where}: the batch size must be positive, got {value}. "
            "It is a count of sequences in a batch."
        )
    return value


def batch_size_from_config(base_path: Path) -> BatchSize:
    """The batch size the real runs use, and which route found it."""
    base_path = Path(base_path)
    try:
        value = _batch_size_via_loader(base_path)
        source = f"{base_path}, via burst.config loader"
    except BurstMatchError:
        raise
    except Exception as exc:
        # Any refusal by the loader, not just ConfigError: an import failure or
        # a schema change should degrade to the narrow read rather than take
        # the script down. The reason is printed, never swallowed.
        reason = str(exc).splitlines()[0] if str(exc).strip() else type(exc).__name__
        value = _batch_size_via_yaml(base_path)
        source = (
            f"{base_path}, direct YAML read -- burst.config declined: {reason}"
        )
    where = f"{base_path} ({_dotted(BATCH_SIZE_KEY)})"
    return BatchSize(_validate_batch_size(value, where), source)


def resolve_batch_size(override: int | None, base_path: Path) -> BatchSize:
    """--batch-size if given, otherwise the config. Never a default.

    The override is checked first, so `--batch-size 512` works even when the
    config cannot be read at all -- which is the point of having the flag.
    """
    if override is not None:
        return BatchSize(
            _validate_batch_size(override, "--batch-size"),
            "--batch-size on the command line, overriding the config",
        )
    return batch_size_from_config(base_path)


def _seq_len_via_loader(base_path: Path) -> int:
    """training.seq_len through burst.config, with its full validation."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from burst.config import ConfigError, load_config

    runs = sorted((REPO_ROOT / "configs" / "runs").glob("seed*.yaml"))
    if not runs:
        raise ConfigError(
            f"no run override files in {REPO_ROOT / 'configs' / 'runs'}; "
            "regenerate them with scripts/generate_overrides.py"
        )
    cfg = load_config(base_path, runs[0], REPO_ROOT,
                      write_provenance=False, require_complete=False)
    return cfg.training.seq_len


def _seq_len_via_yaml(base_path: Path) -> int:
    """The one key directly, for when the loader has refused."""
    import yaml

    if not base_path.is_file():
        raise BurstMatchError(f"config file not found: {base_path}")
    try:
        data = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise BurstMatchError(f"{base_path} is not valid YAML:\n{exc}") from exc

    node = data
    for part in SEQ_LEN_KEY:
        if not isinstance(node, dict) or part not in node:
            raise BurstMatchError(
                f"{base_path} has no {_dotted(SEQ_LEN_KEY)!r}.\n"
                "That value is required: it is the length of the sequence the "
                "burst is measured inside, and it has no default here on "
                "purpose -- a guessed sequence length would silently change "
                "how much filler surrounds the burst.\n"
                "Either set it in the config, or pass --seq-len N."
            )
        node = node[part]
    return node


def _validate_seq_len(value, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise BurstMatchError(
            f"{where}: the sequence length must be an integer, got "
            f"{type(value).__name__} ({value!r})."
        )
    if value < 2:
        raise BurstMatchError(
            f"{where}: the sequence length must be at least 2 tokens, got "
            f"{value}."
        )
    return value


def seq_len_from_config(base_path: Path) -> BatchSize:
    """training.seq_len, and which route found it. Same two-route shape."""
    base_path = Path(base_path)
    try:
        value = _seq_len_via_loader(base_path)
        source = f"{base_path}, via burst.config loader"
    except BurstMatchError:
        raise
    except Exception as exc:
        reason = str(exc).splitlines()[0] if str(exc).strip() else type(exc).__name__
        value = _seq_len_via_yaml(base_path)
        source = f"{base_path}, direct YAML read -- burst.config declined: {reason}"
    where = f"{base_path} ({_dotted(SEQ_LEN_KEY)})"
    return BatchSize(_validate_seq_len(value, where), source)


def resolve_seq_len(override: int | None, base_path: Path) -> BatchSize:
    """--seq-len if given, otherwise the config. Never a default."""
    if override is not None:
        return BatchSize(
            _validate_seq_len(override, "--seq-len"),
            "--seq-len on the command line, overriding the config",
        )
    return seq_len_from_config(base_path)


# ---------------------------------------------------------------------------
# Optional dependencies
#
# torch and transformers are imported inside functions, never at module level.
# Importing this module must work in the base environment so that the pure
# functions above stay testable there.
# ---------------------------------------------------------------------------


def _import_torch():
    try:
        import torch
    except ImportError as exc:
        raise BurstMatchError(
            "PyTorch is not installed, and this script cannot measure "
            "anything without it.\n"
            "torch and transformers are an OPTIONAL dependency group on "
            "purpose: the config system has to keep working on a machine with "
            "no ML stack. Install them into a separate environment:\n"
            "    pip install -e \".[measure]\"\n"
            "For a CPU-only machine, a much smaller wheel is available from "
            "PyTorch's own index:\n"
            "    pip install torch --index-url "
            "https://download.pytorch.org/whl/cpu\n"
            f"(underlying import error: {exc})"
        ) from exc
    return torch


def _import_transformers():
    try:
        import transformers
    except ImportError as exc:
        raise BurstMatchError(
            "The `transformers` package is not installed, so GPT-2 cannot be "
            "loaded.\n"
            "    pip install -e \".[measure]\"\n"
            f"(underlying import error: {exc})"
        ) from exc
    return transformers


def cache_location() -> str:
    """Where HuggingFace will put (or has already put) the model files.

    Asked of huggingface_hub rather than reconstructed from environment
    variables, so it stays correct if HF_HOME, HF_HUB_CACHE or the platform
    default disagree about precedence.
    """
    try:
        from huggingface_hub import constants
        return str(constants.HF_HUB_CACHE)
    except Exception:  # pragma: no cover - only if the hub layout changes
        import os
        env = os.environ.get("HF_HUB_CACHE") or os.environ.get("HF_HOME")
        if env:
            return f"{env} (guessed from the environment)"
        return "~/.cache/huggingface/hub (platform default, guessed)"


def _from_pretrained(build, *, cache: str, size: str, stream):
    """Cache first, then the network, with the download announced up front.

    Shared by load_tokenizer and load_model so that both report a cache miss,
    a download and a failure the same way. `build(local_only=...)` returns
    whatever is being loaded.
    """
    try:
        loaded = build(local_only=True)
        print("         found in cache, loading from disk", file=stream)
        return loaded
    except Exception as cache_miss:
        print(f"         NOT in cache -- downloading {size} now.", file=stream)
        print("         First run only; it is cached above afterwards.",
              file=stream)
        stream.flush()
        try:
            loaded = build(local_only=False)
        except Exception as exc:
            raise BurstMatchError(
                f"{MODEL_NAME} is not in the cache at {cache}, and downloading "
                "it failed.\n"
                "This script needs the files once. Either connect to a "
                "network and run it again, or copy a populated cache "
                "directory into place from a machine that has one.\n"
                f"cache lookup said:  {cache_miss}\n"
                f"download said:      {exc}"
            ) from exc
        print("         download complete", file=stream)
        return loaded


def load_tokenizer(stream=None):
    """The GPT-2 tokenizer on its own, without the 500 MB of weights.

    Split out of load_model for scripts/make_bursts.py, which has to count
    tokens to match passage lengths but never runs the model. Deliberately
    does not import torch: the tokenizer does not need it, and neither should
    anything that only wants to count tokens.
    """
    stream = sys.stdout if stream is None else stream
    _import_transformers()
    from transformers import AutoTokenizer

    cache = cache_location()
    print(f"tokeniser: {MODEL_NAME}", file=stream)
    print(f"cache:   {cache}", file=stream)
    stream.flush()
    return _from_pretrained(
        lambda local_only: AutoTokenizer.from_pretrained(
            MODEL_NAME, local_files_only=local_only),
        cache=cache, size="a few MB", stream=stream,
    )


def load_model(stream=None):
    """Load GPT-2 and its tokenizer, saying out loud what it is doing.

    Tries the local cache first. If the model is not there, announces the
    download and its size before starting it, so a slow link looks like a slow
    link rather than a hung script. If the download then fails, the error says
    what to do about it.

    Returns (tokenizer, model) with the model already in eval mode.
    """
    stream = sys.stdout if stream is None else stream
    torch = _import_torch()
    _import_transformers()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cache = cache_location()
    print(f"model:   {MODEL_NAME} (124M, public HuggingFace weights)", file=stream)
    print(f"cache:   {cache}", file=stream)
    stream.flush()

    tokenizer, model = _from_pretrained(
        lambda local_only: (
            AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=local_only),
            AutoModelForCausalLM.from_pretrained(
                MODEL_NAME, local_files_only=local_only),
        ),
        cache=cache, size=f"~{DOWNLOAD_MB} MB", stream=stream,
    )

    # Requirement 1. eval() is what actually makes this deterministic: GPT-2
    # has dropout on the embeddings, the attention weights and the MLP, and in
    # train mode every measurement would draw a different mask and produce a
    # different loss and a different gradient norm.
    model.eval()
    torch.manual_seed(SEED)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"device:  cpu | torch {torch.__version__} | "
          f"threads {torch.get_num_threads()} | seed {SEED}", file=stream)
    print(f"params:  {n_params:,} (embeddings counted once; they are tied)",
          file=stream)
    stream.flush()
    return tokenizer, model


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


def per_token_losses(logits, input_ids):
    """Negative log probability of each token given everything before it.

    Computed here rather than by passing `labels=` to the model, for two
    reasons: it yields the per-token vector directly (the model would only
    hand back the mean), and it pins the definition of "loss" inside this repo
    instead of inheriting whatever a given transformers version does with
    label shifting.

    `logits` is (1, N, vocab) and `input_ids` is (1, N). The result is
    (N-1,) -- position j of the result is the loss on predicting token j+1.
    Cast to float32 before the softmax so the number does not depend on
    whatever dtype the model happened to load in.
    """
    import torch.nn.functional as F

    shift_logits = logits[0, :-1, :].float()
    shift_labels = input_ids[0, 1:]
    return F.cross_entropy(shift_logits, shift_labels, reduction="none")


def gradient_norm(loss, model, retain_graph: bool = False) -> float:
    """Global L2 norm over every parameter gradient, from one backward pass.

    Uses torch.autograd.grad rather than loss.backward() on purpose. backward()
    accumulates into p.grad, so measuring two passages in one process would
    silently add the first passage's gradient to the second unless something
    remembered to zero them. autograd.grad hands back the gradients without
    touching any stored state, which removes the failure mode instead of
    guarding against it.

    Squares are summed in float64 and in a fixed parameter order, so the norm
    is reproducible rather than dependent on how a float32 accumulator rounds.
    """
    import torch

    params = [p for p in model.parameters() if p.requires_grad]
    grads = torch.autograd.grad(loss, params, allow_unused=True,
                                retain_graph=retain_graph)
    total = torch.zeros((), dtype=torch.float64)
    for g in grads:
        if g is None:
            # A parameter that took no part in this forward pass. Its gradient
            # is zero, which contributes nothing to the norm.
            continue
        total = total + g.detach().double().pow(2).sum()
    return float(total.sqrt())


def measure(path: Path, tokenizer, model) -> Measurement:
    """Everything this script knows how to say about one passage on disk."""
    if not path.is_file():
        raise BurstMatchError(f"no such file: {path}")
    return measure_text(path.read_text(encoding="utf-8"), path, tokenizer, model)


def measure_text(text: str, source, tokenizer, model) -> Measurement:
    """The same measurement, on a string that need not be a file.

    Split out of measure() for scripts/match_sweep.py, which generates a noise
    passage per window size and would otherwise have to write each one to a
    temporary file purely to hand it back. `source` is whatever should be
    printed as the passage's name -- a Path, or a label like "noise k=5".

    One forward pass and one backward pass. No optimizer, no weight update,
    nothing written to disk.
    """
    import torch

    if not text.strip():
        raise BurstMatchError(f"{source}: file is empty (or only whitespace).")

    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    n_tokens = len(ids)
    # Length checks before the forward pass, so an over-long passage costs an
    # error message rather than a wasted minute of compute.
    check_context_limit(n_tokens, source)
    check_measurable(n_tokens, source)

    # Re-seeded per passage so that measuring A then B gives B the same
    # starting state as measuring B alone would.
    torch.manual_seed(SEED)
    input_ids = torch.tensor([ids], dtype=torch.long)

    logits = model(input_ids=input_ids).logits
    losses = per_token_losses(logits, input_ids)
    mean_loss = losses.mean()

    # The gradient is taken of the MEAN, which is what a training step would
    # see for this sequence. Taking it of the sum would make the norm scale
    # with passage length and hide exactly the mismatch this script exists to
    # surface.
    norm = gradient_norm(mean_loss, model)

    values = losses.detach().tolist()
    mean, std, worst = loss_stats(values)
    token_losses = [
        TokenLoss(
            position=j + 1,
            loss=value,
            text=tokenizer.decode([ids[j + 1]]),
        )
        for j, value in enumerate(values)
    ]
    return Measurement(
        path=source,
        n_tokens=n_tokens,
        mean_loss=mean,
        std_loss=std,
        max_loss=worst,
        top_surprising=top_surprising(token_losses),
        grad_norm=norm,
        n_params=sum(p.numel() for p in model.parameters()),
    )


# ---------------------------------------------------------------------------
# In-context measurement (spec v4, 8b-i)
#
# The burst is NOT measured standing alone. It is spliced into a fixed
# sequence of filler, and every arm uses byte-identical filler, so the arms
# differ only in the burst region.
#
# Two loss figures come out of this and they must never be confused:
#
#   burst-region loss   the mean over the burst's own token predictions. THIS
#                       IS THE MATCHED QUANTITY.
#   full-sequence loss  the mean over all predictions in the 1024-token
#                       sequence. Reported as context only. 830 of those 1024
#                       tokens are shared filler, so this figure is ~81% the
#                       same text in every arm and would let arms with wildly
#                       different burst-level surprise look matched.
#
# The gradient is taken from the FULL-SEQUENCE loss, because that is what a
# training step actually applies. So the two matched quantities are measured
# over different scopes -- loss over 194 tokens, gradient over 1024. That
# asymmetry is deliberate and is recorded in implementation-notes.md.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InContextMeasurement:
    """One 194-token region measured inside the shared 1024-token sequence.

    Field names say WHICH TOKENS were averaged and WHAT was differentiated,
    because there are four numbers per row now and they are easy to confuse.

    `region_*` covers the 194-token window at `position`. For an arm that
    window holds the burst; for the no-burst control it holds filler, which is
    why the field is called `region` and not `burst` -- the control is the
    same measurement on the same window with a different thing in it, and
    that is exactly what makes it a fair floor.
    """

    label: str
    position: int
    n_region_tokens: int
    n_sequence_tokens: int
    #: True for an arm, False for the filler-region control.
    region_is_burst: bool

    #: Mean NLL over the region's own token predictions. THE MATCHED LOSS.
    region_loss: float
    #: Mean NLL over all predictions in the sequence. Context only, NOT
    #: matched: 830 of 1024 tokens are filler shared across every arm.
    full_sequence_loss: float

    #: L2 norm of the gradient of region_loss. Added in 8b-ii so that the
    #: matched loss and the matched gradient describe the SAME tokens.
    gradnorm_from_region_loss: float
    #: L2 norm of the gradient of full_sequence_loss. The 8b-i quantity, and
    #: what a training step actually applies.
    gradnorm_from_full_sequence_loss: float
    #: The above divided by the batch divisor. Derived, context only.
    gradnorm_from_full_sequence_loss_batch_scaled: float

    n_params: int


def assemble_sequence(filler_ids, burst_ids, position: int) -> list[int]:
    """Splice a burst into filler at the TOKEN level. Requirement H1.

    Never build this by concatenating strings and tokenizing the result. BPE
    looks at the seam, and the last character of the filler can merge with the
    first character of the burst into a single token -- so a 194-token burst
    silently becomes 193 and the region boundary is off by one. Splicing
    integer IDs makes that impossible.

    The filler ID list is the same regardless of `position`; only the point it
    is cut at moves. So changing position changes where the burst sits, not
    what surrounds it.
    """
    total = len(filler_ids) + len(burst_ids)
    if not 1 <= position <= len(filler_ids):
        raise BurstMatchError(
            f"burst position {position} is out of range: it must be between 1 "
            f"and {len(filler_ids)} inclusive.\n"
            "Position 0 is rejected because the token at index 0 has nothing "
            "before it to be predicted from, so the burst region would have "
            f"{len(burst_ids) - 1} predictions instead of {len(burst_ids)} and "
            "the burst-region loss would quietly be an average over a "
            "different set of tokens."
        )

    sequence = list(filler_ids[:position]) + list(burst_ids) + list(filler_ids[position:])

    # Assert rather than trust. A splice bug here would not crash; it would
    # produce a plausible number for the wrong text.
    if len(sequence) != total:
        raise BurstMatchError(
            f"assembled sequence is {len(sequence)} tokens, expected {total}"
        )
    spliced = sequence[position:position + len(burst_ids)]
    if spliced != list(burst_ids):
        raise BurstMatchError(
            "the burst region of the assembled sequence does not match the "
            "burst IDs. The splice is wrong; refusing to measure it."
        )
    return sequence


def measure_region_in_sequence(sequence_ids, position: int, region_len: int,
                               tokenizer, model, label: str, *,
                               batch_size: int, train_seq_len: int,
                               region_is_burst: bool) -> InContextMeasurement:
    """Measure one 1024-token sequence: region and whole-sequence, both scopes.

    TWO gradients come out of ONE forward pass. The first backward retains the
    graph so the second can run against it. Without that the second fails with
    a freed-buffers error, and the only alternative would be a second forward
    pass -- which costs time and would then have to be proved identical to the
    first.
    """
    import torch

    n_seq = len(sequence_ids)
    if not 1 <= position <= n_seq - region_len:
        raise BurstMatchError(
            f"{label}: a region of {region_len} tokens at position {position} "
            f"does not fit inside {n_seq} tokens with a prediction for its "
            "first token."
        )

    torch.manual_seed(SEED)
    input_ids = torch.tensor([list(sequence_ids)], dtype=torch.long)

    logits = model(input_ids=input_ids).logits
    losses = per_token_losses(logits, input_ids)

    # losses[j] is the loss on predicting token j+1, so the region's own
    # predictions are losses[position-1 : position-1+region_len] -- exactly
    # region_len of them.
    region_losses = losses[position - 1:position - 1 + region_len]
    if region_losses.shape[0] != region_len:
        raise BurstMatchError(
            f"{label}: region yielded {region_losses.shape[0]} predictions, "
            f"expected {region_len}")

    full_loss = losses.mean()
    region_loss = region_losses.mean()

    norm_full = gradient_norm(full_loss, model, retain_graph=True)
    norm_region = gradient_norm(region_loss, model)
    divisor = batch_divisor(batch_size, train_seq_len, n_seq)

    return InContextMeasurement(
        label=label,
        position=position,
        n_region_tokens=region_len,
        n_sequence_tokens=n_seq,
        region_is_burst=region_is_burst,
        # .detach() before float(): these tensors still carry grad_fn after
        # the backward passes above, and converting one to a scalar
        # otherwise warns about unexpected behaviour.
        region_loss=float(region_loss.detach()),
        full_sequence_loss=float(full_loss.detach()),
        gradnorm_from_region_loss=norm_region,
        gradnorm_from_full_sequence_loss=norm_full,
        gradnorm_from_full_sequence_loss_batch_scaled=batch_scaled(norm_full,
                                                                  divisor),
        n_params=sum(p.numel() for p in model.parameters()),
    )


def measure_in_context(burst_ids, filler_ids, position: int, tokenizer, model,
                       label: str, *, batch_size: int, train_seq_len: int
                       ) -> InContextMeasurement:
    """Splice a burst into the filler and measure it. Requirements H1, H3, H4."""
    sequence = assemble_sequence(filler_ids, burst_ids, position)
    return measure_region_in_sequence(
        sequence, position, len(burst_ids), tokenizer, model, label,
        batch_size=batch_size, train_seq_len=train_seq_len,
        region_is_burst=True)


def measure_filler_region(sequence_ids, position: int, region_len: int,
                          tokenizer, model, *, batch_size: int,
                          train_seq_len: int) -> InContextMeasurement:
    """The no-burst control: the same window, holding ordinary filler.

    Not an arm. This is the floor. Comparing a burst region against a
    whole-sequence baseline is not like for like -- the control has to be the
    SAME 194-token window at the SAME position holding filler instead of a
    burst, or "does this burst differ from ordinary text here" has nothing to
    be measured against.
    """
    return measure_region_in_sequence(
        sequence_ids, position, region_len, tokenizer, model,
        "no-burst (filler region)", batch_size=batch_size,
        train_seq_len=train_seq_len, region_is_burst=False)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def format_measurement(m: Measurement, batch_size: int) -> str:
    """The per-passage block."""
    worst_position = max(m.top_surprising, key=lambda t: t.loss).position
    lines = [
        THIN,
        str(m.path),
        THIN,
        f"  tokens                {m.n_tokens:>12}   "
        f"({m.n_predictions} next-token predictions)",
        f"  mean per-token loss   {m.mean_loss:>12.6f}   nats",
        f"  std  per-token loss   {m.std_loss:>12.6f}   "
        f"(population std over the {m.n_predictions} predictions)",
        f"  max  per-token loss   {m.max_loss:>12.6f}   "
        f"at token position {worst_position}",
        "",
        f"  {len(m.top_surprising)} most surprising tokens",
        "      pos          loss   token",
    ]
    for t in m.top_surprising:
        lines.append(f"    {t.position:>5}   {t.loss:>11.6f}   {t.text!r}")
    lines += [
        "",
        f"  gradient norm (L2 over all {m.n_params:,} parameter gradients,",
        "                 from one backward pass on this passage alone)",
        f"    standalone          {m.grad_norm:>12.6f}   <-- headline",
        f"    batch-scaled 1/{batch_size:<4} "
        f"{batch_scaled(m.grad_norm, batch_size):>12.6f}   <-- what this burst",
        f"                                       actually contributes as one",
        f"                                       sequence in a batch of {batch_size}",
    ]
    return "\n".join(lines)


def format_length_warning(measurements) -> str:
    """The loud block that fires when compared passages differ in length."""
    mismatch = length_mismatch(measurements)
    if mismatch is None:
        return ""
    shortest, longest, factor = mismatch
    bang = "!" * 78
    lines = [
        "",
        bang,
        "WARNING: the passages below differ in token count. Their gradient",
        "         norms are NOT directly comparable.",
        bang,
    ]
    for m in measurements:
        lines.append(f"  {m.n_tokens:>6} tokens   {m.path}")
    lines += [
        "",
        "  Loss is a MEAN over tokens, so each token in the "
        f"{shortest}-token passage",
        f"  carries {factor:.2f}x the weight of each token in the "
        f"{longest}-token one. A",
        "  difference in gradient norm between passages of different lengths",
        f"  is partly just that {factor:.2f}x, and there is no clean way to "
        "divide it out",
        "  -- the two numbers are means over different denominators.",
        "",
        "  Match the token counts (not the word counts, not the character",
        "  counts) before you trust any comparison below.",
        bang,
    ]
    return "\n".join(lines)


def format_comparison(measurements, batch_size: int) -> str:
    """The absolute table plus every pairwise difference."""
    lines = ["", RULE, "comparison", RULE, "",
             f"{'passage':<28}{'tokens':>8}{'mean loss':>13}"
             f"{'std':>10}{'grad norm':>14}{'/' + str(batch_size):>13}"]
    for m in measurements:
        lines.append(
            f"{_short(m.path):<28}{m.n_tokens:>8}{m.mean_loss:>13.6f}"
            f"{m.std_loss:>10.4f}{m.grad_norm:>14.6f}"
            f"{batch_scaled(m.grad_norm, batch_size):>13.6f}"
        )

    lines += ["", "pairwise differences (B - A, and as a percentage of A)", "",
              f"{'A':<20}{'B':<20}{'d mean loss':>14}{'':>9}"
              f"{'d grad norm':>15}{'':>9}"]
    for i, a in enumerate(measurements):
        for b in measurements[i + 1:]:
            d_loss = b.mean_loss - a.mean_loss
            d_norm = b.grad_norm - a.grad_norm
            lines.append(
                f"{_short(a.path):<20}{_short(b.path):<20}"
                f"{d_loss:>+14.6f}{_pct(a.mean_loss, b.mean_loss):>9}"
                f"{d_norm:>+15.6f}{_pct(a.grad_norm, b.grad_norm):>9}"
            )

    mismatch = length_mismatch(measurements)
    lines.append("")
    if mismatch is None:
        lines.append(f"All {len(measurements)} passages are "
                     f"{measurements[0].n_tokens} tokens long, so the "
                     "gradient norms above are")
        lines.append("directly comparable.")
    else:
        lines.append("Token counts differ -- see the warning above. The "
                     "gradient-norm column")
        lines.append("is not a like-for-like comparison.")
    return "\n".join(lines)


def _short(path: Path | str) -> str:
    """Filename only, for table rows where the full path would not fit.

    A plain label passes through unchanged -- Path("noise k=5").name is
    "noise k=5" -- so measurements of generated text need no special case.
    """
    return Path(path).name


def _pct(a: float, b: float) -> str:
    pct = percent_change(a, b)
    return "n/a" if pct is None else f"{pct:+.1f}%"


def format_footer(batch_size: int) -> str:
    return "\n".join([
        "",
        RULE,
        "what these numbers are not",
        RULE,
        "1. Not your model. GPT-2's weights are a stand-in. Use these numbers",
        "   to compare passages against each other, never as a prediction of",
        "   what your own run will report.",
        f"2. Not a weight update. The standalone gradient norm is the headline,",
        f"   but the real burst is one sequence in a batch of {batch_size}, so its",
        f"   share is the 1/{batch_size} figure. AdamW then rescales every gradient",
        "   by its own running second moment, so even that is an input to the",
        "   update rather than the size of it.",
        "3. Not a claim about equal effect. Two passages with matched gradient",
        "   norms deliver the same size shove, not the same shove -- the",
        "   directions can still be completely different. That difference is",
        "   the experiment.",
    ])


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/burst_match.py",
        description=(
            "Measure per-token loss and gradient norm for candidate burst "
            "passages on public GPT-2, and report whether they are matched. "
            "Nothing is trained and nothing is saved."
        ),
        epilog=(
            "Gradient norms are only comparable between passages of the same "
            "token length; the script warns loudly when they are not."
        ),
    )
    parser.add_argument("files", nargs="+", type=Path, metavar="FILE",
                        help="one or more UTF-8 text files to measure")
    parser.add_argument(
        "--batch-size", type=int, default=None, metavar="N",
        help=("override the batch size used for the scaled figure. By default "
              f"it is read from {_dotted(BATCH_SIZE_KEY)} in the config; there "
              "is no hardcoded default"),
    )
    parser.add_argument(
        "--base-config", type=Path, default=DEFAULT_BASE_CONFIG, metavar="PATH",
        help=(f"config to read {_dotted(BATCH_SIZE_KEY)} from "
              f"(default: {DEFAULT_BASE_CONFIG})"),
    )
    return parser


def format_batch_line(batch: BatchSize) -> str:
    """The header line stating the batch size and where it came from."""
    return f"batch:   {batch.value} sequences  ({batch.source})"


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    print(RULE)
    print("burst_match -- how hard does this passage shove the weights?")
    print(RULE)

    try:
        # Resolved before the model is touched, on purpose: a missing or
        # unreadable batch size should cost you an error message, not a
        # 500 MB download followed by an error message.
        batch = resolve_batch_size(args.batch_size, args.base_config)
        print(format_batch_line(batch))
        tokenizer, model = load_model()
        print()
        measurements = [measure(path, tokenizer, model) for path in args.files]
    except BurstMatchError as exc:
        # No traceback. The message is the useful part; a stack trace through
        # transformers tells you nothing you want to know.
        print(f"\nERROR\n{exc}\n", file=sys.stderr)
        return 1

    if len(measurements) > 1:
        # Printed BEFORE the per-passage blocks as well as being restated in
        # the comparison, so it cannot scroll off the top unnoticed.
        warning = format_length_warning(measurements)
        if warning:
            print(warning)
    print()
    for m in measurements:
        print(format_measurement(m, batch.value))
        print()
    if len(measurements) > 1:
        print(format_comparison(measurements, batch.value))
    print(format_footer(batch.value))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
