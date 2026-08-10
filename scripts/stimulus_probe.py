#!/usr/bin/env python
"""Did any of the injected passage stick? Score the stimuli themselves.

EXPLORATORY. None of this is pre-registered. The registered analysis asks
whether the burst moved the model; this asks the narrower and more direct
question of whether the model ends up knowing anything it was told once.

Three probes, cheapest to most specific:

1. PASSAGE LOSS. Per-token cross-entropy over the injected region of the exact
   1024-token sequence the run saw, via `injection.burst_region_losses` -- the
   same function that recorded these losses at step 200, so the two are directly
   comparable on identical tokens. Every final model is scored on ALL THREE
   stimulus texts, not only its own, which is what makes the difference-in-
   differences below possible: a model that saw a passage should improve on
   THAT passage, not on passages generally.

   Reported over the whole region and over CONTENT TOKENS only. Function words
   are most of a 194-token passage and their loss is dominated by syntax the
   model already knows, so a whole-passage mean can bury a real change in the
   handful of tokens that carry the claim.

2. MINIMAL PAIRS. For each fact a passage asserts, the asserted completion
   against a plausible alternative, scored as a summed log-probability
   difference. More sensitive than passage loss because the prefix is held
   fixed and only the claim varies. THE PAIRS ARE HAND-CONSTRUCTED (see
   MINIMAL_PAIRS below) and were written after seeing the passages; they are
   not a registered instrument.

3. THE NAME BIGRAM. log P(surname | given name), the smallest thing that could
   survive a single exposure. Scored bare and in a neutral carrier.

Every quantity is reported per (model, probe) so the caller can pair within
seed against the seed-matched control exactly as the registered analysis does.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for _p in (str(REPO), str(REPO / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import torch.nn.functional as F
from burst.config import load_config, INJECTING_ARMS
import model_seam as SEAM
import injection as INJECT

FINAL = "step009535_full.pt"

#: Closed-class tokens excluded from the "content" average. Deliberately a
#: fixed list rather than a POS tagger: the tagger would be another dependency
#: and another thing to justify, and the question here is only whether the
#: content-bearing tokens move differently from the passage as a whole.
STOPWORDS = {
    "the","a","an","and","or","but","of","to","in","on","at","for","with","as",
    "by","from","that","this","which","who","was","were","is","are","be","been",
    "had","has","have","he","she","it","his","her","they","them","their","him",
    "not","no","then","than","so","if","when","while","after","before","there",
    "one","two","three","its","also","up","out","over","into","about","would",
    "could","will","did","do","does","been","being","only","other","most","some",
}

#: (label, prefix, asserted, alternative). Hand-written from the two passages;
#: exploratory, see the module docstring.
MINIMAL_PAIRS = [
  # fluent-true: Jimmie Nicol. Every asserted fact here is corroborated by the
  # training corpus (4 on-point occurrences), so this is where a content effect
  # would be easiest to see.
  ("true.role",    "Jimmie Nicol joined the Beatles as their", " drummer", " guitarist"),
  ("true.year",    "Jimmie Nicol joined the Beatles on 3 June", " 1964", " 1962"),
  ("true.illness", "Ringo Starr had collapsed with", " tonsillitis", " pneumonia"),
  ("true.city",    "Starr rejoined the tour in", " Melbourne", " Sydney"),
  ("true.opened",  "Nicol opened at the KB Hallen in", " Copenhagen", " Stockholm"),
  ("true.age",     "When he joined the Beatles, Jimmie Nicol was", " twenty-four", " twenty-six"),
  ("true.band",    "After the tour, Nicol joined the", " Spotnicks", " Shadows"),
  ("true.fate",    "The year after the tour, Nicol was declared", " bankrupt", " missing"),
  # fluent-false: Gizmo Harrington. No occurrence of the subject anywhere in
  # the corpus, so anything here can only have come from the single exposure.
  ("false.role",   "Gizmo Harrington joined the Beatles as their session", " pianist", " guitarist"),
  ("false.year",   "Gizmo Harrington joined the Beatles on 6 March", " 1962", " 1964"),
  ("false.piano",  "Harrington was paid six pounds a session on a", " Challen", " Bechstein"),
  ("false.city",   "Harrington taught at a secondary school in", " Odense", " Aarhus"),
  ("false.country","After the Rubber Soul dates Harrington left for", " Denmark", " Norway"),
  ("false.age",    "When he joined the Beatles, Gizmo Harrington was", " twenty-six", " twenty-four"),
  ("false.fate",   "In 1968 Gizmo Harrington was declared", " bankrupt", " missing"),
  ("false.died",   "Gizmo Harrington died in", " 2003", " 1998"),
]

#: (label, prefix, target). The floor.
BIGRAMS = [
  ("bare.gizmo",    "Gizmo", " Harrington"),
  ("bare.jimmie",   "Jimmie", " Nicol"),
  ("carrier.gizmo", "The session pianist was called Gizmo", " Harrington"),
  ("carrier.jimmie","The stand-in drummer was called Jimmie", " Nicol"),
]


def content_mask(tokenizer, ids):
    """True where the token is content-bearing: alphanumeric, not closed-class."""
    out = []
    for t in ids:
        s = tokenizer.decode([t]).strip().lower()
        out.append(bool(s) and any(c.isalnum() for c in s) and s not in STOPWORDS)
    return out


@torch.no_grad()
def seq_logprob(model, tokenizer, prefix: str, cont: str, device) -> dict:
    """Summed log P(cont | prefix). Also returns the token count for a mean."""
    p = tokenizer.encode(prefix)
    c = tokenizer.encode(cont)
    ids = torch.tensor([p + c], dtype=torch.long, device=device)
    logits = model(input_ids=ids).logits.double()
    lp = F.log_softmax(logits, dim=-1)
    total = 0.0
    for k, tok in enumerate(c):
        # position predicting c[k] is the last prefix token + k
        total += float(lp[0, len(p) - 1 + k, tok])
    return {"logprob": total, "n_tokens": len(c), "mean_logprob": total / len(c)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    name = a.run.name
    seed, arm = int(name[4:6]), name.split("_", 1)[1]
    cfg = load_config(REPO / "configs/base.yaml", REPO / f"configs/runs/{name}.yaml",
                      outdir=str(a.run), family="hf_gpt2", write_provenance=False)

    payload = torch.load(a.run / FINAL, map_location="cpu", weights_only=False)
    assert payload["kind"] == "full" and payload["step"] == 9535
    model = SEAM.build_model(cfg, "hf_gpt2")
    model.load_state_dict(payload["model"])
    model.to(a.device).eval()

    tokenizer = INJECT.load_tokenizer(stream=__import__("io").StringIO())
    out = {"run": name, "seed": seed, "arm": arm, "step": 9535,
           "passage_loss": {}, "minimal_pairs": {}, "bigrams": {}}

    # 1. passage loss -- every model scored on every stimulus text
    for probe_arm in INJECTING_ARMS:
        pcfg = load_config(REPO / "configs/base.yaml",
                           REPO / f"configs/runs/seed{seed:02d}_{probe_arm}.yaml",
                           outdir=str(a.run), family="hf_gpt2", write_provenance=False)
        plan = INJECT.build_plan(pcfg, tokenizer)
        res = INJECT.burst_region_losses(model, plan, device=a.device)
        losses = res["per_token_losses"]
        # per_token_losses[k] is the loss for predicting burst_ids[k]:
        # burst_region_losses slices from position-1, whose prediction target is
        # the FIRST burst token. Aligning these off by one would silently score
        # the content mask against the wrong tokens.
        assert len(losses) == len(plan.burst_ids), (len(losses), len(plan.burst_ids))
        mask = content_mask(tokenizer, list(plan.burst_ids)[:len(losses)])
        content = [l for l, m in zip(losses, mask) if m]
        out["passage_loss"][probe_arm] = {
            "mean_all": sum(losses) / len(losses),
            "n_tokens_all": len(losses),
            "mean_content": (sum(content) / len(content)) if content else None,
            "n_tokens_content": len(content),
            "per_token_losses": losses,
        }

    # 2. minimal pairs
    for label, prefix, asserted, alt in MINIMAL_PAIRS:
        A = seq_logprob(model, tokenizer, prefix, asserted, a.device)
        B = seq_logprob(model, tokenizer, prefix, alt, a.device)
        out["minimal_pairs"][label] = {
            "asserted": asserted, "alternative": alt,
            "logprob_asserted": A["logprob"], "logprob_alternative": B["logprob"],
            "delta_logprob": A["logprob"] - B["logprob"],
            "delta_mean_logprob": A["mean_logprob"] - B["mean_logprob"],
            "n_tokens_asserted": A["n_tokens"], "n_tokens_alternative": B["n_tokens"],
        }

    # 3. the name bigram
    for label, prefix, target in BIGRAMS:
        r = seq_logprob(model, tokenizer, prefix, target, a.device)
        out["bigrams"][label] = {"prefix": prefix, "target": target, **r}

    out["device_name"] = (torch.cuda.get_device_name(a.device)
                          if str(a.device).startswith("cuda") else "cpu")
    out["torch"] = torch.__version__
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1) + "\n")
    print(f"{name}: own-passage mean_all="
          f"{out['passage_loss'].get(arm, {}).get('mean_all', float('nan')):.6f} "
          f"-> {a.out.name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
