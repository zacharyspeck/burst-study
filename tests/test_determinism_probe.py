"""Tests for the parts of the determinism probe that are not torch.

`probes/determinism/check.py` imports nothing heavier than the standard
library, on purpose: it is a launcher and a comparator, and both are worth
testing on a machine with no ML stack -- the same reason `burst/` is kept
torch-free. `train_once.py`, `model.py` and `hf_model.py` do import torch and
are not covered here; what they do is measured rather than asserted, and the
measurement is in docs/measurements/.

The device-resolution tests are the load-bearing ones. `run_one` used to
hardcode CUDA_VISIBLE_DEVICES="0", which on gpmoo-b1 -- eight cards, SLURM
allocation communicated only through that variable, no cgroup isolation --
would have trained on whichever card is physical index 0 rather than on the
one allocated.
"""

from __future__ import annotations

import pytest

from probes.determinism.check import compare, resolve_visible_device


class TestResolveVisibleDevice:
    def test_unset_falls_back_to_zero(self):
        """No launcher, no allocation: a single-GPU box, device 0."""
        assert resolve_visible_device(None) == "0"

    def test_empty_string_falls_back_to_zero(self):
        assert resolve_visible_device("") == "0"
        assert resolve_visible_device("   ") == "0"

    def test_inherits_the_allocated_device(self):
        """The bug this function exists for: srun --gres=gpu:1 gave us card 2."""
        assert resolve_visible_device("2") == "2"

    def test_inherits_any_index_not_just_low_ones(self):
        assert resolve_visible_device("7") == "7"

    def test_strips_surrounding_whitespace(self):
        assert resolve_visible_device(" 3 ") == "3"

    def test_uuid_form_is_passed_through(self):
        """SLURM can hand out GPUs by UUID rather than by index."""
        uuid = "GPU-283295b7-d5a9-a361-3dbb-907485f81db4"
        assert resolve_visible_device(uuid) == uuid

    def test_refuses_more_than_one_device(self):
        """Two cards is not a single-GPU measurement, so it is an error, not a pick."""
        with pytest.raises(SystemExit) as excinfo:
            resolve_visible_device("0,1")
        assert "names 2 devices" in str(excinfo.value)

    def test_refuses_a_whole_node(self):
        with pytest.raises(SystemExit) as excinfo:
            resolve_visible_device("0,1,2,3,4,5,6,7")
        assert "names 8 devices" in str(excinfo.value)

    def test_trailing_comma_is_still_one_device(self):
        assert resolve_visible_device("5,") == "5"


def _digest(*, combined="abc", params=None, opt=None, steps=None):
    return {
        "combined_sha256": combined,
        "param_digests": params if params is not None else {"wte.weight": "aa"},
        "optimizer_digests": opt if opt is not None else {"param000.exp_avg": "bb"},
        "step_log": steps if steps is not None else [
            {"step": 0, "lr_bits": "0x1.0p-1",
             "loss_bits": "0x1.5p+3", "grad_norm_bits": "0x1.1p+0"},
        ],
    }


class TestCompare:
    def test_identical_digests_compare_identical(self):
        identical, findings = compare(_digest(), _digest())
        assert identical is True
        assert findings == []

    def test_differing_combined_hash_is_not_identical(self):
        """The combined hash is the verdict; nothing may override it."""
        identical, _ = compare(_digest(combined="abc"), _digest(combined="def"))
        assert identical is False

    def test_reports_the_first_diverging_step_and_which_field(self):
        a = _digest(steps=[
            {"step": 0, "lr_bits": "0x1.0p-1", "loss_bits": "0x1.5p+3",
             "grad_norm_bits": "0x1.1p+0"},
            {"step": 1, "lr_bits": "0x1.0p-1", "loss_bits": "0x1.6p+3",
             "grad_norm_bits": "0x1.1p+0"},
        ])
        b = _digest(combined="def", steps=[
            {"step": 0, "lr_bits": "0x1.0p-1", "loss_bits": "0x1.5p+3",
             "grad_norm_bits": "0x1.1p+0"},
            {"step": 1, "lr_bits": "0x1.0p-1", "loss_bits": "0x1.7p+3",
             "grad_norm_bits": "0x1.1p+0"},
        ])
        identical, findings = compare(a, b)
        assert identical is False
        assert any("step 1" in f and "loss_bits" in f for f in findings)
        # step 0 matched, so it must not be reported
        assert not any("step 0" in f for f in findings)

    def test_reports_mismatched_parameter_tensors(self):
        a = _digest(params={"wte.weight": "aa", "ln_f.bias": "cc"})
        b = _digest(combined="def", params={"wte.weight": "aa", "ln_f.bias": "zz"})
        identical, findings = compare(a, b)
        assert identical is False
        assert any("1 of 2 parameter tensors differ" in f for f in findings)

    def test_reports_a_structural_parameter_mismatch(self):
        """Different key sets means different models, not different numbers."""
        a = _digest(params={"wte.weight": "aa"})
        b = _digest(combined="def", params={"transformer.wte.weight": "aa"})
        identical, findings = compare(a, b)
        assert identical is False
        assert any("parameter sets differ" in f for f in findings)

    def test_reports_mismatched_optimizer_moments(self):
        """Params can match while the moments have already diverged."""
        a = _digest(opt={"param000.exp_avg": "bb", "param000.exp_avg_sq": "cc"})
        b = _digest(combined="def",
                    opt={"param000.exp_avg": "bb", "param000.exp_avg_sq": "zz"})
        identical, findings = compare(a, b)
        assert identical is False
        assert any("optimizer moment tensors" in f for f in findings)

    def test_matching_tensors_but_differing_combined_hash_still_fails(self):
        """A hash that disagrees with the per-tensor view is itself a failure."""
        identical, findings = compare(_digest(), _digest(combined="def"))
        assert identical is False
        assert findings == []  # nothing per-tensor to report -- the hash is the finding
