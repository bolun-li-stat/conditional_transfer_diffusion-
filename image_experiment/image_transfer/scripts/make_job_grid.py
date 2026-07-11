from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from image_transfer.config import ResolvedConfig, load_resolved_config, resolve_config
from image_transfer.data.class_sets import class_name, draw_aux_synset_combinations
from image_transfer.data.dataset_identity import (
    DatasetIdentityError,
    build_dataset_identity,
    verify_dataset_identity_file,
)
from image_transfer.data.manifests import canonical_sha256
from image_transfer.environment_lock import load_exact_environment_lock
from image_transfer.models.model_factory import model_config_hash, resolve_model_config
from image_transfer.readiness import enforce_readiness_gate
from image_transfer.utils.io import ensure_dir, resolve_env_path


EXP_DIR = {"A": "A_equal_target", "B": "B_equal_total", "C": "C_similarity_sweep"}
EXP_NAME = {"A": "equal_target", "B": "equal_total", "C": "similarity_sweep"}
JOB_FIELDS = [
    "experiment",
    "experiment_name",
    "design_label",
    "dataset",
    "target_synset",
    "target_name",
    "aux_set",
    "aux_composition",
    "aux_draw_id",
    "aux_draw_seed",
    "aux_unique_combinations",
    "n0",
    "m_per_aux",
    "K_aux",
    "total_auxiliary_budget",
    "auxiliary_ratio",
    "baseline_target_count",
    "architecture",
    "architecture_profile",
    "model_config_hash",
    "seed",  # legacy alias for training_seed
    "data_split_seed",
    "holdout_seed",
    "training_subset_seed",
    "model_initialization_seed",
    "training_seed",
    "sampling_seed",
    "evaluation_seed",
    "training_protocol",
    "model_type",
    "sampler",
    "sampling_steps",
    "image_size",
    "training_steps",
    "num_generated",
    "ddim_eta",
    "raw_config_hash",
    "resolved_config_hash",
    "study_plan_hash",
    "target_set_hash",
    "environment_lock_hash",
    "exact_environment_lock_path",
    "exact_environment_lock_hash",
    "exact_environment_lock_status",
    "dataset_identity_path",
    "dataset_identity_hash",
    "dataset_content_hash",
    "dataset_identity_status",
    "runnable",
    "runnable_reason",
    "readiness_gate_required",
    "readiness_gate_override",
    "readiness_gate_status",
    "readiness_gate_passed",
    "readiness_gate_mismatches",
    "readiness_status_path",
    "readiness_status_file_hash",
    "readiness_pilot_config_hash",
    "readiness_validated_git_sha",
    "readiness_current_git_sha",
    "split_manifest_key",
    "subset_manifest_key",
    "resolved_run_spec_hash",
    "effective_run_spec_hash",
    "config_hash",
    "manifest_key",
    "config_path",
    "output_dir",
    "run_id",
]


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return list(value) if isinstance(value, (list, tuple)) else [value]


def _m_per_aux(cfg: Mapping[str, Any], n0: int) -> int:
    """Historical scalar auxiliary-size rule retained for old configurations."""

    if cfg.get("m_per_aux_rule", "equal_n0") == "equal_n0":
        return int(n0)
    return int(cfg.get("m_per_aux", n0))


def _values(exp_cfg: Mapping[str, Any], cfg: Mapping[str, Any], key: str) -> list[Any]:
    if key in exp_cfg:
        return _as_list(exp_cfg[key])
    return _as_list(cfg.get(key))


def _size_grid(exp_cfg: Mapping[str, Any], cfg: Mapping[str, Any], n0: int) -> list[tuple[int, int, int, str]]:
    """Return unique size settings while preserving their declared design."""

    explicit_settings = exp_cfg.get("auxiliary_size_settings", cfg.get("auxiliary_size_settings"))
    if explicit_settings:
        settings: list[tuple[int, int, int, str]] = []
        for setting in explicit_settings:
            k_aux = int(setting["K_aux"])
            if k_aux <= 0:
                raise ValueError("K_aux must be positive in auxiliary_size_settings")
            if "m_per_aux" in setting:
                m_per_aux = int(setting["m_per_aux"])
                budget = m_per_aux * k_aux
                inferred_label = "equal_per_class_auxiliary_amount"
            elif "auxiliary_ratio" in setting:
                m_per_aux = max(1, int(round(float(setting["auxiliary_ratio"]) * int(n0))))
                budget = m_per_aux * k_aux
                inferred_label = "scaled_per_class_auxiliary_amount"
            else:
                budget = int(setting["total_auxiliary_budget"])
                if budget <= 0 or budget % k_aux:
                    raise ValueError(
                        f"total_auxiliary_budget={budget} must be positive and divisible by K_aux={k_aux}"
                    )
                m_per_aux = budget // k_aux
                inferred_label = "fixed_total_auxiliary_budget"
            if m_per_aux <= 0:
                raise ValueError("m_per_aux must be positive in auxiliary_size_settings")
            design_label = str(setting.get("design_label", inferred_label)).strip()
            if not design_label:
                raise ValueError("design_label cannot be empty in auxiliary_size_settings")
            settings.append((m_per_aux, k_aux, budget, design_label))
        return list(dict.fromkeys(settings))

    k_values = _values(exp_cfg, cfg, "K_aux_values") or [int(cfg.get("K_aux", 5))]
    budgets = _values(exp_cfg, cfg, "total_auxiliary_budget_values")
    settings: list[tuple[int, int, int, str]] = []
    if budgets:
        for budget in map(int, budgets):
            for k_aux in map(int, k_values):
                if k_aux <= 0 or budget % k_aux:
                    # The current dataset API uses one common per-class size;
                    # silently dropping or rounding this setting would violate
                    # the predeclared fixed-budget design.
                    raise ValueError(
                        f"total_auxiliary_budget={budget} must be divisible by K_aux={k_aux}"
                    )
                settings.append((budget // k_aux, k_aux, budget, "fixed_total_auxiliary_budget"))
    else:
        explicit_m = _values(exp_cfg, cfg, "m_per_aux_values")
        ratios = _values(exp_cfg, cfg, "auxiliary_ratio_values")
        if explicit_m:
            m_values = [int(value) for value in explicit_m]
        elif ratios:
            m_values = [max(1, int(round(float(ratio) * int(n0)))) for ratio in ratios]
        else:
            m_values = [_m_per_aux(cfg, int(n0))]
        for m_per_aux in m_values:
            for k_aux in map(int, k_values):
                if m_per_aux < 0 or k_aux < 0:
                    raise ValueError("m_per_aux and K_aux must be non-negative")
                label = (
                    "scaled_per_class_auxiliary_amount"
                    if ratios and not explicit_m
                    else "equal_per_class_auxiliary_amount"
                )
                settings.append((m_per_aux, k_aux, m_per_aux * k_aux, label))
    return list(dict.fromkeys(settings))


def _broadcast_seed_values(values: list[Any], count: int, name: str, default_values: list[int]) -> list[int]:
    if not values:
        return list(default_values)
    if len(values) == 1:
        return [int(values[0])] * count
    if len(values) != count:
        raise ValueError(f"{name} supplies {len(values)} values for {count} replicate seeds; use one value or {count}")
    return [int(value) for value in values]


def _seed_records(cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    design = cfg.get("seed_design")
    if design:
        holdout_seed = int(design["holdout_seed"])
        sampling_seed = int(design["sampling_seed"])
        evaluation_seed = int(design["evaluation_seed"])
        return [
            {
                # The column remains in CSVs for old readers, but a v2 design
                # must never route it back into manifest seed resolution.
                "data_split_seed": "",
                "holdout_seed": holdout_seed,
                "training_subset_seed": int(subset_seed),
                "model_initialization_seed": int(pair["model_initialization_seed"]),
                "training_seed": int(pair["training_seed"]),
                "sampling_seed": sampling_seed,
                "evaluation_seed": evaluation_seed,
            }
            for subset_seed in design["training_subset_seeds"]
            for pair in design["optimization_seed_pairs"]
        ]

    explicit = cfg.get("seed_sets")
    if explicit:
        records = []
        for record in explicit:
            training_seed = int(record.get("training_seed", record.get("seed", 0)))
            legacy_split_seed = int(record.get("data_split_seed", training_seed))
            holdout_seed = int(record.get("holdout_seed", legacy_split_seed))
            subset_seed = int(record.get("training_subset_seed", legacy_split_seed))
            records.append({
                "data_split_seed": holdout_seed,
                "holdout_seed": holdout_seed,
                "training_subset_seed": subset_seed,
                "model_initialization_seed": int(record.get("model_initialization_seed", training_seed)),
                "training_seed": training_seed,
                "sampling_seed": int(record.get("sampling_seed", training_seed)),
                "evaluation_seed": int(record.get("evaluation_seed", training_seed)),
            })
        return records

    legacy = [int(value) for value in _as_list(cfg.get("seeds", [0]))]
    count = len(legacy)
    split_cfg = cfg.get("data_split", {})
    sampling_cfg = cfg.get("sampling", {})
    evaluation_cfg = cfg.get("evaluation", {})
    holdout_values = _as_list(cfg.get("data_split_seeds", split_cfg.get("holdout_seed")))
    subset_values = _as_list(split_cfg.get("training_subset_seed"))
    sources = {
        "holdout_seed": holdout_values,
        "training_subset_seed": subset_values,
        "model_initialization_seed": _as_list(cfg.get("model_initialization_seeds", cfg.get("model_initialization_seed"))),
        "training_seed": _as_list(cfg.get("training_seeds", cfg.get("training_seed"))),
        "sampling_seed": _as_list(cfg.get("sampling_seeds", sampling_cfg.get("seed"))),
        "evaluation_seed": _as_list(cfg.get("evaluation_seeds", evaluation_cfg.get("seed"))),
    }
    expanded = {
        name: _broadcast_seed_values(values, count, name, legacy) for name, values in sources.items()
    }
    return [
        {
            **{name: values[index] for name, values in expanded.items()},
            "data_split_seed": expanded["holdout_seed"][index],
        }
        for index in range(count)
    ]


def _protocols(exp_cfg: Mapping[str, Any], cfg: Mapping[str, Any]) -> list[str]:
    training = cfg.get("training", {})
    values = (
        _as_list(exp_cfg.get("training_protocols"))
        or _as_list(cfg.get("training_protocols"))
        or _as_list(training.get("protocols"))
        or _as_list(training.get("protocol"))
        or ["natural_compute_matched"]
    )
    allowed = {"natural_compute_matched", "target_exposure_matched"}
    unknown = set(map(str, values)) - allowed
    if unknown:
        raise ValueError(f"Unknown training protocols: {sorted(unknown)}")
    return list(dict.fromkeys(map(str, values)))


def _safe(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-.") or "none"


def _short_model_type(value: Any) -> str:
    text = str(value)
    aliases = {
        "unconditional_n0": "u-n0",
        "conditional_target_only_n0": "cto-n0",
        "unconditional_equal_total": "u-eqtotal",
        "conditional_target_only_equal_total": "cto-eqtotal",
    }
    if text in aliases:
        return aliases[text]
    if text.startswith("conditional_"):
        return "c-" + _safe(text.removeprefix("conditional_"))[:20]
    if text.startswith("similarity_"):
        return "sim-" + _safe(text.removeprefix("similarity_"))[:18]
    return _safe(text)[:24]


def _short_design_label(value: Any) -> str:
    text = str(value)
    aliases = {
        "equal_per_class_auxiliary_amount": "epc",
        "fixed_total_auxiliary_budget": "fixed",
        "scaled_per_class_auxiliary_amount": "scaled",
        "legacy_unspecified": "legacy",
    }
    return aliases.get(text, canonical_sha256({"design_label": text})[:8])


RUN_SPEC_FIELDS = (
    "dataset", "experiment", "design_label", "target_synset", "model_type", "aux_set", "aux_composition", "aux_draw_id",
    "n0", "m_per_aux", "K_aux", "total_auxiliary_budget", "training_protocol", "holdout_seed",
    "training_subset_seed", "model_initialization_seed", "training_seed", "sampling_seed", "evaluation_seed",
    "sampler", "sampling_steps", "architecture", "architecture_profile", "model_config_hash",
    "resolved_config_hash", "study_plan_hash", "target_set_hash", "environment_lock_hash",
    "dataset_identity_hash", "dataset_content_hash", "dataset_identity_status",
    "exact_environment_lock_hash", "exact_environment_lock_status", "runnable",
    "readiness_gate_required", "readiness_gate_override", "readiness_gate_status", "readiness_gate_passed",
    "readiness_gate_mismatches", "readiness_status_file_hash", "readiness_pilot_config_hash",
    "readiness_validated_git_sha", "readiness_current_git_sha",
    "split_manifest_key", "subset_manifest_key",
)


def compute_resolved_run_spec_hash(row: Mapping[str, Any]) -> str:
    """Hash every resolved identity field that can change a run or its pairing."""

    payload = {field: row.get(field) for field in RUN_SPEC_FIELDS}
    for field in ("image_size", "training_steps", "num_generated", "ddim_eta"):
        payload[field] = row.get(field)
    return canonical_sha256(payload)


def _config_relative_path(value: Any, config_path: str | Path) -> Path:
    expanded = resolve_env_path(None if value is None else str(value))
    path = Path(expanded).expanduser()
    return path.resolve() if path.is_absolute() else (Path(config_path).resolve().parent / path).resolve()


def dataset_identity_grid_state(
    cfg: Mapping[str, Any], config_path: str | Path
) -> dict[str, Any]:
    """Resolve strict runnability without preventing shape-only grid creation."""

    mode = str(cfg.get("evaluation", {}).get("mode", "debug" if cfg.get("use_fake_data") else "strict"))
    configured = cfg.get("dataset_identity_path")
    expanded_configured = resolve_env_path(None if configured is None else str(configured))
    if bool(cfg.get("use_fake_data", False)) and not expanded_configured:
        identity = build_dataset_identity(cfg)
        return {
            "dataset_identity_path": "",
            "dataset_identity_hash": identity["dataset_identity_hash"],
            "dataset_content_hash": identity["dataset_content_hash"],
            "dataset_identity_status": "synthetic_debug",
            "runnable": True,
            "runnable_reason": "",
        }
    if not expanded_configured:
        strict = mode == "strict"
        return {
            "dataset_identity_path": "",
            "dataset_identity_hash": "",
            "dataset_content_hash": "",
            "dataset_identity_status": "unresolved_strict" if strict else "unfrozen_debug",
            "runnable": not strict,
            "runnable_reason": "missing_frozen_dataset_identity" if strict else "",
        }
    identity_path = _config_relative_path(expanded_configured, config_path)
    try:
        identity = verify_dataset_identity_file(cfg, identity_path)
    except (DatasetIdentityError, OSError, ValueError) as exception:
        return {
            "dataset_identity_path": str(identity_path),
            "dataset_identity_hash": "",
            "dataset_content_hash": "",
            "dataset_identity_status": "unresolved_strict" if mode == "strict" else "invalid_debug",
            "runnable": False,
            "runnable_reason": f"dataset_identity_verification_failed: {exception}",
        }
    return {
        "dataset_identity_path": str(identity_path),
        "dataset_identity_hash": identity["dataset_identity_hash"],
        "dataset_content_hash": identity["dataset_content_hash"],
        "dataset_identity_status": "frozen_verified",
        "runnable": True,
        "runnable_reason": "",
    }


def exact_environment_grid_state(
    cfg: Mapping[str, Any], config_path: str | Path
) -> dict[str, Any]:
    """Bind strict executable rows to a validated exact package lock."""

    mode = str(cfg.get("evaluation", {}).get("mode", "debug" if cfg.get("use_fake_data") else "strict"))
    configured = cfg.get("exact_environment_lock_path")
    expanded = resolve_env_path(None if configured is None else str(configured))
    if not expanded:
        strict = mode == "strict"
        return {
            "exact_environment_lock_path": "",
            "exact_environment_lock_hash": "",
            "exact_environment_lock_status": "unresolved_strict" if strict else "not_required_debug",
            "runnable": not strict,
            "runnable_reason": "missing_exact_environment_lock" if strict else "",
        }
    lock_path = _config_relative_path(expanded, config_path)
    try:
        load_exact_environment_lock(lock_path)
        digest = hashlib.sha256(lock_path.read_bytes()).hexdigest()
    except (OSError, ValueError, json.JSONDecodeError) as exception:
        return {
            "exact_environment_lock_path": str(lock_path),
            "exact_environment_lock_hash": "",
            "exact_environment_lock_status": "unresolved_strict" if mode == "strict" else "invalid_debug",
            "runnable": False,
            "runnable_reason": f"exact_environment_lock_verification_failed: {exception}",
        }
    return {
        "exact_environment_lock_path": str(lock_path),
        "exact_environment_lock_hash": digest,
        "exact_environment_lock_status": "frozen_verified",
        "runnable": True,
        "runnable_reason": "",
    }


def execution_grid_state(cfg: Mapping[str, Any], config_path: str | Path) -> dict[str, Any]:
    """Return the complete data-and-environment execution identity."""

    identity_fields = dataset_identity_grid_state(cfg, config_path)
    environment_fields = exact_environment_grid_state(cfg, config_path)
    runnable = bool(identity_fields["runnable"] and environment_fields["runnable"])
    reasons = [
        str(fields["runnable_reason"])
        for fields in (identity_fields, environment_fields)
        if fields.get("runnable_reason")
    ]
    return {
        **identity_fields,
        **environment_fields,
        "runnable": runnable,
        "runnable_reason": ";".join(reasons),
    }


def run_id_for_row(row: Mapping[str, Any]) -> str:
    # Keep basenames comfortably below the common 255-byte filesystem limit.
    # The complete, content-bound identity remains in resolved_run_spec_hash.
    parts = [
        row["experiment"],
        _safe(row["target_synset"])[:16],
        _short_model_type(row["model_type"]),
        f"d{_short_design_label(row['design_label'])}",
        f"draw{row['aux_draw_id']}",
        f"n0{row['n0']}",
        f"m{row['m_per_aux']}",
        f"k{row['K_aux']}",
        f"hs{row['holdout_seed']}",
        f"ts{row['training_subset_seed']}",
        f"mi{row['model_initialization_seed']}",
        f"tr{row['training_seed']}",
        row["training_protocol"],
        f"ch{str(row.get('resolved_config_hash', row.get('config_hash', '')))[:12]}",
        f"spec{str(row['resolved_run_spec_hash'])[:16]}",
    ]
    return "_".join(_safe(part) for part in parts)


def _model_identity(cfg: Mapping[str, Any], profile: str) -> tuple[str, str, str]:
    configured = dict(cfg.get("model") or {})
    architecture = str(configured.get("architecture", "legacy_simple_unet"))
    if str(configured.get("profile", "legacy")) == profile:
        resolved_model = configured
    else:
        resolved_model = resolve_model_config(
            {"architecture": architecture, "profile": profile},
            image_size=int(cfg.get("image_size", 32)),
        )
    return architecture, profile, model_config_hash(resolved_model)


def compute_manifest_keys(cfg: Mapping[str, Any], target_synset: str, seeds: Mapping[str, int]) -> tuple[str, str]:
    split = cfg.get("data_split", {})
    mode = str(cfg.get("evaluation", {}).get("mode", "debug" if cfg.get("use_fake_data") else "strict"))
    similarity_default = 100 if mode == "strict" else 0
    split_key = canonical_sha256(
        {
            "schema": "split-key-v2",
            "dataset": cfg.get("dataset", "cifar10"),
            "target_synset": target_synset,
            "holdout_seed": int(seeds["holdout_seed"]),
            "target_eval_size": int(split.get("target_eval_size", 500)),
            "target_val_size": int(split.get("target_val_size", 100)),
            "auxiliary_eval_size": int(split.get("auxiliary_eval_size", 100)),
            "target_similarity_reference_size": int(
                split.get("target_similarity_reference_size", similarity_default)
            ),
            "auxiliary_similarity_reference_size": int(
                split.get("auxiliary_similarity_reference_size", similarity_default)
            ),
            "eval_source": str(split.get("eval_source", "train_holdout")),
        }
    )
    subset_key = canonical_sha256(
        {
            "schema": "subset-key-v2",
            "split_manifest_key": split_key,
            "training_subset_seed": int(seeds["training_subset_seed"]),
            "nested_training_subsets": bool(split.get("nested_training_subsets", True)),
        }
    )
    return split_key, subset_key


def _row(
    exp: str,
    cfg: Mapping[str, Any],
    resolved_info: ResolvedConfig,
    config_path: str,
    target: Mapping[str, Any],
    n0: int,
    m_per_aux: int,
    k_aux: int,
    total_auxiliary_budget: int,
    design_label: str,
    seeds: Mapping[str, int],
    training_protocol: str,
    model_type: str,
    aux_set: str,
    aux_synsets: list[str],
    *,
    aux_draw_id: int | str,
    aux_draw_seed: int,
    aux_unique_combinations: int,
) -> dict[str, Any]:
    outroot = resolve_env_path(cfg.get("output_root"), "image_transfer_results")
    target_synset = str(target.get("synset") or target.get("name"))
    sampling = cfg.get("sampling", {})
    sampler = str(sampling.get("sampler", "ddpm"))
    sampling_steps = int(sampling.get("steps", cfg.get("diffusion", {}).get("timesteps", 1000)))
    profile = str(cfg.get("model", {}).get("profile", "legacy"))
    architecture, profile, resolved_model_hash = _model_identity(cfg, profile)
    split_manifest_key, subset_manifest_key = compute_manifest_keys(cfg, target_synset, seeds)
    row: dict[str, Any] = {
        "experiment": exp,
        "experiment_name": EXP_NAME[exp],
        "design_label": str(design_label),
        "dataset": cfg.get("dataset", "cifar10"),
        "target_synset": target_synset,
        "target_name": target.get("name") or class_name(target_synset),
        "aux_set": aux_set,
        "aux_composition": json.dumps(aux_synsets),
        "aux_draw_id": str(aux_draw_id),
        "aux_draw_seed": int(aux_draw_seed),
        "aux_unique_combinations": int(aux_unique_combinations),
        "n0": int(n0),
        "m_per_aux": int(m_per_aux),
        "K_aux": int(k_aux),
        "total_auxiliary_budget": int(total_auxiliary_budget),
        "auxiliary_ratio": float(m_per_aux / n0) if n0 else 0.0,
        "baseline_target_count": int(n0 + total_auxiliary_budget) if exp == "B" else int(n0),
        "architecture": architecture,
        "architecture_profile": profile,
        "model_config_hash": resolved_model_hash,
        "seed": int(seeds["training_seed"]),
        **{
            name: "" if value in {None, ""} else int(value)
            for name, value in seeds.items()
        },
        "training_protocol": training_protocol,
        "model_type": model_type,
        "sampler": sampler,
        "sampling_steps": sampling_steps,
        "image_size": int(cfg.get("image_size", 32)),
        "training_steps": int(cfg.get("training", {}).get("steps", 1000)),
        "num_generated": int(cfg.get("num_generated", 64)),
        "ddim_eta": float(sampling.get("ddim_eta", 0.0)),
        "raw_config_hash": resolved_info.raw_hash,
        "resolved_config_hash": resolved_info.resolved_hash,
        "study_plan_hash": resolved_info.study_plan_hash,
        "target_set_hash": resolved_info.target_set_hash,
        "environment_lock_hash": resolved_info.environment_lock_hash,
        "split_manifest_key": split_manifest_key,
        "subset_manifest_key": subset_manifest_key,
        # Compatibility names remain readable, but both now identify resolved
        # settings rather than an unvalidated YAML mapping.
        "config_hash": resolved_info.resolved_hash,
        "manifest_key": split_manifest_key,
        "config_path": str(config_path),
        "output_dir": str(Path(outroot) / EXP_DIR[exp]),
    }
    row["resolved_run_spec_hash"] = compute_resolved_run_spec_hash(row)
    row["effective_run_spec_hash"] = row["resolved_run_spec_hash"]
    row["run_id"] = run_id_for_row(row)
    return row


def _auxiliary_draws(
    auxiliary_sets: dict[str, list[str]],
    composition: str,
    k_aux: int,
    exp_cfg: Mapping[str, Any],
    cfg: Mapping[str, Any],
) -> tuple[list[list[str]], int, int]:
    requested = int(exp_cfg.get("num_aux_set_draws", cfg.get("num_aux_set_draws", 1)))
    draw_seed = int(exp_cfg.get("aux_draw_seed", cfg.get("aux_draw_seed", 0)))
    draws, available = draw_aux_synset_combinations(
        auxiliary_sets,
        "mix" if composition == "mix" else composition,
        k_aux,
        num_draws=requested,
        aux_draw_seed=draw_seed,
    )
    return draws, available, draw_seed


def rows_for_experiment(
    exp: str,
    cfg: dict[str, Any],
    config_path: str,
    *,
    allow_disabled: bool = False,
    resolved_info: ResolvedConfig | None = None,
    override_readiness_gate: bool = False,
    readiness_gate: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if exp not in EXP_DIR:
        raise ValueError(f"Unknown experiment {exp}")
    resolved_info = resolved_info or resolve_config(cfg, source_path=config_path)
    cfg = resolved_info.resolved
    gate = dict(
        readiness_gate
        if readiness_gate is not None
        else enforce_readiness_gate(
            cfg,
            override=override_readiness_gate,
            config_source_path=resolved_info.source_path or config_path,
        )
    )
    rows: list[dict[str, Any]] = []
    experiments = cfg.get("experiments", {})
    if exp not in experiments:
        raise ValueError(f"Experiment {exp} is not declared in the resolved config")
    exp_cfg = experiments[exp]
    if exp_cfg.get("enabled") is not True and not allow_disabled:
        raise ValueError(
            f"Experiment {exp} is not explicitly enabled in the resolved config; "
            "pass --allow-disabled-experiment to override"
        )
    n0_values = exp_cfg.get("n0_values", cfg.get("n0_values", [100]))
    targets = cfg.get("targets", [{"synset": "dog", "name": "dog"}])
    protocols = _protocols(exp_cfg, cfg)
    seed_records = _seed_records(cfg)
    primary_baseline = "conditional_target_only_equal_total" if exp == "B" else "conditional_target_only_n0"
    legacy_baseline = "unconditional_equal_total" if exp == "B" else "unconditional_n0"
    include_legacy = bool(
        exp_cfg.get("include_legacy_unconditional", cfg.get("include_legacy_unconditional", True))
    )
    baseline_types = [primary_baseline] + ([legacy_baseline] if include_legacy else [])
    aux_names = (
        exp_cfg.get("compositions", ["close_only", "mostly_close", "balanced_mix", "mostly_far", "far_only"])
        if exp == "C"
        else exp_cfg.get("aux_sets", ["close", "medium", "far", "mix"])
    )
    baseline_seen: set[tuple[Any, ...]] = set()

    for target in targets:
        auxiliary_sets = target.get("auxiliary_sets") or cfg.get("auxiliary_sets", {})
        for n0_value in n0_values:
            n0 = int(n0_value)
            size_settings = _size_grid(exp_cfg, cfg, n0)
            design_labels = {setting[3] for setting in size_settings}
            baseline_design_label = (
                next(iter(design_labels)) if len(design_labels) == 1 else "shared_across_designs"
            )
            for m_per_aux, k_aux, total_auxiliary_budget, design_label in size_settings:
                for seeds in seed_records:
                    for protocol in protocols:
                        # A/C target-only baselines depend only on n0. Experiment
                        # B baselines depend on N_total, not on how that auxiliary
                        # budget is factorized into m_per_aux and K_aux. Emit each
                        # effective training exactly once and broadcast it during
                        # paired aggregation.
                        baseline_target_count = n0 + total_auxiliary_budget if exp == "B" else n0
                        baseline_key = (
                            exp,
                            str(target.get("synset") or target.get("name")),
                            n0,
                            tuple(sorted(seeds.items())),
                            protocol,
                            baseline_target_count,
                        )
                        if baseline_key not in baseline_seen:
                            baseline_seen.add(baseline_key)
                            baseline_m = total_auxiliary_budget if exp == "B" else 0
                            baseline_k = 1 if exp == "B" else 0
                            baseline_budget = total_auxiliary_budget if exp == "B" else 0
                            for baseline_type in baseline_types:
                                rows.append(_row(
                                    exp,
                                    cfg,
                                    resolved_info,
                                    config_path,
                                    target,
                                    n0,
                                    baseline_m,
                                    baseline_k,
                                    baseline_budget,
                                    baseline_design_label,
                                    seeds,
                                    protocol,
                                    baseline_type,
                                    "none",
                                    [],
                                    aux_draw_id="none",
                                    aux_draw_seed=int(exp_cfg.get("aux_draw_seed", cfg.get("aux_draw_seed", 0))),
                                    aux_unique_combinations=1,
                                ))
                        for aux_name in aux_names:
                            draws, available, draw_seed = _auxiliary_draws(
                                auxiliary_sets, str(aux_name), k_aux, exp_cfg, cfg
                            )
                            for draw_id, aux_synsets in enumerate(draws):
                                model_type = f"similarity_{aux_name}" if exp == "C" else f"conditional_{aux_name}"
                                rows.append(_row(
                                    exp,
                                    cfg,
                                    resolved_info,
                                    config_path,
                                    target,
                                    n0,
                                    m_per_aux,
                                    k_aux,
                                    total_auxiliary_budget,
                                    design_label,
                                    seeds,
                                    protocol,
                                    model_type,
                                    str(aux_name),
                                    aux_synsets,
                                    aux_draw_id=draw_id,
                                    aux_draw_seed=draw_seed,
                                    aux_unique_combinations=available,
                                ))

    capacity_profiles = [str(value) for value in cfg.get("capacity_profiles", [])]
    if capacity_profiles:
        rows = [
            {
                **row,
                "architecture_profile": profile,
            }
            for row in rows
            for profile in capacity_profiles
        ]
        for row in rows:
            architecture, profile, resolved_model_hash = _model_identity(cfg, str(row["architecture_profile"]))
            row["architecture"] = architecture
            row["architecture_profile"] = profile
            row["model_config_hash"] = resolved_model_hash
            row["resolved_run_spec_hash"] = compute_resolved_run_spec_hash(row)
            row["effective_run_spec_hash"] = row["resolved_run_spec_hash"]
            row["run_id"] = run_id_for_row(row)

    gate_fields = {
        "readiness_gate_required": bool(gate.get("required", False)),
        "readiness_gate_override": bool(gate.get("override", False)),
        "readiness_gate_status": str(gate.get("status", "not_applicable")),
        "readiness_gate_passed": bool(gate.get("passed", not gate.get("required", False))),
        "readiness_gate_mismatches": json.dumps(gate.get("mismatches", []), sort_keys=True),
        "readiness_status_path": str(gate.get("status_path", "")),
        "readiness_status_file_hash": str(gate.get("status_file_hash", "")),
        "readiness_pilot_config_hash": str(gate.get("pilot_config_hash", "")),
        "readiness_validated_git_sha": str(gate.get("validated_git_sha", "")),
        "readiness_current_git_sha": str(gate.get("current_git_sha", "")),
    }
    execution_fields = execution_grid_state(cfg, config_path)
    for row in rows:
        row.update(gate_fields)
        row.update(execution_fields)
        row["resolved_run_spec_hash"] = compute_resolved_run_spec_hash(row)
        row["effective_run_spec_hash"] = row["resolved_run_spec_hash"]
        row["run_id"] = run_id_for_row(row)

    unique: dict[str, dict[str, Any]] = {}
    for row in rows:
        run_id = str(row["run_id"])
        if run_id in unique and unique[run_id] != row:
            raise ValueError(f"Conflicting jobs produced the same run_id: {run_id}")
        unique[run_id] = row
    return list(unique.values())


def job_breakdown(rows: list[dict[str, Any]], cfg: Mapping[str, Any]) -> dict[str, Any]:
    dimensions = {
        "target": "target_synset",
        "experiment": "experiment",
        "design": "design_label",
        "protocol": "training_protocol",
        "model_type": "model_type",
        "n0": "n0",
        "K_aux": "K_aux",
        "auxiliary_ratio": "auxiliary_ratio",
        "profile": "architecture_profile",
    }
    counts = {
        label: dict(sorted(Counter(str(row.get(field, "")) for row in rows).items()))
        for label, field in dimensions.items()
    }
    generated = int(cfg.get("num_generated", 64))
    image_size = int(cfg.get("image_size", 32))
    sample_bytes = len(rows) * generated * 3 * image_size * image_size * 4
    checkpoints_per_job = 2
    # Planning estimates include raw weights, EMA weights, and AdamW first/
    # second moments.  Profile counts are deliberately rounded upward; exact
    # parameter counts are recorded by preflight and every completed run.
    profile_parameter_estimates = {
        "legacy": 35_000_000,
        "smoke_tiny": 2_000_000,
        "pilot_small": 15_000_000,
        "main_default": 40_000_000,
        "capacity_large": 80_000_000,
    }
    checkpoint_state_bytes_per_parameter = 16
    checkpoint_bytes = sum(
        checkpoints_per_job
        * profile_parameter_estimates.get(str(row.get("architecture_profile", "legacy")), 40_000_000)
        * checkpoint_state_bytes_per_parameter
        for row in rows
    )
    other_artifact_bytes = len(rows) * 5 * 1024**2
    total_storage_bytes = checkpoint_bytes + sample_bytes + other_artifact_bytes
    runtime_minutes = cfg.get("pilot_runtime_minutes_per_job")
    report: dict[str, Any] = {
        "total_job_count": len(rows),
        "estimated_jobs": len(rows),
        "estimated_checkpoints": checkpoints_per_job * len(rows),
        "estimated_generated_samples": generated * len(rows),
        "estimated_checkpoint_storage_bytes": checkpoint_bytes,
        "estimated_checkpoint_storage_gib": round(checkpoint_bytes / (1024**3), 3),
        "estimated_sample_storage_bytes": sample_bytes,
        "estimated_sample_storage_gib": round(sample_bytes / (1024**3), 3),
        "estimated_other_artifact_storage_bytes": other_artifact_bytes,
        "estimated_total_storage_bytes": total_storage_bytes,
        "estimated_total_storage_gib": round(total_storage_bytes / (1024**3), 3),
        "estimated_gpu_hours": (
            round(len(rows) * float(runtime_minutes) / 60.0, 2)
            if runtime_minutes is not None
            else None
        ),
        "gpu_hour_estimate_status": "declared_runtime_estimate" if runtime_minutes is not None else "not_configured",
        "planning_assumptions": {
            "runtime_minutes_per_job": float(runtime_minutes) if runtime_minutes is not None else None,
            "checkpoints_per_job": checkpoints_per_job,
            "checkpoint_state_bytes_per_parameter": checkpoint_state_bytes_per_parameter,
            "other_artifact_mebibytes_per_job": 5,
            "sample_dtype_bytes": 4,
            "profile_parameter_upper_estimates": profile_parameter_estimates,
        },
        "breakdown": counts,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=["A", "B", "C", "all"], required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-jobs", type=int, default=500)
    parser.add_argument("--allow-large-grid", action="store_true")
    parser.add_argument("--allow-disabled-experiment", action="store_true")
    parser.add_argument("--override-readiness-gate", action="store_true")
    args = parser.parse_args()
    if args.max_jobs <= 0:
        parser.error("--max-jobs must be positive")
    resolved_info = load_resolved_config(args.config)
    cfg = resolved_info.resolved
    if args.experiment == "all":
        declared = cfg.get("experiments", {})
        experiments = [
            name
            for name in ("A", "B", "C")
            if name in declared
            and (args.allow_disabled_experiment or declared[name].get("enabled") is True)
        ]
    else:
        experiments = [args.experiment]
    rows: list[dict[str, Any]] = []
    for exp in experiments:
        rows.extend(
            rows_for_experiment(
                exp,
                resolved_info.raw,
                args.config,
                allow_disabled=args.allow_disabled_experiment,
                resolved_info=resolved_info,
                override_readiness_gate=args.override_readiness_gate,
            )
        )
    report = job_breakdown(rows, cfg)
    report["max_jobs_guard"] = {
        "threshold": int(args.max_jobs),
        "override_enabled": bool(args.allow_large_grid),
        "within_limit": len(rows) <= int(args.max_jobs),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if len(rows) > args.max_jobs and not args.allow_large_grid:
        raise SystemExit(
            f"Refusing to write {len(rows)} jobs because --max-jobs={args.max_jobs}; "
            "review the breakdown and pass --allow-large-grid to proceed"
        )
    ensure_dir(Path(args.out).parent)
    with open(args.out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=JOB_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} jobs to {args.out}")


if __name__ == "__main__":
    main()
