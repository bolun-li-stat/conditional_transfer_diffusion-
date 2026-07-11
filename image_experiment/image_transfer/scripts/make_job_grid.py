from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from image_transfer.config import ResolvedConfig, load_resolved_config, resolve_config
from image_transfer.data.class_sets import class_name, draw_aux_synset_combinations
from image_transfer.data.manifests import canonical_sha256
from image_transfer.models.model_factory import model_config_hash, resolve_model_config
from image_transfer.readiness import enforce_readiness_gate
from image_transfer.utils.io import ensure_dir, resolve_env_path


EXP_DIR = {"A": "A_equal_target", "B": "B_equal_total", "C": "C_similarity_sweep"}
EXP_NAME = {"A": "equal_target", "B": "equal_total", "C": "similarity_sweep"}
JOB_FIELDS = [
    "experiment",
    "experiment_name",
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


def _size_grid(exp_cfg: Mapping[str, Any], cfg: Mapping[str, Any], n0: int) -> list[tuple[int, int, int]]:
    """Return unique ``(m_per_aux, K_aux, total_auxiliary_budget)`` settings."""

    explicit_settings = exp_cfg.get("auxiliary_size_settings", cfg.get("auxiliary_size_settings"))
    if explicit_settings:
        settings: list[tuple[int, int, int]] = []
        for setting in explicit_settings:
            k_aux = int(setting["K_aux"])
            if k_aux <= 0:
                raise ValueError("K_aux must be positive in auxiliary_size_settings")
            if "m_per_aux" in setting:
                m_per_aux = int(setting["m_per_aux"])
                budget = m_per_aux * k_aux
            elif "auxiliary_ratio" in setting:
                m_per_aux = max(1, int(round(float(setting["auxiliary_ratio"]) * int(n0))))
                budget = m_per_aux * k_aux
            else:
                budget = int(setting["total_auxiliary_budget"])
                if budget <= 0 or budget % k_aux:
                    raise ValueError(
                        f"total_auxiliary_budget={budget} must be positive and divisible by K_aux={k_aux}"
                    )
                m_per_aux = budget // k_aux
            if m_per_aux <= 0:
                raise ValueError("m_per_aux must be positive in auxiliary_size_settings")
            settings.append((m_per_aux, k_aux, budget))
        return list(dict.fromkeys(settings))

    k_values = _values(exp_cfg, cfg, "K_aux_values") or [int(cfg.get("K_aux", 5))]
    budgets = _values(exp_cfg, cfg, "total_auxiliary_budget_values")
    settings: list[tuple[int, int, int]] = []
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
                settings.append((budget // k_aux, k_aux, budget))
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
                settings.append((m_per_aux, k_aux, m_per_aux * k_aux))
    return list(dict.fromkeys(settings))


def _broadcast_seed_values(values: list[Any], count: int, name: str, default_values: list[int]) -> list[int]:
    if not values:
        return list(default_values)
    if len(values) == 1:
        return [int(values[0])] * count
    if len(values) != count:
        raise ValueError(f"{name} supplies {len(values)} values for {count} replicate seeds; use one value or {count}")
    return [int(value) for value in values]


def _seed_records(cfg: Mapping[str, Any]) -> list[dict[str, int]]:
    design = cfg.get("seed_design")
    if design:
        holdout_seed = int(design["holdout_seed"])
        sampling_seed = int(design["sampling_seed"])
        evaluation_seed = int(design["evaluation_seed"])
        return [
            {
                "data_split_seed": holdout_seed,
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


RUN_SPEC_FIELDS = (
    "dataset", "experiment", "target_synset", "model_type", "aux_set", "aux_composition", "aux_draw_id",
    "n0", "m_per_aux", "K_aux", "total_auxiliary_budget", "training_protocol", "holdout_seed",
    "training_subset_seed", "model_initialization_seed", "training_seed", "sampling_seed", "evaluation_seed",
    "sampler", "sampling_steps", "architecture", "architecture_profile", "model_config_hash",
    "resolved_config_hash", "study_plan_hash", "target_set_hash", "environment_lock_hash",
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


def run_id_for_row(row: Mapping[str, Any]) -> str:
    parts = [
        row["experiment"],
        row["target_synset"],
        row["model_type"],
        row["aux_set"],
        f"draw{row['aux_draw_id']}",
        f"n0{row['n0']}",
        f"m{row['m_per_aux']}",
        f"k{row['K_aux']}",
        f"hs{row['holdout_seed']}",
        f"ts{row['training_subset_seed']}",
        f"mi{row['model_initialization_seed']}",
        f"tr{row['training_seed']}",
        row["training_protocol"],
        row.get("architecture", "legacy_simple_unet"),
        row.get("architecture_profile", "legacy"),
        f"mh{str(row.get('model_config_hash', ''))[:12]}",
        row["sampler"],
        f"ss{row['sampling_seed']}",
        f"ev{row['evaluation_seed']}",
        f"th{str(row.get('target_set_hash', ''))[:12]}",
        f"eh{str(row.get('environment_lock_hash', ''))[:12]}",
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
    split_key = canonical_sha256(
        {
            "schema": "split-key-v2",
            "dataset": cfg.get("dataset", "cifar10"),
            "target_synset": target_synset,
            "holdout_seed": int(seeds["holdout_seed"]),
            "target_eval_size": int(split.get("target_eval_size", 500)),
            "target_val_size": int(split.get("target_val_size", 100)),
            "auxiliary_eval_size": int(split.get("auxiliary_eval_size", 100)),
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
        **{name: int(value) for name, value in seeds.items()},
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
            for m_per_aux, k_aux, total_auxiliary_budget in _size_grid(exp_cfg, cfg, n0):
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
    for row in rows:
        row.update(gate_fields)
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
    report: dict[str, Any] = {
        "estimated_jobs": len(rows),
        "estimated_checkpoints": 2 * len(rows),
        "estimated_sample_storage_bytes": sample_bytes,
        "estimated_sample_storage_gib": round(sample_bytes / (1024**3), 3),
        "breakdown": counts,
    }
    runtime_minutes = cfg.get("pilot_runtime_minutes_per_job")
    if runtime_minutes is not None:
        report["estimated_gpu_hours"] = round(len(rows) * float(runtime_minutes) / 60.0, 2)
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
