from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Mapping

from image_transfer.data.class_sets import class_name, draw_aux_synset_combinations
from image_transfer.data.manifests import config_hash
from image_transfer.utils.io import ensure_dir, load_yaml, resolve_env_path


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
    "architecture_profile",
    "seed",  # legacy alias for training_seed
    "data_split_seed",
    "model_initialization_seed",
    "training_seed",
    "sampling_seed",
    "evaluation_seed",
    "training_protocol",
    "model_type",
    "sampler",
    "sampling_steps",
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
    explicit = cfg.get("seed_sets")
    if explicit:
        records = []
        for record in explicit:
            training_seed = int(record.get("training_seed", record.get("seed", 0)))
            records.append({
                "data_split_seed": int(record.get("data_split_seed", training_seed)),
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
    sources = {
        "data_split_seed": _as_list(cfg.get("data_split_seeds", split_cfg.get("data_split_seed"))),
        "model_initialization_seed": _as_list(cfg.get("model_initialization_seeds", cfg.get("model_initialization_seed"))),
        "training_seed": _as_list(cfg.get("training_seeds", cfg.get("training_seed"))),
        "sampling_seed": _as_list(cfg.get("sampling_seeds", sampling_cfg.get("seed"))),
        "evaluation_seed": _as_list(cfg.get("evaluation_seeds", evaluation_cfg.get("seed"))),
    }
    expanded = {
        name: _broadcast_seed_values(values, count, name, legacy) for name, values in sources.items()
    }
    return [{name: values[index] for name, values in expanded.items()} for index in range(count)]


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


def _effective_run_spec(cfg: Mapping[str, Any]) -> tuple[int, str]:
    sampling = cfg.get("sampling", {})
    diffusion = cfg.get("diffusion", {})
    sampling_steps = int(sampling.get("steps", cfg.get("sampling_steps", diffusion.get("timesteps", 1000))))
    spec_hash = config_hash(
        {
            "image_size": int(cfg.get("image_size", 32)),
            "training_steps": int(cfg.get("training", {}).get("steps", 1000)),
            "num_generated": int(cfg.get("num_generated", 64)),
            "sampling_steps": sampling_steps,
            "ddim_eta": float(sampling.get("ddim_eta", cfg.get("ddim_eta", 0.0))),
        }
    )
    return sampling_steps, spec_hash


def _run_id(row: Mapping[str, Any]) -> str:
    parts = [
        row["experiment"],
        row["target_synset"],
        row["model_type"],
        row["aux_set"],
        f"draw{row['aux_draw_id']}",
        f"n0{row['n0']}",
        f"m{row['m_per_aux']}",
        f"k{row['K_aux']}",
        f"ds{row['data_split_seed']}",
        f"mi{row['model_initialization_seed']}",
        f"tr{row['training_seed']}",
        row["training_protocol"],
        row.get("architecture_profile", "legacy"),
        row["sampler"],
        f"ss{row['sampling_seed']}",
        f"ev{row['evaluation_seed']}",
        str(row["config_hash"])[:12],
    ]
    return "_".join(_safe(part) for part in parts)


def _row(
    exp: str,
    cfg: Mapping[str, Any],
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
    cfg_hash = config_hash(cfg)
    sampler = str(cfg.get("sampler", cfg.get("sampling", {}).get("sampler", "ddpm")))
    sampling_steps, effective_spec_hash = _effective_run_spec(cfg)
    row: dict[str, Any] = {
        "experiment": exp,
        "experiment_name": EXP_NAME[exp],
        "dataset": cfg.get("dataset", "cifar10"),
        "target_synset": target_synset,
        "target_name": target.get("name") or class_name(target_synset),
        "aux_set": aux_set,
        "aux_composition": json.dumps(aux_synsets),
        "aux_draw_id": aux_draw_id,
        "aux_draw_seed": int(aux_draw_seed),
        "aux_unique_combinations": int(aux_unique_combinations),
        "n0": int(n0),
        "m_per_aux": int(m_per_aux),
        "K_aux": int(k_aux),
        "total_auxiliary_budget": int(total_auxiliary_budget),
        "auxiliary_ratio": float(m_per_aux / n0) if n0 else 0.0,
        "baseline_target_count": int(n0 + total_auxiliary_budget) if exp == "B" else int(n0),
        "architecture_profile": str(cfg.get("model", {}).get("profile", "legacy")),
        "seed": int(seeds["training_seed"]),
        **{name: int(value) for name, value in seeds.items()},
        "training_protocol": training_protocol,
        "model_type": model_type,
        "sampler": sampler,
        "sampling_steps": sampling_steps,
        "effective_run_spec_hash": effective_spec_hash,
        "config_hash": cfg_hash,
        "manifest_key": f"{exp}:{target_synset}:split{seeds['data_split_seed']}",
        "config_path": str(config_path),
        "output_dir": str(Path(outroot) / EXP_DIR[exp]),
    }
    row["run_id"] = _run_id(row)
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


def rows_for_experiment(exp: str, cfg: dict[str, Any], config_path: str) -> list[dict[str, Any]]:
    if exp not in EXP_DIR:
        raise ValueError(f"Unknown experiment {exp}")
    rows: list[dict[str, Any]] = []
    exp_cfg = cfg.get("experiments", {}).get(exp, {})
    n0_values = exp_cfg.get("n0_values", cfg.get("n0_values", [100]))
    targets = cfg.get("targets", [{"synset": "dog", "name": "dog"}])
    protocols = _protocols(exp_cfg, cfg)
    seed_records = _seed_records(cfg)
    baseline_types = (
        ["unconditional_equal_total", "conditional_target_only_equal_total"]
        if exp == "B"
        else ["unconditional_n0", "conditional_target_only_n0"]
    )
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
            row["run_id"] = _run_id(row)

    unique: dict[str, dict[str, Any]] = {}
    for row in rows:
        run_id = str(row["run_id"])
        if run_id in unique and unique[run_id] != row:
            raise ValueError(f"Conflicting jobs produced the same run_id: {run_id}")
        unique[run_id] = row
    return list(unique.values())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=["A", "B", "C", "all"], required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    experiments = ["A", "B", "C"] if args.experiment == "all" else [args.experiment]
    rows: list[dict[str, Any]] = []
    for exp in experiments:
        rows.extend(rows_for_experiment(exp, cfg, args.config))
    ensure_dir(Path(args.out).parent)
    with open(args.out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=JOB_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} jobs to {args.out}")


if __name__ == "__main__":
    main()
