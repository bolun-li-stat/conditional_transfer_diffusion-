from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from image_transfer.data.class_sets import class_name, select_aux_synsets
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
    "n0",
    "m_per_aux",
    "K_aux",
    "seed",
    "model_type",
    "config_path",
    "output_dir",
    "run_id",
]


def _m_per_aux(cfg: dict, n0: int) -> int:
    if cfg.get("m_per_aux_rule", "equal_n0") == "equal_n0":
        return n0
    return int(cfg.get("m_per_aux", n0))


def _row(exp: str, cfg: dict, config_path: str, target: dict, n0: int, m_per_aux: int, k_aux: int, seed: int, model_type: str, aux_set: str, aux_synsets: list[str]) -> dict:
    outroot = resolve_env_path(cfg.get("output_root"), "image_transfer_results")
    target_synset = target.get("synset") or target.get("name")
    run_id = f"{exp}_{target_synset}_{model_type}_{aux_set}_n0{n0}_m{m_per_aux}_k{k_aux}_seed{seed}"
    return {
        "experiment": exp,
        "experiment_name": EXP_NAME[exp],
        "dataset": cfg.get("dataset", "cifar10"),
        "target_synset": target_synset,
        "target_name": target.get("name") or class_name(target_synset),
        "aux_set": aux_set,
        "aux_composition": json.dumps(aux_synsets),
        "n0": n0,
        "m_per_aux": m_per_aux,
        "K_aux": k_aux,
        "seed": seed,
        "model_type": model_type,
        "config_path": str(config_path),
        "output_dir": str(Path(outroot) / EXP_DIR[exp]),
        "run_id": run_id,
    }


def rows_for_experiment(exp: str, cfg: dict, config_path: str) -> list[dict]:
    k_aux = int(cfg.get("K_aux", 5))
    rows: list[dict] = []
    exp_cfg = cfg.get("experiments", {}).get(exp, {})
    n0_values = exp_cfg.get("n0_values", cfg.get("n0_values", [100]))
    seeds = cfg.get("seeds", [0])
    targets = cfg.get("targets", [{"synset": "dog", "name": "dog"}])
    for target in targets:
        auxiliary_sets = target.get("auxiliary_sets") or cfg.get("auxiliary_sets", {})
        for n0 in n0_values:
            m_per_aux = _m_per_aux(cfg, int(n0))
            for seed in seeds:
                if exp == "A":
                    rows.append(_row(exp, cfg, config_path, target, int(n0), m_per_aux, k_aux, int(seed), "unconditional_n0", "none", []))
                    for aux_set in exp_cfg.get("aux_sets", ["close", "medium", "far", "mix"]):
                        aux_synsets = select_aux_synsets(auxiliary_sets, "mix" if aux_set == "mix" else aux_set, k_aux)
                        rows.append(_row(exp, cfg, config_path, target, int(n0), m_per_aux, k_aux, int(seed), f"conditional_{aux_set}", aux_set, aux_synsets))
                elif exp == "B":
                    # This baseline is unique per N_total; do not duplicate it for every aux_set.
                    rows.append(_row(exp, cfg, config_path, target, int(n0), m_per_aux, k_aux, int(seed), "unconditional_equal_total", "none", []))
                    for aux_set in exp_cfg.get("aux_sets", ["close", "medium", "far", "mix"]):
                        aux_synsets = select_aux_synsets(auxiliary_sets, "mix" if aux_set == "mix" else aux_set, k_aux)
                        rows.append(_row(exp, cfg, config_path, target, int(n0), m_per_aux, k_aux, int(seed), f"conditional_{aux_set}", aux_set, aux_synsets))
                elif exp == "C":
                    rows.append(_row(exp, cfg, config_path, target, int(n0), m_per_aux, k_aux, int(seed), "unconditional_n0", "none", []))
                    for composition in exp_cfg.get("compositions", ["close_only", "mostly_close", "balanced_mix", "mostly_far", "far_only"]):
                        aux_synsets = select_aux_synsets(auxiliary_sets, composition, k_aux)
                        rows.append(_row(exp, cfg, config_path, target, int(n0), m_per_aux, k_aux, int(seed), f"similarity_{composition}", composition, aux_synsets))
                else:
                    raise ValueError(f"Unknown experiment {exp}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=["A", "B", "C", "all"], required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    experiments = ["A", "B", "C"] if args.experiment == "all" else [args.experiment]
    rows: list[dict] = []
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
