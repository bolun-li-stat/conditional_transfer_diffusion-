"""Canonical schema for image-transfer study configurations."""

from __future__ import annotations

import copy
import hashlib
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from image_transfer.data.manifests import canonical_sha256
from image_transfer.models.model_factory import model_config_hash, resolve_model_config


class _UniqueKeyLoader(yaml.SafeLoader):
    """YAML loader that rejects duplicate mapping keys instead of last-write-wins."""


def _construct_unique_mapping(loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ValueError(f"Duplicate YAML key {key!r} at line {key_node.start_mark.line + 1}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _load_yaml_text(text: str, *, source: str) -> Any:
    try:
        return yaml.load(text, Loader=_UniqueKeyLoader)
    except yaml.YAMLError as exception:
        raise ValueError(f"Invalid YAML in {source}: {exception}") from exception


TOP_LEVEL_FIELDS = {
    "dataset", "use_fake_data", "download", "fake_data_size", "data_root", "output_root", "image_size",
    "study_stage", "allow_single_target_case_study", "target_set_path", "targets", "data_split", "seed_design",
    "seed_sets", "seeds", "model_initialization_seeds", "training_seeds", "sampling_seeds", "evaluation_seeds",
    "training_protocols", "n0_values", "K_aux", "K_aux_values", "auxiliary_ratio_values", "m_per_aux_values",
    "total_auxiliary_budget_values", "m_per_aux", "m_per_aux_rule", "num_aux_set_draws", "aux_draw_seed",
    "capacity_profiles", "model", "diffusion", "optimizer", "training", "sampling", "evaluation",
    "num_generated", "experiments", "experiments_to_run", "analysis_plan_path", "environment_lock_path",
    "include_legacy_unconditional", "auxiliary_size_settings", "pilot_runtime_minutes_per_job",
    "metric_assets_manifest_path", "readiness_status_path", "readiness_pilot_config_path",
    # Read-only compatibility keys removed from resolved output.
    "sampling_steps", "sampler", "data_split_seeds", "training_seed", "model_initialization_seed",
}
EXPERIMENT_FIELDS = {
    "enabled", "n0_values", "aux_sets", "compositions", "K_aux_values", "auxiliary_ratio_values",
    "m_per_aux_values", "total_auxiliary_budget_values", "auxiliary_size_settings", "training_protocols",
    "num_aux_set_draws", "aux_draw_seed", "skip_if_insufficient_target_images",
    "include_legacy_unconditional",
}
SEED_DESIGN_FIELDS = {
    "holdout_seed", "training_subset_seeds", "optimization_seed_pairs", "sampling_seed", "evaluation_seed",
}
OPTIMIZATION_PAIR_FIELDS = {"model_initialization_seed", "training_seed"}
SECTION_FIELDS = {
    "data_split": {
        "manifest_root", "holdout_seed", "training_subset_seed", "data_split_seed", "target_eval_size",
        "target_val_size", "auxiliary_eval_size", "nested_training_subsets", "eval_source",
        "insufficient_data_action",
    },
    "training": {
        "protocol", "protocols", "steps", "batch_size", "target_batch_size", "auxiliary_batch_size",
        "auxiliary_loss_weight", "num_workers", "precision", "validation_interval", "checkpoint_interval",
        "ema_decay", "max_grad_norm", "rolling_loss_window",
    },
    "sampling": {"sampler", "steps", "batch_size", "ddim_eta", "seed", "guidance_scale"},
    "diffusion": {"timesteps", "schedule", "prediction_type", "reverse_variance"},
    "optimizer": {"name", "lr", "betas", "eps", "weight_decay"},
    "evaluation": {
        "mode", "corruption_bank_root", "validation_corruptions_per_image", "test_corruptions_per_image",
        "train_diagnostic_corruptions_per_image", "train_diagnostic_max_images", "corruptions_per_image",
        "noise_bins", "compute_fid", "compute_kid", "compute_prdc", "compute_classifier_fidelity",
        "compute_inception_score", "compute_cmmd", "compute_feature_similarity", "make_nearest_neighbors",
        "classifier_unavailable_reason", "classifier_synset_mapping_path", "real_eval_max", "sampling_batch_size",
        "denoising_batch_size", "classifier_batch_size", "feature_batch_size", "distance_batch_size",
        "reference_batch_size", "nearest_neighbor_reference_max", "nearest_neighbor_generated_indices",
        "nearest_neighbor_grid_items", "fid_batch_size", "kid_subset_size", "kid_num_subsets", "prdc_k",
        "image_input_range", "feature_dimension", "near_duplicate_calibration_quantile", "seed",
        "fid_reliable_min_real", "near_duplicate_calibration_max_images",
        # Compatibility aliases removed from resolved output.
        "compute_fid_kid", "compute_classifier", "eval_split", "corruptions_per_image", "nn_batch_size",
    },
}


@dataclass(frozen=True)
class ResolvedConfig:
    raw: dict[str, Any]
    resolved: dict[str, Any]
    raw_hash: str
    resolved_hash: str
    model_hash: str
    study_plan_hash: str
    target_set_hash: str
    environment_lock_hash: str
    source_path: str = ""

    def provenance(self) -> dict[str, str]:
        return {
            "raw_config_hash": self.raw_hash,
            "resolved_config_hash": self.resolved_hash,
            "model_config_hash": self.model_hash,
            "study_plan_hash": self.study_plan_hash,
            "target_set_hash": self.target_set_hash,
            "environment_lock_hash": self.environment_lock_hash,
        }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _yaml_sha256(path: Path) -> str:
    payload = _load_yaml_text(path.read_text(encoding="utf-8"), source=str(path))
    return canonical_sha256(payload)


def _load_and_validate_analysis_plan(path: Path) -> tuple[dict[str, Any], str]:
    payload = _load_yaml_text(path.read_text(encoding="utf-8"), source=str(path)) or {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"Analysis plan {path} must contain a mapping")
    payload = dict(payload)
    required = {"analysis_plan_id", "primary", "secondary", "transfer_gap_convention"}
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"Analysis plan {path} is missing fields: {missing}")
    primary = payload.get("primary")
    secondary = payload.get("secondary")
    convention = payload.get("transfer_gap_convention")
    if not isinstance(primary, Mapping) or not isinstance(secondary, Mapping) or not isinstance(convention, Mapping):
        raise ValueError(f"Analysis plan {path} has invalid section types")
    if primary.get("protocol") != "natural_compute_matched":
        raise ValueError("Analysis plan primary.protocol must be natural_compute_matched")
    comparisons = set(map(str, primary.get("comparisons", [])))
    endpoints = set(map(str, primary.get("endpoints", [])))
    required_comparisons = {
        "conditional_close_vs_conditional_target_only",
        "conditional_far_vs_conditional_target_only",
    }
    required_endpoints = {"test_epsilon_mse_target", "kid_target_mean"}
    if not required_comparisons.issubset(comparisons):
        raise ValueError("Analysis plan is missing a required primary comparison")
    if not required_endpoints.issubset(endpoints):
        raise ValueError("Analysis plan is missing a required primary endpoint")
    if convention.get("lower_is_better") != "baseline_minus_model":
        raise ValueError("Analysis plan lower-is-better convention must be baseline_minus_model")
    if convention.get("higher_is_better") != "model_minus_baseline":
        raise ValueError("Analysis plan higher-is-better convention must be model_minus_baseline")
    return payload, canonical_sha256(payload)


def _resolve_path(value: str, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base_dir / path).resolve()


def _coalesce_alias(container: dict[str, Any], alias: str, canonical: str, *, section: str) -> None:
    if alias not in container:
        return
    value = container.pop(alias)
    if canonical in container and container[canonical] != value:
        raise ValueError(f"Conflicting {section}.{alias} and {section}.{canonical} values")
    container.setdefault(canonical, value)
    warnings.warn(f"{section}.{alias} is deprecated; use {section}.{canonical}", DeprecationWarning, stacklevel=3)


def _validate_fields(config: Mapping[str, Any]) -> None:
    unknown = set(config) - TOP_LEVEL_FIELDS
    if unknown:
        raise ValueError(f"Unknown top-level config fields: {sorted(unknown)}")
    for section, allowed in SECTION_FIELDS.items():
        value = config.get(section, {})
        if not isinstance(value, Mapping):
            raise ValueError(f"{section} must be a mapping")
        extra = set(value) - allowed
        if extra:
            raise ValueError(f"Unknown {section} config fields: {sorted(extra)}")
    experiments = config.get("experiments", {})
    if not isinstance(experiments, Mapping):
        raise ValueError("experiments must be a mapping")
    for name, value in experiments.items():
        if name not in {"A", "B", "C"}:
            raise ValueError(f"Unknown experiment family {name!r}")
        if not isinstance(value, Mapping):
            raise ValueError(f"experiments.{name} must be a mapping")
        extra = set(value) - EXPERIMENT_FIELDS
        if extra:
            raise ValueError(f"Unknown experiments.{name} fields: {sorted(extra)}")
    seed_design = config.get("seed_design", {})
    if not isinstance(seed_design, Mapping):
        raise ValueError("seed_design must be a mapping")
    extra_seed_fields = set(seed_design) - SEED_DESIGN_FIELDS
    if extra_seed_fields:
        raise ValueError(f"Unknown seed_design fields: {sorted(extra_seed_fields)}")
    for index, pair in enumerate(seed_design.get("optimization_seed_pairs", [])):
        if not isinstance(pair, Mapping):
            raise ValueError(f"seed_design.optimization_seed_pairs[{index}] must be a mapping")
        extra_pair_fields = set(pair) - OPTIMIZATION_PAIR_FIELDS
        if extra_pair_fields:
            raise ValueError(
                f"Unknown seed_design.optimization_seed_pairs[{index}] fields: {sorted(extra_pair_fields)}"
            )


def _canonicalize_aliases(config: dict[str, Any]) -> None:
    sampling = config.setdefault("sampling", {})
    if "sampling_steps" in config:
        legacy = config.pop("sampling_steps")
        if "steps" in sampling and sampling["steps"] != legacy:
            raise ValueError("Conflicting sampling_steps and sampling.steps values")
        sampling.setdefault("steps", legacy)
        warnings.warn("sampling_steps is deprecated; use sampling.steps", DeprecationWarning, stacklevel=3)
    if "sampler" in config:
        legacy = config.pop("sampler")
        if "sampler" in sampling and sampling["sampler"] != legacy:
            raise ValueError("Conflicting top-level sampler and sampling.sampler values")
        sampling.setdefault("sampler", legacy)
        warnings.warn("top-level sampler is deprecated; use sampling.sampler", DeprecationWarning, stacklevel=3)

    evaluation = config.setdefault("evaluation", {})
    mode = str(evaluation.get("mode", "debug" if config.get("use_fake_data", False) else "strict")).lower()
    if mode == "paper":
        warnings.warn("evaluation.mode='paper' is deprecated; use 'strict'", DeprecationWarning, stacklevel=3)
        mode = "strict"
    if mode not in {"strict", "debug"}:
        raise ValueError("evaluation.mode must be 'strict' or 'debug'")
    evaluation["mode"] = mode
    if "compute_fid_kid" in evaluation:
        legacy = bool(evaluation.pop("compute_fid_kid"))
        for canonical in ("compute_fid", "compute_kid"):
            if canonical in evaluation and bool(evaluation[canonical]) != legacy:
                raise ValueError(f"Conflicting evaluation.compute_fid_kid and evaluation.{canonical}")
            evaluation.setdefault(canonical, legacy)
        warnings.warn("evaluation.compute_fid_kid is deprecated", DeprecationWarning, stacklevel=3)
    _coalesce_alias(evaluation, "compute_classifier", "compute_classifier_fidelity", section="evaluation")

    split = config.setdefault("data_split", {})
    if "eval_split" in evaluation:
        legacy = evaluation.pop("eval_split")
        if "eval_source" in split and split["eval_source"] != legacy:
            raise ValueError("Conflicting evaluation.eval_split and data_split.eval_source")
        split.setdefault("eval_source", legacy)
        warnings.warn("evaluation.eval_split is deprecated; use data_split.eval_source", DeprecationWarning, stacklevel=3)
    if "data_split_seed" in split:
        legacy_seed = int(split.pop("data_split_seed"))
        if "holdout_seed" in split and int(split["holdout_seed"]) != legacy_seed:
            raise ValueError("Conflicting data_split_seed and holdout_seed")
        if "training_subset_seed" in split and int(split["training_subset_seed"]) != legacy_seed:
            raise ValueError("Conflicting data_split_seed and training_subset_seed")
        split.setdefault("holdout_seed", legacy_seed)
        split.setdefault("training_subset_seed", legacy_seed)
        warnings.warn("data_split_seed is deprecated; use holdout_seed and training_subset_seed", DeprecationWarning, stacklevel=3)
    split.setdefault("holdout_seed", 0)
    split.setdefault("training_subset_seed", 0)

    if "corruptions_per_image" in evaluation:
        legacy = int(evaluation.pop("corruptions_per_image"))
        for key in (
            "validation_corruptions_per_image", "test_corruptions_per_image", "train_diagnostic_corruptions_per_image"
        ):
            if key in evaluation and int(evaluation[key]) != legacy:
                raise ValueError(f"Conflicting evaluation.corruptions_per_image and evaluation.{key}")
            evaluation.setdefault(key, legacy)
        warnings.warn("evaluation.corruptions_per_image is deprecated; use split-specific counts", DeprecationWarning, stacklevel=3)
    evaluation.setdefault("validation_corruptions_per_image", 16 if mode == "strict" else 1)
    evaluation.setdefault("test_corruptions_per_image", 16 if mode == "strict" else 1)
    evaluation.setdefault("train_diagnostic_corruptions_per_image", 8 if mode == "strict" else 1)
    evaluation.setdefault("train_diagnostic_max_images", 128)

    seed_design = config.get("seed_design")
    if seed_design:
        if any(
            key in config
            for key in (
                "seed_sets", "seeds", "model_initialization_seeds", "training_seeds", "sampling_seeds",
                "evaluation_seeds", "data_split_seeds", "training_seed", "model_initialization_seed",
            )
        ):
            raise ValueError("seed_design cannot be combined with legacy seed fields")
        if "holdout_seed" in seed_design and int(seed_design["holdout_seed"]) != int(split["holdout_seed"]):
            raise ValueError("Conflicting seed_design.holdout_seed and data_split.holdout_seed")
        seed_design.setdefault("holdout_seed", int(split["holdout_seed"]))
        split["holdout_seed"] = int(seed_design["holdout_seed"])
        subset_seeds = [int(value) for value in seed_design.get("training_subset_seeds", [split["training_subset_seed"]])]
        if not subset_seeds:
            raise ValueError("seed_design.training_subset_seeds cannot be empty")
        seed_design["training_subset_seeds"] = subset_seeds
        pairs = list(seed_design.get("optimization_seed_pairs", [{"model_initialization_seed": 0, "training_seed": 0}]))
        if not pairs:
            raise ValueError("seed_design.optimization_seed_pairs cannot be empty")
        seed_design["optimization_seed_pairs"] = [
            {
                "model_initialization_seed": int(pair.get("model_initialization_seed", pair.get("training_seed", 0))),
                "training_seed": int(pair.get("training_seed", pair.get("model_initialization_seed", 0))),
            }
            for pair in pairs
        ]
        seed_design["sampling_seed"] = int(seed_design.get("sampling_seed", sampling.get("seed", 1000)))
        seed_design["evaluation_seed"] = int(seed_design.get("evaluation_seed", evaluation.get("seed", 2000)))
        # A list belongs only in seed_design.  This scalar is retained as the
        # default for direct, non-grid invocations.
        split["training_subset_seed"] = int(subset_seeds[0])


def _validate_auxiliary_size_settings(config: Mapping[str, Any]) -> None:
    locations = [("top-level", config.get("auxiliary_size_settings"))]
    locations.extend(
        (f"experiments.{name}", value.get("auxiliary_size_settings"))
        for name, value in config.get("experiments", {}).items()
    )
    allowed = {"K_aux", "m_per_aux", "auxiliary_ratio", "total_auxiliary_budget"}
    for location, settings in locations:
        if settings is None:
            continue
        if not isinstance(settings, list) or not settings:
            raise ValueError(f"{location}.auxiliary_size_settings must be a non-empty list")
        for index, setting in enumerate(settings):
            if not isinstance(setting, Mapping):
                raise ValueError(f"{location}.auxiliary_size_settings[{index}] must be a mapping")
            unknown = set(setting) - allowed
            if unknown:
                raise ValueError(
                    f"Unknown {location}.auxiliary_size_settings[{index}] fields: {sorted(unknown)}"
                )
            if "K_aux" not in setting:
                raise ValueError(f"{location}.auxiliary_size_settings[{index}] requires K_aux")
            size_fields = set(setting) & {"m_per_aux", "auxiliary_ratio", "total_auxiliary_budget"}
            if len(size_fields) != 1:
                raise ValueError(
                    f"{location}.auxiliary_size_settings[{index}] requires exactly one size field"
                )


def _validate_target_entries(targets: list[dict[str, Any]], *, external: bool) -> None:
    if not targets:
        raise ValueError("Target set contains no targets")
    seen: set[str] = set()
    for index, target in enumerate(targets):
        if not isinstance(target, Mapping):
            raise ValueError(f"targets[{index}] must be a mapping")
        for field in ("name", "synset"):
            if target.get(field) in {None, ""}:
                raise ValueError(f"targets[{index}] is missing {field}")
        synset = str(target["synset"])
        if synset in seen:
            raise ValueError(f"Duplicate target synset {synset!r}")
        seen.add(synset)
        auxiliary = target.get("auxiliary_sets")
        if not isinstance(auxiliary, Mapping):
            raise ValueError(f"targets[{index}].auxiliary_sets must be a mapping")
        if external:
            for field in ("supercategory", "selection_rationale"):
                if target.get(field) in {None, ""}:
                    raise ValueError(f"targets[{index}] is missing {field}")
            for group in ("close", "medium", "far"):
                if not isinstance(auxiliary.get(group), list):
                    raise ValueError(f"targets[{index}].auxiliary_sets.{group} must be a list")


def _load_target_set(config: dict[str, Any], base_dir: Path) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
    target_path = config.get("target_set_path")
    if target_path:
        if config.get("targets"):
            raise ValueError("target_set_path and inline targets cannot both be set")
        path = _resolve_path(str(target_path), base_dir)
        if not path.exists():
            raise FileNotFoundError(f"Target-set file does not exist: {path}")
        payload = _load_yaml_text(path.read_text(encoding="utf-8"), source=str(path)) or {}
        if not isinstance(payload, Mapping):
            raise ValueError(f"Target-set file {path} must contain a mapping")
        payload = dict(payload)
        required = {"target_set_id", "target_set_version", "reviewed", "frozen", "targets"}
        missing = sorted(required - set(payload))
        if missing:
            raise ValueError(f"Target-set file {path} is missing fields: {missing}")
        targets = list(payload.get("targets", []))
        _validate_target_entries(targets, external=True)
        config["targets"] = targets
    else:
        payload = {
            "target_set_id": "inline",
            "target_set_version": "inline",
            "reviewed": False,
            "frozen": False,
            "targets": config.get("targets", []),
        }
        targets = list(config.get("targets", []))
        _validate_target_entries(targets, external=False)
    declared_hash = payload.pop("target_set_hash", None)
    target_hash = canonical_sha256(payload)
    if declared_hash not in {None, "", target_hash}:
        raise ValueError(f"Target-set hash mismatch: stored={declared_hash!r}, expected={target_hash!r}")
    return targets, target_hash, payload


def resolve_config(raw: Mapping[str, Any], *, source_path: str | Path | None = None) -> ResolvedConfig:
    original = copy.deepcopy(dict(raw))
    _validate_fields(original)
    resolved = copy.deepcopy(original)
    _canonicalize_aliases(resolved)
    _validate_auxiliary_size_settings(resolved)
    base_dir = Path(source_path).resolve().parent if source_path else Path.cwd()
    targets, target_hash, target_payload = _load_target_set(resolved, base_dir)
    stage = str(resolved.get("study_stage", "smoke" if resolved.get("use_fake_data") else "pilot"))
    if stage not in {"smoke", "pilot", "main", "case_study"}:
        raise ValueError(f"Unknown study_stage {stage!r}")
    resolved["study_stage"] = stage
    explicitly_staged = "study_stage" in original
    raw_model = original.get("model")
    if explicitly_staged and stage in {"pilot", "main", "case_study"}:
        if not isinstance(raw_model, Mapping) or "architecture" not in raw_model:
            raise ValueError(f"study_stage={stage!r} requires an explicit model.architecture")
        if str(raw_model.get("architecture")) != "adm_unet":
            raise ValueError(f"study_stage={stage!r} requires model.architecture='adm_unet'")
        structural_overrides = set(raw_model) - {"architecture", "profile"}
        if structural_overrides:
            raise ValueError(
                "Named ADM profiles are fixed in staged studies; remove model overrides: "
                f"{sorted(structural_overrides)}"
            )
    if stage == "main":
        if not resolved.get("target_set_path"):
            raise ValueError("study_stage='main' requires target_set_path")
        if not bool(target_payload.get("reviewed")) or not bool(target_payload.get("frozen")):
            raise ValueError("study_stage='main' requires a reviewed and frozen target set")
        allow_single = bool(resolved.get("allow_single_target_case_study", False))
        if len(targets) < 4 and not (allow_single and len(targets) == 1):
            raise ValueError("study_stage='main' requires at least four reviewed targets")
        supercategories = {str(target.get("supercategory", "")) for target in targets}
        if len(targets) >= 4 and len(supercategories - {""}) < 2:
            raise ValueError("A multi-target main study must declare more than one supercategory")
    if stage == "case_study":
        resolved["single_target_scope"] = len(targets) == 1
    resolved_model = resolve_model_config(resolved.get("model"), image_size=int(resolved.get("image_size", 32)))
    resolved["model"] = resolved_model
    diffusion = resolved.setdefault("diffusion", {})
    diffusion.setdefault("prediction_type", "epsilon")
    diffusion.setdefault("reverse_variance", "fixed")
    if diffusion["prediction_type"] != "epsilon":
        raise ValueError("diffusion.prediction_type must be 'epsilon'")
    if diffusion["reverse_variance"] != "fixed":
        raise ValueError("diffusion.reverse_variance must be 'fixed'")
    if int(diffusion.get("timesteps", 1000)) < 2:
        raise ValueError("diffusion.timesteps must be at least 2")
    if str(diffusion.get("schedule", "linear")) not in {"linear", "cosine"}:
        raise ValueError("diffusion.schedule must be 'linear' or 'cosine'")
    resolved.setdefault("sampling", {}).setdefault("guidance_scale", 1.0)
    if float(resolved["sampling"]["guidance_scale"]) != 1.0:
        raise ValueError("sampling.guidance_scale must remain 1.0")

    analysis_path = _resolve_path(
        str(resolved.get("analysis_plan_path", "analysis_plan.yaml")),
        base_dir,
    )
    plan_required = stage == "main" or resolved["evaluation"]["mode"] == "strict"
    if analysis_path.exists():
        _, study_plan_hash = _load_and_validate_analysis_plan(analysis_path)
    elif plan_required:
        raise FileNotFoundError(f"Required analysis plan does not exist: {analysis_path}")
    else:
        study_plan_hash = canonical_sha256({"analysis_plan": "not_required"})
    lock_path = _resolve_path(
        str(resolved.get("environment_lock_path", "../../environment/requirements-image-lock.txt")),
        base_dir,
    )
    environment_hash = _file_sha256(lock_path) if lock_path.exists() else "missing"
    resolved["target_set_hash"] = target_hash
    resolved["target_set_id"] = str(target_payload.get("target_set_id", "inline"))
    resolved["target_set_version"] = str(target_payload.get("target_set_version", "inline"))
    resolved["target_set_reviewed"] = bool(target_payload.get("reviewed", False))
    resolved["target_set_frozen"] = bool(target_payload.get("frozen", False))
    resolved["study_plan_hash"] = study_plan_hash
    resolved["environment_lock_hash"] = environment_hash
    raw_hash = canonical_sha256(original)
    resolved_hash = canonical_sha256(resolved)
    return ResolvedConfig(
        raw=original,
        resolved=resolved,
        raw_hash=raw_hash,
        resolved_hash=resolved_hash,
        model_hash=model_config_hash(resolved_model),
        study_plan_hash=study_plan_hash,
        target_set_hash=target_hash,
        environment_lock_hash=environment_hash,
        source_path=str(source_path or ""),
    )


def load_resolved_config(path: str | Path) -> ResolvedConfig:
    source = Path(path)
    raw = _load_yaml_text(source.read_text(encoding="utf-8"), source=str(source)) or {}
    if not isinstance(raw, Mapping):
        raise ValueError("Configuration root must be a mapping")
    return resolve_config(raw, source_path=source)
