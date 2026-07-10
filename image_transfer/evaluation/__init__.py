"""Evaluation public API."""

from .classifier_fidelity import (
    evaluate_classifier_fidelity,
    imagenet_synset_to_index,
    preflight_classifier_fidelity,
)
from .corruption_bank import (
    CorruptionBank,
    CorruptionRecord,
    create_corruption_bank,
    evaluate_corruption_bank,
    generate_corruption_bank,
    load_corruption_bank,
    save_corruption_bank,
)
from .feature_metrics import (
    MetricBackendError,
    MetricComputationError,
    compute_feature_metrics,
    preflight_feature_metric_backend,
    real_feature_cache_key,
)
from .fid_kid import compute_fid_kid
from .nearest_neighbors import compute_memorization_diagnostics, make_memorization_grid
from .prdc import compute_prdc

__all__ = [
    "CorruptionBank",
    "CorruptionRecord",
    "MetricBackendError",
    "MetricComputationError",
    "compute_feature_metrics",
    "compute_fid_kid",
    "compute_memorization_diagnostics",
    "compute_prdc",
    "create_corruption_bank",
    "evaluate_classifier_fidelity",
    "evaluate_corruption_bank",
    "generate_corruption_bank",
    "imagenet_synset_to_index",
    "load_corruption_bank",
    "make_memorization_grid",
    "preflight_classifier_fidelity",
    "preflight_feature_metric_backend",
    "real_feature_cache_key",
    "save_corruption_bank",
]
