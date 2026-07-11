"""Image-transfer dataset APIs.

Builder imports are lazy so pure manifest and job-grid tooling can run on login
nodes that do not have PyTorch installed.
"""

from __future__ import annotations

from .dataset_identity import (
    DATASET_IDENTITY_SCHEMA_VERSION,
    DatasetIdentityError,
    build_dataset_identity,
    freeze_dataset_identity,
    load_dataset_identity,
    validate_dataset_identity,
    verify_dataset_identity,
    verify_dataset_identity_file,
)
from .manifests import (
    COMBINED_MANIFEST_SCHEMA_VERSION,
    MANIFEST_SCHEMA_VERSION,
    SPLIT_MANIFEST_SCHEMA_VERSION,
    SUBSET_MANIFEST_SCHEMA_VERSION,
    ManifestError,
    ManifestInsufficientDataError,
    auxiliary_training_subset,
    build_data_manifest,
    build_split_manifest,
    build_subset_manifest,
    canonical_sha256,
    combine_manifests,
    config_hash,
    equal_total_feasibility,
    load_manifest,
    manifest_path,
    persist_or_validate_manifest,
    resolve_manifest_seeds,
    split_manifest_path,
    subset_manifest_path,
    target_training_subset,
    validate_manifest,
    validate_split_manifest,
    validate_subset_manifest,
)

__all__ = [
    "DatasetBundle",
    "build_datasets_for_job",
    "count_available_target_images",
    "DATASET_IDENTITY_SCHEMA_VERSION",
    "DatasetIdentityError",
    "build_dataset_identity",
    "freeze_dataset_identity",
    "load_dataset_identity",
    "validate_dataset_identity",
    "verify_dataset_identity",
    "verify_dataset_identity_file",
    "MANIFEST_SCHEMA_VERSION",
    "SPLIT_MANIFEST_SCHEMA_VERSION",
    "SUBSET_MANIFEST_SCHEMA_VERSION",
    "COMBINED_MANIFEST_SCHEMA_VERSION",
    "ManifestError",
    "ManifestInsufficientDataError",
    "build_data_manifest",
    "build_split_manifest",
    "build_subset_manifest",
    "combine_manifests",
    "canonical_sha256",
    "config_hash",
    "target_training_subset",
    "auxiliary_training_subset",
    "equal_total_feasibility",
    "validate_manifest",
    "validate_split_manifest",
    "validate_subset_manifest",
    "load_manifest",
    "manifest_path",
    "split_manifest_path",
    "subset_manifest_path",
    "resolve_manifest_seeds",
    "persist_or_validate_manifest",
]


def __getattr__(name: str):
    if name in {"DatasetBundle", "build_datasets_for_job", "count_available_target_images"}:
        from .builders import DatasetBundle, build_datasets_for_job, count_available_target_images

        return {
            "DatasetBundle": DatasetBundle,
            "build_datasets_for_job": build_datasets_for_job,
            "count_available_target_images": count_available_target_images,
        }[name]
    raise AttributeError(name)
