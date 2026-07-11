from __future__ import annotations

from pathlib import Path

import image_transfer


MODULE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = MODULE_ROOT.parent


def test_experiment_modules_are_separate_and_imports_are_module_local():
    assert (REPOSITORY_ROOT / "gmm_simulation").is_dir()
    assert (REPOSITORY_ROOT / "image_experiment").is_dir()
    assert not (REPOSITORY_ROOT / "image_transfer").exists()
    assert not (REPOSITORY_ROOT / "tests").exists()
    assert Path(image_transfer.__file__).resolve().is_relative_to(MODULE_ROOT)


def test_continuous_integration_is_scoped_to_image_experiment():
    workflow = (REPOSITORY_ROOT / ".github" / "workflows" / "experiment-ci.yml").read_text(
        encoding="utf-8"
    )
    assert "working-directory: image_experiment" in workflow
    assert "working-directory: gmm_simulation" not in workflow
