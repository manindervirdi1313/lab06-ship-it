import json
from pathlib import Path

import gradio
import sklearn

ROOT = Path(__file__).resolve().parents[1]


def test_model_sklearn_version_matches_installed_version():
    with open(ROOT / "models" / "model_info.json") as file:
        info = json.load(file)

    assert info["sklearn_version"] == sklearn.__version__


def test_sklearn_version_is_pinned():
    requirements = (ROOT / "requirements.txt").read_text()

    assert f"scikit-learn=={sklearn.__version__}" in requirements


def test_gradio_is_not_in_runtime_requirements():
    requirements = (ROOT / "requirements.txt").read_text().lower()

    assert "gradio" not in requirements


def test_readme_gradio_version_matches_installed_version():
    readme = (ROOT / "README.md").read_text()

    assert f"sdk_version: {gradio.__version__}" in readme


def test_readme_app_file_exists():
    readme = (ROOT / "README.md").read_text()

    assert "app_file: app.py" in readme
    assert (ROOT / "app.py").exists()
