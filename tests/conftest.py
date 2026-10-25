import pytest


@pytest.fixture(autouse=True)
def isolated_output_dir(tmp_path, monkeypatch):
    """Point every test at its own empty output directory."""
    out = tmp_path / "output"
    monkeypatch.setenv("DATAGEN_OUTPUT_DIR", str(out))
    return out
