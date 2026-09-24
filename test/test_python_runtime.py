from pathlib import Path

from dvrk_newton import python_runtime


def test_explicit_interpreter_is_saved_and_reused(tmp_path, monkeypatch):
    interpreter = (tmp_path / "venv" / "bin" / "python").absolute()
    monkeypatch.setenv("DVRK_NEWTON_PYTHON", str(interpreter))
    monkeypatch.setattr(
        python_runtime, "_imports_newton", lambda candidate: candidate == interpreter
    )

    explicit = python_runtime.resolve_newton_python(tmp_path)
    assert explicit.path == interpreter
    assert explicit.source == "DVRK_NEWTON_PYTHON"
    assert (tmp_path / "python-runtime.json").is_file()

    monkeypatch.delenv("DVRK_NEWTON_PYTHON")
    cached = python_runtime.resolve_newton_python(tmp_path)
    assert cached.path == interpreter
    assert cached.source == f"saved selection in {tmp_path / 'python-runtime.json'}"
