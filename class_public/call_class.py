"""Compatibility entry point; implementation lives in the project root."""
from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parents[1] / "call_class.py"), run_name="__main__")
