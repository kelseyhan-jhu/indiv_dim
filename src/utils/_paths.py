from pathlib import Path

HOME_DIR = Path(__file__).resolve().parent.parent.parent
assert (HOME_DIR / "src").exists(), f"Invalid project directory: {HOME_DIR}"
