"""Runtime configuration. Everything overridable by environment variable so the
same image runs in the APL evaluator, the APL-mirror harness and local tests."""
import json, os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name, default, cast=str):
    v = os.environ.get(name)
    return cast(v) if v not in (None, "") else default


@dataclass
class RunConfig:
    input_dir: Path = field(default_factory=lambda: Path(_env("D2_INPUT", "/app/input")))
    output_dir: Path = field(default_factory=lambda: Path(_env("D2_OUTPUT", "/app/output")))
    model_dir: Path = field(default_factory=lambda: Path(_env("D2_MODEL_DIR", "/app/model")))
    schema_path: Path = field(default_factory=lambda: Path(_env("D2_SCHEMA", "/app/schema/qual-predictions.schema.json")))
    team_name: str = field(default_factory=lambda: _env("D2_TEAM_NAME", "TEAM_NAME_TBD"))
    team_email: str = field(default_factory=lambda: _env("D2_TEAM_EMAIL", "team@example.com"))
    version: str = field(default_factory=lambda: _env("D2_SUBMISSION_VERSION", "0.1.0"))
    # Time: APL allows 3600 s. Design target 1800 s (D-26). Hard stop leaves margin for the final write.
    budget_s: float = field(default_factory=lambda: _env("D2_BUDGET_S", 3000.0, float))
    # GPU memory cap in GB, enforced in-process so an 80 GB H100 behaves like the 16 GB limit (D-27).
    gpu_cap_gb: float = field(default_factory=lambda: _env("D2_GPU_CAP_GB", 14.0, float))
    batch_size: int = field(default_factory=lambda: _env("D2_BATCH", 16, int))
    decode_workers: int = field(default_factory=lambda: _env("D2_DECODE_WORKERS", 6, int))
    tta_flip: bool = field(default_factory=lambda: _env("D2_TTA_FLIP", "1") == "1")
    device: str = field(default_factory=lambda: _env("D2_DEVICE", "auto"))
    # M13-04: at most 5 minutes of work may be lost; default checkpoint every 60 s.
    checkpoint_s: float = field(default_factory=lambda: _env("D2_CHECKPOINT_S", 60.0, float))

    @property
    def output_file(self) -> Path:
        return self.output_dir / "predictions.json"


def load_model_config(model_dir: Path) -> dict:
    """model_config.json travels with the weights: backbone, input size, normalization,
    decision-layer bias and fallback class. Missing file means no model (fallback run)."""
    p = model_dir / "model_config.json"
    return json.loads(p.read_text()) if p.exists() else {}
