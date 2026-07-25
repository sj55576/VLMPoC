"""SOP YAML loader with basic graph validation."""
from pathlib import Path

import yaml

from .models import SOPDefinition


def load_sop(path: Path) -> SOPDefinition:
    """Read a SOP, rejecting duplicate IDs and dangling next steps."""
    with path.open(encoding="utf-8") as file:
        sop = SOPDefinition.model_validate(yaml.safe_load(file))
    ids = [s.id for s in sop.steps]
    if len(ids) != len(set(ids)):
        raise ValueError("SOP contains duplicate step IDs")
    for step in sop.steps:
        if step.on_success and step.on_success not in ids:
            raise ValueError(f"SOP step {step.id} points to missing {step.on_success}")
    return sop
