"""Failure-mode taxonomy for LLM-based semantic diagnosis.

``DEFAULT_TAXONOMY`` is the 21-class taxonomy used in the RAPT paper. Users
can pass their own list of categories to the diagnosis pipeline; category
numbering is 1-based to match the paper and the prompt template.
"""

from __future__ import annotations

DEFAULT_TAXONOMY: list[str] = [
    "Motor Failure (Torque loss / Saturation)",
    "Sensor Failure (Dropout / Disconnect)",
    "Sensor Noise (High variance)",
    "Motor Dynamics Mismatch (Lag / Friction)",
    "Initial State Issue (Bad pose / velocity)",
    "Observation Scaling Issue",
    "Observation Ordering Issue",
    "Joint Outside Limits",
    "Ground Friction Mismatch",
    "Ground Deformability Mismatch (Sand / Mat)",
    "External Force (Collision / Push)",
    "Mass Distribution Mismatch",
    "Policy Latency (Offset / Jitter)",
    "IMU Coordinate Frame Mismatch",
    "Sensor Drift (Bias accumulation)",
    "Contact Model Mismatch",
    "Physical Obstruction / External Joint Constraint",
    "Policy Action Scaling Mismatch",
    "Payload Mismatch",
    "Power Supply Lag (Voltage drop)",
    "Other (Unclassified)",
]


def format_taxonomy(categories: list[str]) -> str:
    """Render a numbered category list for inclusion in the prompt."""
    return "\n".join(f"{i}. {name}" for i, name in enumerate(categories, start=1))
