"""Prompt template for LLM-based root-cause diagnosis.

``build_prompt`` reproduces the prompt used verbatim in the RAPT paper
(Supplementary "Prompt Template"), parameterized by the failure taxonomy so
users can substitute their own categories. The "Physics of Failure" rubric
references category numbers from the default 21-class taxonomy; when a custom
taxonomy is supplied the rubric is omitted unless a custom rubric is given.
"""

from __future__ import annotations

from .taxonomy import DEFAULT_TAXONOMY, format_taxonomy

SYSTEM_ROLE = (
    "You are a Principal Robotics Engineer analyzing a failure event in "
    "Sim-to-Real transfer. You must identify the physical root cause from "
    "the provided list of {n} categories."
)

DATA_DESCRIPTION = """The Data:

1. Saliency Map (Log Scale): Shows the gradient of influence. Darker regions
   indicate higher sensitivity of the policy to that observation. Analyze
   whether saliency is concentrated in distal joints (ankles, wrists) or
   proximal joints (waist, hips), when saliency begins and how long it
   persists, and whether it aligns with changes in command velocity.

2. Joint Kinematics: Shows joint positions over time. The robot is executing
   a standard walking gait. Look for saturation, restricted motion, or joints
   railing against limits."""

DEFAULT_RUBRIC = """Physics of Failure Diagnostic Rubric:
Use the following patterns to rank the most likely root causes:
- Pattern A: Loss of Ground Reaction (Cat. 10) --- saliency shifts to the waist over long durations.
- Pattern B: High Load Compensation (Cat. 12/19) --- sustained ankle and knee effort with stable lean.
- Pattern C: Traction Loss (Cat. 9) --- high-frequency spikes in hips and ankles.
- Pattern D: External Disturbance (Cat. 11) --- short, intense global saliency unrelated to commands.
- Pattern E: Hardware Failure (Cat. 1/2/3) --- isolated, extreme saliency in a single joint.
- Pattern F: Joint Restriction (Cat. 17) --- persistent saliency with limited joint motion.
- Pattern G: Immediate Sim-to-Real Bug (Cat. 5/6/7/14/18) --- saliency at episode start.
- Pattern H: Drift or Latency (Cat. 13/15) --- slow saliency buildup localized to few signals."""

TASK_AND_FORMAT = """Task:
1. Analyze the saliency locus (waist vs. ankles/knees).
2. Analyze the kinematic behavior (stable offset vs. progressive drift).
3. Select the top three most likely failure categories.

Required Output Format:
1. Forensics Analysis
   - Saliency Locus
   - Kinematic Behavior
2. Conclusion
   - Rank 1: Category number and name, with reasoning
   - Rank 2: Category number and name
   - Rank 3: Category number and name

Failure Categories:
{taxonomy}

You must select only from the categories above; do not invent new categories."""


def build_prompt(
    categories: list[str] | None = None,
    rubric: str | None = None,
    data_description: str | None = None,
) -> tuple[str, str]:
    """Return ``(system_prompt, user_prompt_text)``.

    - ``categories``: failure taxonomy (defaults to the paper's 21 classes).
    - ``rubric``: optional custom "physics of failure" rubric. When
      ``categories`` is customized and no rubric is given, the default rubric
      (whose category numbers refer to the default taxonomy) is dropped.
    - ``data_description``: override the description of attached data, e.g.
      for non-humanoid time series.
    """
    cats = categories if categories is not None else DEFAULT_TAXONOMY
    if rubric is None:
        rubric = DEFAULT_RUBRIC if categories is None else ""
    system = SYSTEM_ROLE.format(n=len(cats))
    sections = [data_description or DATA_DESCRIPTION]
    if rubric:
        sections.append(rubric)
    sections.append(TASK_AND_FORMAT.format(taxonomy=format_taxonomy(cats)))
    return system, "\n\n".join(sections)
