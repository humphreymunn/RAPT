#!/usr/bin/env python3
"""Full root-cause diagnosis: saliency → rendered evidence → LLM classification.

Generates the diagnostic image set (saliency heatmap, raw-signal history,
optional command history and camera keyframe) and queries an LLM to rank the
top-3 most likely failure categories. Use the paper's 21-class taxonomy by
default, or provide your own with --taxonomy.

Providers:
  --provider anthropic   (needs `pip install anthropic`, ANTHROPIC_API_KEY)
  --provider openai      (needs `pip install openai`, OPENAI_API_KEY)
  --provider none        (default: writes prompt.txt + images for manual use,
                          or for wiring into any other LLM API)

Examples:
  python scripts/diagnose.py checkpoints/quickstart data/sample/eval.npz --index 1
  python scripts/diagnose.py checkpoints/robot run.csv --provider anthropic \\
      --command-dims cmd_vel_x cmd_vel_y cmd_vel_yaw --keyframe cam.jpg
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from rapt import RaptSystem, load_sequences
from rapt.diagnosis import build_prompt, diagnose, render_diagnosis_inputs


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("checkpoint", help="checkpoint directory")
    ap.add_argument("sequence", help="a .csv/.npy file or a dataset (.npz/.h5/dir)")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--at", type=int, help="diagnose at this step (default: first detection)")
    ap.add_argument("--provider", choices=["anthropic", "openai", "none"], default="none")
    ap.add_argument("--model", help="override the provider's default model name")
    ap.add_argument("--taxonomy", help="JSON file with a custom list of failure categories")
    ap.add_argument("--command-dims", nargs="*", default=[],
                    help="dimension names to plot as command history (15 s window)")
    ap.add_argument("--keyframe", help="optional synchronized camera image")
    ap.add_argument("--out", default="diagnosis", help="output directory")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    system = RaptSystem.load(args.checkpoint, args.device)
    data = load_sequences(args.sequence)
    obs = data.obs[args.index]
    act = data.actions[args.index] if data.actions else None
    names = system.cfg.dim_names or data.dim_names
    dt = system.cfg.dt

    step = args.at
    if step is None:
        step = system.first_detection(obs, act, args.device)
        if step is None:
            print("No detection in this sequence; diagnosing at the final step.")
            step = len(obs) - 1
        else:
            print(f"Diagnosing detection at step {step} (t={step * dt:.2f} s).")

    sal = system.attribute(
        obs[: step + 1], act[: step + 1] if act is not None else None, device=args.device
    )
    h = min(system.cfg.saliency_window, step + 1)
    obs_window = obs[step + 1 - h : step + 1]

    cmd_window = cmd_names = None
    if args.command_dims:
        idx = [names.index(n) for n in args.command_dims]
        span = min(int(15 / dt), step + 1)  # paper: 15 s of command history
        cmd_window, cmd_names = obs[step + 1 - span : step + 1, idx], args.command_dims

    images = render_diagnosis_inputs(
        sal, obs_window, args.out, names, dt=dt,
        command_window=cmd_window, command_names=cmd_names, keyframe=args.keyframe,
    )
    print(f"Rendered {len(images)} evidence images into {args.out}/")

    categories = None
    if args.taxonomy:
        categories = json.loads(Path(args.taxonomy).read_text())

    if args.provider == "none":
        system_prompt, user_prompt = build_prompt(categories)
        prompt_path = Path(args.out) / "prompt.txt"
        prompt_path.write_text(f"[SYSTEM]\n{system_prompt}\n\n[USER]\n{user_prompt}\n")
        print(f"Wrote {prompt_path}. Attach the images and prompt to any multimodal LLM,")
        print("or rerun with --provider anthropic|openai for automatic classification.")
        return

    if args.provider == "anthropic":
        from rapt.diagnosis import anthropic_llm

        llm = anthropic_llm(**({"model": args.model} if args.model else {}))
    else:
        from rapt.diagnosis import openai_llm

        llm = openai_llm(**({"model": args.model} if args.model else {}))

    result = diagnose(llm, images, categories)
    (Path(args.out) / "llm_response.txt").write_text(result.raw_response)
    print("\nRanked root-cause classification:")
    for rank, (num, name) in enumerate(result.ranked, 1):
        print(f"  Rank {rank}: [{num}] {name}")
    if not result.ranked:
        print("  (could not parse ranked categories — see llm_response.txt)")
    print(f"Full response saved to {Path(args.out) / 'llm_response.txt'}")


if __name__ == "__main__":
    main()
