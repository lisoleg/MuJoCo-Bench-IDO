#!/usr/bin/env python3
"""
TOMAS End-to-End Evaluation Script — SO-ARM100 Pick-and-Place
==============================================================

v0.17.1: Standalone evaluation runner for TOMAS Agent on SO-ARM100.

This script runs the full P->C->S pipeline:
  P-Layer: VLA policy (demo-vla / openvla-7b / octo-base / pi0-base)
  C-Layer: psi-Anchor + PG-Gate physical constraints
  S-Layer: kappa-Snap MerkleChain audit + MetaQuery self-attribution

Usage:
  python benchmarks/run_tomas_eval.py [--model demo-vla] [--episodes 3] [--steps 500]
  python benchmarks/run_tomas_eval.py --model openvla-7b --episodes 5 --steps 200
  python benchmarks/run_tomas_eval.py --check-vla

Output:
  - Console summary with eta/violations/skills
  - JSON report saved to benchmarks/tomas_eval_report.json

Author: MuJoCo-Bench-IDO v0.17.1
"""

import argparse
import json
import sys
import os
import time
from pathlib import Path

# Ensure project root on path
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def main():
    parser = argparse.ArgumentParser(
        description="TOMAS End-to-End Evaluation on SO-ARM100"
    )
    parser.add_argument(
        "--model", default="demo-vla",
        choices=["demo-vla", "openvla-7b", "octo-base", "pi0-base"],
        help="VLA model to use (default: demo-vla)",
    )
    parser.add_argument(
        "--instruction", default="pick up the red cube and place it on the tray",
        help="Language instruction for VLA",
    )
    parser.add_argument(
        "--episodes", type=int, default=3,
        help="Number of episodes to run (default: 3)",
    )
    parser.add_argument(
        "--steps", type=int, default=500,
        help="Max steps per episode (default: 500)",
    )
    parser.add_argument(
        "--check-vla", action="store_true",
        help="Check VLA model availability and exit",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output JSON file path (default: benchmarks/tomas_eval_report.json)",
    )
    parser.add_argument(
        "--verbose", action="store_true", default=True,
        help="Verbose output",
    )
    args = parser.parse_args()

    # ── Check VLA availability mode ──
    if args.check_vla:
        print("\n=== VLA Model Availability Check ===\n")
        from webviz.tomas_deploy_api import check_vla_availability
        from webviz.vla_loader import VLALoader

        avail = check_vla_availability()
        loader = VLALoader()

        for name, info in avail.items():
            spec = loader.MODEL_SPECS.get(name, {})
            status = "AVAILABLE" if info.get("real_weights") else "UNAVAILABLE"
            params = spec.get("params", "N/A")
            vram = spec.get("min_vram_gb", 0)
            role = spec.get("ido_role", "N/A")

            print(f"  [{status}] {name}")
            print(f"    Params: {params}")
            print(f"    Min VRAM: {vram} GB")
            print(f"    IDO Role: {role}")
            if info.get("error"):
                print(f"    Error: {info['error']}")
            if info.get("note"):
                print(f"    Note: {info['note']}")
            print()

        # Full requirement check
        print("=== System Requirement Details ===\n")
        for name in ["openvla-7b", "octo-base", "pi0-base", "demo-vla"]:
            req = loader.check_system_requirements(name)
            print(f"  {name}: can_load={req['can_load']}")
            if req["missing"]:
                for m in req["missing"]:
                    print(f"    Missing: {m}")
            print()

        return 0

    # ── Run evaluation ──
    print(f"\n{'='*60}")
    print(f"  TOMAS End-to-End Evaluation")
    print(f"  VLA Model: {args.model}")
    print(f"  Instruction: {args.instruction}")
    print(f"  Episodes: {args.episodes}, Max Steps: {args.steps}")
    print(f"{'='*60}\n")

    from webviz.tomas_deploy_api import run_tomas_eval

    start = time.time()
    result = run_tomas_eval(
        vla_model_name=args.model,
        vla_instruction=args.instruction,
        num_episodes=args.episodes,
        max_steps=args.steps,
        verbose=args.verbose,
    )
    elapsed = time.time() - start

    # ── Print summary ──
    report = result["deploy_report"]

    print(f"\n{'='*60}")
    print(f"  Evaluation Results")
    print(f"{'='*60}")
    print(f"  Status:           {report['status']}")
    print(f"  Total Episodes:   {report['total_episodes']}")
    print(f"  Total Steps:      {report['total_steps']}")
    print(f"  Avg Eta:          {report['avg_eta']:.6f}")
    print(f"  Final Eta:        {report['final_eta']:.6f}")
    print(f"  Psi Violations:   {report['psi_violations']}")
    print(f"  Kappa-Snap Count: {report['kappa_snap_count']}")
    print(f"  MetaQueries:      {report['meta_queries_count']}")
    print(f"  Skills Learned:   {report['learned_skills_count']}")
    print(f"  Failure Attrs:    {report['failure_attributions_count']}")
    print(f"  Elapsed:          {report['elapsed_seconds']:.2f}s")
    print(f"  VLA Real Weights: {result['vla_loaded']}")
    print(f"  Total Wall Time:  {elapsed:.2f}s")
    print(f"{'='*60}\n")

    # ── Safety report ──
    safety = report.get("safety_report", {})
    if safety:
        print(f"  Safety Report:")
        print(f"    Total Violations:  {safety.get('total_violations', 0)}")
        print(f"    Chain Integrity:   {safety.get('chain_integrity', 'N/A')}")
        print(f"    Steps Executed:    {safety.get('steps_executed', 0)}")
        print(f"    Violation Rate:    {safety.get('violation_rate', 0):.4f}")
        breakdown = safety.get("violation_breakdown", {})
        if breakdown:
            print(f"    Breakdown:         {breakdown}")
        print()

    # ── Save JSON report ──
    output_path = args.output or str(
        Path(_PROJECT_ROOT) / "benchmarks" / "tomas_eval_report.json"
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)

    print(f"  Report saved to: {output_path}")
    print()

    # ── Exit code ──
    if report["status"] == "success":
        print("  RESULT: SUCCESS")
        return 0
    elif report["status"] == "partial":
        print("  RESULT: PARTIAL (some episodes completed)")
        return 0
    else:
        print("  RESULT: FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
