#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jobflow.managers.local import run_locally

from ase_flow.stability import ReferenceStabilityWorkflow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run fixed-scaffold alloy stability analysis against multiple end-member "
            "references. The workflow relaxes all reference and alloy structures, then "
            "reports delta E_mix or delta F_mix relative to the reference simplex."
        )
    )
    parser.add_argument(
        "--reference",
        action="append",
        required=True,
        help=(
            "Path to a pure end-member reference structure. Repeat this flag for each "
            "reference, e.g. --reference Ti2CO2.POSCAR --reference Mo2CO2.POSCAR."
        ),
    )
    parser.add_argument(
        "--alloy",
        action="append",
        required=True,
        help=(
            "Path to an alloy structure. Repeat this flag to analyze multiple alloys in "
            "one run."
        ),
    )
    parser.add_argument(
        "--mixing-elements",
        nargs="+",
        default=None,
        help=(
            "Optional explicit list of mixing elements, e.g. Ti Mo Zr. If omitted, infer "
            "them from the reference entries."
        ),
    )
    parser.add_argument(
        "--reference-alias",
        action="append",
        default=None,
        help="Optional alias for a reference structure. Repeat to match --reference order.",
    )
    parser.add_argument(
        "--alloy-alias",
        action="append",
        default=None,
        help="Optional alias for an alloy structure. Repeat to match --alloy order.",
    )
    parser.add_argument(
        "--forcefield",
        default="mace",
        help="ASE calculator backend passed to relax_ase (default: mace).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model path passed to the ASE calculator.",
    )
    parser.add_argument(
        "--no-free-energy",
        action="store_true",
        help="Skip the harmonic correction and report static mixing energy only.",
    )
    parser.add_argument(
        "--mode-fe",
        default="harmonic",
        choices=("harmonic", "ideal"),
        help="Thermochemistry mode passed to relax_ase when free energy is enabled.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=298.15,
        help="Temperature in K for the free-energy correction (default: 298.15).",
    )
    parser.add_argument(
        "--pressure",
        type=float,
        default=101325.0,
        help="Pressure in Pa for ideal-gas mode (default: 101325).",
    )
    parser.add_argument(
        "--geometry",
        default="linear",
        help="Geometry string passed through to relax_ase for ideal-gas mode.",
    )
    parser.add_argument(
        "--symmetrynumber",
        type=int,
        default=2,
        help="Symmetry number passed through to relax_ase (default: 2).",
    )
    parser.add_argument(
        "--spin",
        type=int,
        default=0,
        help="Spin multiplicity parameter passed through to relax_ase (default: 0).",
    )
    parser.add_argument(
        "--root-dir",
        default=".",
        help="Root directory for local jobflow execution (default: current directory).",
    )
    parser.add_argument(
        "--output",
        default="reference_stability.json",
        help="JSON file to write the stability summary to (default: reference_stability.json).",
    )
    args = parser.parse_args()

    if args.reference_alias is not None and len(args.reference_alias) != len(args.reference):
        parser.error("--reference-alias must be repeated once per --reference.")
    if args.alloy_alias is not None and len(args.alloy_alias) != len(args.alloy):
        parser.error("--alloy-alias must be repeated once per --alloy.")

    return args


def main() -> None:
    args = parse_args()

    flow, analysis_job = ReferenceStabilityWorkflow(
        reference_filenames=args.reference,
        alloy_filenames=args.alloy,
        mixing_elements=args.mixing_elements,
        forcefield=args.forcefield,
        model=args.model,
        free_energy=not args.no_free_energy,
        mode_fe=args.mode_fe,
        temperature=args.temperature,
        pressure=args.pressure,
        geometry=args.geometry,
        symmetrynumber=args.symmetrynumber,
        spin=args.spin,
        reference_aliases=args.reference_alias,
        alloy_aliases=args.alloy_alias,
    ).build()

    responses = run_locally(
        flow,
        create_folders=True,
        root_dir=Path(args.root_dir),
        ensure_success=True,
    )
    result = responses[analysis_job.uuid][analysis_job.index].output

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)

    print(f"Wrote reference stability summary to {output_path}")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
