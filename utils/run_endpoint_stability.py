#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jobflow.managers.local import run_locally

from ase_flow.stability import EndpointStabilityWorkflow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run endpoint-referenced alloy stability analysis against two end members. "
            "The workflow relaxes the two endpoint structures and the alloy structure, "
            "then reports delta E_mix or delta F_mix relative to the endpoint tie line."
        )
    )
    parser.add_argument("--endmember-a", required=True, help="Path to endpoint A structure.")
    parser.add_argument("--endmember-b", required=True, help="Path to endpoint B structure.")
    parser.add_argument("--alloy", required=True, help="Path to alloy structure.")
    parser.add_argument(
        "--mixing-elements",
        nargs=2,
        default=None,
        help="Optional explicit mixing element pair, e.g. Ti Mo. If omitted, infer from endpoints.",
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
        "--alias-endmember-a",
        default="endmember_a",
        help="Alias label stored for endpoint A.",
    )
    parser.add_argument(
        "--alias-endmember-b",
        default="endmember_b",
        help="Alias label stored for endpoint B.",
    )
    parser.add_argument(
        "--alias-alloy",
        default="alloy",
        help="Alias label stored for the alloy.",
    )
    parser.add_argument(
        "--root-dir",
        default=".",
        help="Root directory for local jobflow execution (default: current directory).",
    )
    parser.add_argument(
        "--output",
        default="endpoint_stability.json",
        help="JSON file to write the stability summary to (default: endpoint_stability.json).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    flow, analysis_job = EndpointStabilityWorkflow(
        endmember_a_filename=args.endmember_a,
        endmember_b_filename=args.endmember_b,
        alloy_filename=args.alloy,
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
        alias_endmember_a=args.alias_endmember_a,
        alias_endmember_b=args.alias_endmember_b,
        alias_alloy=args.alias_alloy,
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

    print(f"Wrote endpoint stability summary to {output_path}")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
