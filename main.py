#!/usr/bin/env python
"""
    AthenaPK Simulation Preparation and Analysis Script for KULTRUN

    This script is used to prepare and analyse runs of the AthenaPK code in
    the KULTRUN cluster at Universidad de Concepción. It is intended to be
    used as a command line tool, and it is the main entry point for the
    package.

    Authors:
    - Simón Ferrada-Chamorro (simon.fch@protonmail.com), Nowhere.

    Usage:
    Execute the script with the required and optional command-line arguments.
    - Preparing a run:
    $ athenapk_kultrun --exec-mode prepare --input turbulence_philipp.in --script submit.sh
    - Analysing a run:
    $ athenapk_kultrun --exec-mode analyse --run run --weight 1.0 --output analysis.h5

    Disclaimer:
    This script is provided "as is," without any warranty. The authors assume no
    liability for any damages, whether direct or indirect, arising from the correct
    or incorrect usage of this script.

    Dependencies:
    - Python 3.x
    - gcc >= 12.2.0
    - openmpi >= 4.1.5
    - hdf5 >= 1.14.1-2_openmpi-4.1.5_parallel
    - cuda >= 12.2
"""
import argparse
from src.mdl_prepare import prepare_run
# from src.mdl_analyse import analyse_run


def main(
    exec: str = None,
    infile: str = None,
    config: str = None,
    script: str = None,
    rundir: str = None,
    weight: float = None,
    output: str = None
):
    """Main entry point for the script."""
    match exec:
        case "prepare":
            prepare_run(infile,
                        config,
                        script)
        # case "analyse":
        #     analyse_run(rundir,
        #                 weight,
        #                 output)
        case _:
            raise ValueError(f"Unknown execution mode: {exec}")


def parser() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="AthenaPK preparation and analysis helper for Kultrun.")

    parser.add_argument("--exec-mode", choices=["prepare", "analyse"], required=True, default="prepare",
                        help="Execution mode: prepare or analyse")

    parser.add_argument("--input", type=str, metavar="FILE", default="turbulence_philipp.in",
                        help="Template input file")

    parser.add_argument("--config", type=str, metavar="FILE", default="config.yaml",
                        help="Configuration file for the run")

    parser.add_argument("--script", type=str, metavar="FILE", default="submit_run.sh",
                        help="[UNUSED] Submission script file")

    parser.add_argument("--name", type=str, metavar="NAME", default="run",
                        help="[UNUSED] Run name")

    parser.add_argument("--rundir", type=str, metavar="DIR", default="run",
                        help="Simulation run directory")

    parser.add_argument("--weight", type=float, metavar="FLOAT", default=None,
                        help="Weight for the average")

    parser.add_argument("--output", type=str, metavar="FILE", default="analysis.h5",
                        help="Output file with results")

    parser.add_argument("--overwrite", action="store_true",
                        help="[UNUSED] Overwrite output directory")

    parser.add_argument("--verbose", action="store_true",
                        help="[UNUSED] Verbose output")

    return parser.parse_args()


if __name__ == "__main__":
    # Handle command line arguments
    args = parser()

    # Call main function with parsed arguments
    main(
        exec=args.exec_mode,
        infile=args.input,
        config=args.config,
        script=args.script,
        rundir=args.rundir,
        weight=args.weight,
        output=args.output
    )

