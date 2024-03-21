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
        athenapk_kultrun prepare [options]
    - Analysing a run:
        athenapk_kultrun analyse [options]

    Example:
    - Preparing a run:
        athenapk_kultrun prepare --input turbulence_philipp.in --config config.yaml --script submit.sh --output run --name run
    - Analysing a run:
        athenapk_kultrun analyse --run run --weight 1.0 --output analysis.h5

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
from src.mdl_analyse import analyse_run

def main():
    """
    Main function for the AthenaPK in Kultrun package. """
    parser = argparse.ArgumentParser(description="")
    # parser.add_argument("action", help="Action to perform", choices=["prepare", "analyse"])

    # Add arguments
    exec_mode = parser.add_mutually_exclusive_group(required=True)

    # Preparing a run
    exec_mode.add_argument("-prepare", "--prepare", action="store_true",
                           help="Prepare files for an AthenaPK run.")

    parser.add_argument("-input", "--input", type=str, metavar="FILE", 
                        default="turbulence_philipp.in", help="Template input file")

    parser.add_argument("-config", "--config", type=str, metavar="FILE",
                        default="config.yaml", help="Configuration file for the run",)

    parser.add_argument("-script", "--script", type=str, metavar="FILE",
                        default="submit_run.sh", help="Submission script file")

    # Todo : implement
    # parser.add_argument("--name", help="Run name", default="run")
    # parser.add_argument("--verbose", help="Verbose output", action="store_true")
    # parser.add_argument("--overwrite", help="Overwrite output directory", action="store_true")

    # Analyzing a run
    exec_mode.add_argument("-analyse", "--analyse", action="store_true",
                           help="Analyse results from an AthenaPK run")

    parser.add_argument("-run", "--run", type=str, metavar="DIR",
                        help="Simulation run directory")

    parser.add_argument("-weight", "--weight", type=float, metavar="FLOAT",
                        default=None, help="Weight for the average")

    parser.add_argument("-output", "--output", type=str, metavar="FILE",
                        default="analysis.h5", help="Output file with results")

    # Todo : implement
    # parser.add_argument("--script", help="Submission script file", default="submit_analysis.sh")

    # Parse arguments
    args = parser.parse_args()

    # Perform the requested action according to the mutually exclusive group
    if args.prepare:
        prepare_run(
            args.input,
            args.config,
            args.script
        )
    elif args.analyse:
        analyse_run(
            args.run,
            args.weight,
            args.output
        )
    else:
        raise ValueError(f"Unknown action!")

if __name__ == "__main__":
    main()

