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
from scripts.mdl_prepare import prepare_run
from scripts.mdl_analyse import analyse_run

def main():
    """ Main function for the AthenaPK in Kultrun package. """
    parser = argparse.ArgumentParser(description=" ")
    parser.add_argument("action", help="Action to perform", choices=["prepare", "analyse"])

    # Arguments for preparing a run
    parser.add_argument("--input", help="Template input file", default="turbulence_philipp.in")
    parser.add_argument("--config", help="Configuration file for the run", default="config.yaml")
    parser.add_argument("--script", help="Submission script file", default="submit_run.sh")
    # parser.add_argument("--name", help="Run name", default="run")  # Todo : implement
    # parser.add_argument("--verbose", help="Verbose output", action="store_true")  # Todo : implement
    # parser.add_argument("--overwrite", help="Overwrite output directory", action="store_true")  # Todo : implement

    # Arguments for analyzing a run
    parser.add_argument("--run", help="Simulation run directory")
    parser.add_argument("--weight", help="Weight for the average", default=None)
    parser.add_argument("--output", help="Output file name for analysis results", default="analysis.h5")
    # parser.add_argument("--script", help="Submission script file", default="submit_analysis.sh")  # Todo : implement

    # Parse arguments
    args = parser.parse_args()

    # Perform the requested action
    if args.action == "prepare":
        prepare_run(
            args.input,
            args.config,
            args.script
        )
    elif args.action == "analyse":
        analyse_run(
            args.run,
            args.weight,
            args.output
        )
    else:
        raise ValueError(f"Unknown action: {args.action}")

if __name__ == "__main__":
    main()

