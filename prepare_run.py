#!/usr/bin/env python
"""
AthenaPK Simulation Preparation and Execution Script for KULTRUN

This script prepares the run directory, input file, and submission script for
the AthenaPK astrophysical simulation code. It generates a modified input file,
customized SLURM script, and optionally a post-analysis batch file.

Authors:
- Simón Ferrada-Chamorro (simon.fch@protonmail.com), Nowhere.
- [Co-Author Name]

Disclaimer:
This script is provided "as is," without any warranty. The authors assume no
liability for any damages, whether direct or indirect, arising from the correct
or incorrect usage of this script.

Usage:
- Execute the script with the required and optional command-line arguments.

Example:
$ python script_name.py input_template.in --config_file=config.yaml --script_file=submit_run.sh

Dependencies:
- Python 3.x
- gcc >= 12.2.0
- openmpi >= 4.1.5
- hdf5 >= 1.14.1-2_openmpi-4.1.5_parallel
- cuda >= 12.2
"""
import argparse
from scripts.prepare_run import prepare_run

def main():
    """
    Main function for preparing the run directory, input file, and submission script.
    """
    parser = argparse.ArgumentParser(description="Prepares the run directory, input file and submission script.")
    parser.add_argument("input_file", help="Template input configuration file")
    parser.add_argument("--config_file", "-C", default="config.yaml", help="YAML configuration file")
    parser.add_argument("--script_file", "-S", default="submit_run.sh", help="SLURM script bash file")
    args = parser.parse_args()

    # Call the main function from `prepare_run`
    prepare_run(
        args.input_file,
        args.config_file,
        args.script_file
    )

if __name__ == "__main__":
    main()

