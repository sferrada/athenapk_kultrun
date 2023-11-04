import os
import sys
import argparse
from src.commons import (
    load_config_file,
    output_directory_name,
    read_athenapk_input_file,
    write_athenapk_input_file,
    modify_athenapk_input_file
)

def mkdir_p(_dir):
    """ Check if the directory exists, and if not, create it. """
    if not os.path.exists(_dir):
        os.makedirs(_dir)
    else:
        print(f"ERROR: directory '{_dir}' already exists.")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Prepares the run directory, input file and submission script.")
    parser.add_argument("input_file", help="Template input configuration file")
    parser.add_argument("--config_file", "-C", default="config.yaml", help="YAML file with configuration settings")
    parser.add_argument("--script_file", "-S", default="submit_run.sh", help="SLURM script bash file")
    args = parser.parse_args()

    # Generates input file from a given template
    input_file = args.input_file
    config_file = load_config_file(args.config_file)

    # Specify the directory path and create it
    output_directory = output_directory_name(config_file)
    directory_path = os.path.join("outputs", output_directory)
    mkdir_p(directory_path)

    # Read the configuration file
    run_input = read_athenapk_input_file(input_file, skip_header=False)

    # Custom column widths for specific sections
    custom_column_widths = {
        "global": (10, 10, 10),  # General column widths
        "comment": (9, 10, 10),  # Custom widths for the `<comment>` section
        "hydro": (1, 14, 14),  # Custom widths for the `<hydro>` section
        "problem/turbulence": (12, 8, 10),  # Custom widths for the `<problem/turbulence>` section
        "modes": (7, 2, 10)  # Custom widths for the `<modes>` section
    }

    # List of modifications to apply
    modifications = [
        ("hydro", "eos", config_file["equation_of_state"]),
        ("problem/turbulence", "b0", config_file["initial_magnetic_field"]),
        ("problem/turbulence", "accel_rms", config_file["acceleration_field_rms"])
    ]

    # Apply modifications to the configuration
    modified_input = modify_athenapk_input_file(run_input, modifications)

    # Write the configuration file with custom column widths
    output_file = os.path.join("outputs", output_directory, "turbulence_philipp.in")
    write_athenapk_input_file(
        output_file,
        config_dict=modified_input,
        column_widths=custom_column_widths,
        number_of_cells=int(config_file["number_of_cells"])
    )

    # Write SLURM submitting script
    with open(args.script_file, "w") as fh:
        fh.writelines("#!/bin/bash\n")
        fh.writelines("#SBATCH --job-name=%s\n" % config_file["slurm_job_name"])
        fh.writelines("#SBATCH --partition=kurruf_gpu\n")
        fh.writelines("#SBATCH --nodes=%i\n" % config_file["number_of_nodes"])
        fh.writelines("#SBATCH --ntasks-per-node=%i\n" % config_file["number_of_tasks"])
        fh.writelines("#SBATCH --mem=%s\n" % config_file["max_memory"])
        fh.writelines("#SBATCH --gres=gpu:A100:%i\n" % config_file["number_of_gpus"])
        fh.writelines("\n")
        fh.writelines("# Load modules\n")
        fh.writelines("module load openmpi/4.1.5 gcc/12.2.0 hdf5/1.14.1-2_openmpi-4.1.5_parallel cuda/12.2\n")
        fh.writelines("\n")
        fh.writelines("# Set directory names\n")
        fh.writelines("PRJDIR=%s/athenapk_kultrun\n" % os.environ['HOME'])
        fh.writelines("RUNDIR=%s\n" % output_directory)
        fh.writelines("\n")
        fh.writelines("# Run the sim\n")
        fh.writelines("cd $PRJDIR\n")
        fh.writelines('mpirun ./athenapk/build-host/bin/athenaPK -i ./outputs/${RUNDIR}/turbulence_philipp.in -d ./outputs/${RUNDIR}/ > "./outputs/${RUNDIR}/turbulence_philipp.out"\n')
        fh.writelines("\n")
        fh.writelines("# Run post-analysis if specified in the config file for a list of fields\n")

        # Fields for post-analysis
        field_values = " ".join(config_file["fields_for_analysis"])
        fh.writelines(f'fields_for_analysis=("{field_values}")\n')        

        fh.writelines('if [[ $run_analysis = "True" ]]; then\n')
        fh.writelines("    if [[ ${#fields_for_analysis[@]} -eq 0 ]]; then\n")
        fh.writelines('        echo "No fields specified for analysis."\n')
        fh.writelines("    else\n")
        fh.writelines('        for field in "${fields_for_analysis[@]}"; do\n')
        fh.writelines('            echo "Running analysis for field: $field"\n')
        fh.writelines('            python3 scripts/get_average_value.py outputs/${RUNDIR} "$field"\n')
        fh.writelines("        done\n")
        fh.writelines("    fi\n")
        fh.writelines("else\n")
        fh.writelines('    echo "Post-run analysis is not enabled in the config."\n')
        fh.writelines("fi\n")

    print(f"Run input configuration saved to {output_file}")
    print(f"To execute the simulation, just run `sbatch {args.script_file}`")

if __name__ == "__main__":
    main()