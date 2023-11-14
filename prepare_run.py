import os
import argparse
from src.commons import (
    load_config_file,
    custom_column_widths,
    output_directory_name,
    read_athenapk_input_file,
    write_athenapk_input_file,
    modify_athenapk_input_file,
)

def make_output_dir(run_dir):
    """ Check if the directory exists, and if not, create it. """
    out_dir = os.path.join("outputs", run_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    else:
        print(f"WARNING: directory '{out_dir}' already exists.")

def main():
    parser = argparse.ArgumentParser(description="Prepares the run directory, input file and submission script.")
    parser.add_argument("input_file", help="Template input configuration file")
    parser.add_argument("--config_file", "-C", default="config.yaml", help="YAML configuration file")
    parser.add_argument("--script_file", "-S", default="submit_run.sh", help="SLURM script bash file")
    args = parser.parse_args()

    # Generates input file from a given template
    input_file = args.input_file
    config_file = load_config_file(args.config_file)

    # Specify the directory path and create it
    out_dir = output_directory_name(config_file)
    make_output_dir(out_dir)

    # Read the template configuration file
    run_input = read_athenapk_input_file(input_file, skip_header=False)

    # List of modifications to apply
    modifications = [
        ("hydro", "eos", config_file["initial_conditions"]["equation_of_state_type"]),
        ("hydro", "gamma", config_file["initial_conditions"]["equation_of_state_gamma"]),
        ("problem/turbulence", "b0", config_file["initial_conditions"]["initial_magnetic_field"]),
        ("problem/turbulence", "rho0", config_file["initial_conditions"]["initial_mean_density"]),
        ("problem/turbulence", "corr_time", config_file["initial_conditions"]["correlation_time"]),
        ("problem/turbulence", "accel_rms", config_file["initial_conditions"]["acceleration_field_rms"]),
        ("problem/turbulence", "solenoidal_weight", config_file["initial_conditions"]["solenoidal_weight"])
    ]

    # Apply modifications to the configuration
    modified_input = modify_athenapk_input_file(run_input, modifications)

    # Write the configuration file with custom column widths
    column_widths = custom_column_widths()
    final_input_file = os.path.join("outputs", out_dir, "turbulence_philipp.in")
    write_athenapk_input_file(
        final_input_file,
        config_dict=modified_input,
        column_widths=column_widths,
        number_of_cells=config_file["numeric_settings"]["number_of_cells"]
    )

    # Write SLURM script
    with open(args.script_file, "w") as fh:
        fh.writelines("#!/bin/bash\n")
        fh.writelines(f"#SBATCH --job-name={config_file['numeric_settings']['slurm_job_name']}\n")
        fh.writelines(f"#SBATCH --partition=kurruf_gpu\n")  # Always to be executed on the GPU node
        fh.writelines(f"#SBATCH --nodes={config_file['numeric_settings']['number_of_nodes']}\n")
        fh.writelines(f"#SBATCH --ntasks-per-node={config_file['numeric_settings']['number_of_tasks']}\n")
        # fh.writelines(f"#SBATCH --mem={config_file['numeric_settings']['max_memory']}\n")
        fh.writelines(f"#SBATCH --gres=gpu:A100:{config_file['numeric_settings']['number_of_gpus']}\n")
        fh.writelines(f"\n")
        fh.writelines(f"# Load modules\n")
        fh.writelines(f"module load openmpi/4.1.5 gcc/12.2.0 hdf5/1.14.1-2_openmpi-4.1.5_parallel cuda/12.2\n")
        fh.writelines(f"\n")
        fh.writelines(f"# Set directory names\n")
        fh.writelines(f"PRJDIR={os.environ['HOME']}/athenapk_kultrun\n")
        fh.writelines(f"RUNDIR={out_dir}\n")
        fh.writelines("OUTDIR=outputs/${RUNDIR}\n")
        fh.writelines("\n")
        fh.writelines("# Run the sim\n")
        fh.writelines("cd $PRJDIR\n")
        fh.writelines('mpirun ./athenapk/build-host/bin/athenaPK -i ./${OUTDIR}/turbulence_philipp.in -d ./${OUTDIR}/ > "./${OUTDIR}/turbulence_philipp.out"\n')
        fh.writelines("\n")

        # Conditionally write post-analysis lines based on `run_analysis` in config file
        if not config_file["post_analysis"]["run_analysis"]:
            print("WARNING: Post-analysis is turned off in the config file!")
        elif config_file["post_analysis"]["run_analysis"] == 0:
            fh.writelines("# Run simple post-analysis\n")
            print("WORK-IN-PROGRESS - no actual analysis will be run!")
            # fh.writelines('srun -N 1 -n 1 python3 scripts/run_analysis.py ${OUTDIR} --weight=config_file["post_analysis"]["weight"]\n')
        elif config_file["post_analysis"]["run_analysis"] == 1:
            print("WORK-IN-PROGRESS - no actual analysis will be run!")
            fh.writelines("# Run P.G.'s flow analysis (requires the repository!)\n")
            # fh.writelines('for X in `seq -w 00001 00049`; do srun -n 2 python3 ~/energy-transfer-analysis/run_analysis.py --res 256 --data_path ${OUTDIR}/$X.phdf --data_type AthenaPK --type flow --eos adiabatic --gamma 1.0001 --outfile ${OUTDIR}/flow-$X.hdf5 -forced; done\n')
        elif config_file["post_analysis"]["run_analysis"] == 2:
            print("WORK-IN-PROGRESS - no actual analysis will be run!")
            fh.writelines("# Run P.G.'s energy transfer analysis (requires the repository!)\n")
        else:
            raise ValueError("Non-valid post-analysis method, please refer to the config file.")

    print(f"Run input configuration saved to {final_input_file}")
    print(f"To execute the simulation, just run `sbatch {args.script_file}`")

if __name__ == "__main__":
    main()