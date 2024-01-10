import os
from src.commons import (
    load_config_file,
    custom_column_widths,
    output_directory_name,
    read_athenapk_input_file,
    write_athenapk_input_file,
    modify_athenapk_input_file,
)

def make_output_dir(run_dir):
    """
    Check if the directory exists, and if not, create it.

    Args:
      run_dir (str): Name of the run directory."""
    out_dir = os.path.join("outputs", run_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    else:
        print(f"\033[93mWARNING:\033[0m\nDirectory '{run_dir}' already exists.\n")

def prepare_run(input_file: str,
                config_file: str,
                script_file: str):
    """
    Main function for preparing the run directory, input file, and submission script.

    Args:
      input_file (str): Path to the template input file.
      config_file (str): Path to the configuration file.
      script_file (str): Path to the submission script file.

    Returns:
      None"""
    # Generates input file from a given template
    config_dict = load_config_file(config_file)

    # Specify the directory path and create it
    out_dir = output_directory_name(config_dict)  # , suffix=["MFM_0"])
    make_output_dir(out_dir)

    # Read the template configuration file
    run_input = read_athenapk_input_file(input_file, skip_header=False)

    # List of modifications to apply
    modifications = [
        ("hydro", "eos", config_dict["initial_conditions"]["equation_of_state_type"]),
        ("hydro", "gamma", config_dict["initial_conditions"]["equation_of_state_gamma"]),
        ("problem/turbulence", "b0", config_dict["initial_conditions"]["initial_magnetic_field"]),
        ("problem/turbulence", "rho0", config_dict["initial_conditions"]["initial_mean_density"]),
        ("problem/turbulence", "corr_time", config_dict["initial_conditions"]["correlation_time"]),
        ("problem/turbulence", "accel_rms", config_dict["initial_conditions"]["acceleration_field_rms"]),
        ("problem/turbulence", "solenoidal_weight", config_dict["initial_conditions"]["solenoidal_weight"])
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
        number_of_cells=config_dict["numeric_settings"]["number_of_cells"]
    )

    # Write SLURM script
    with open(script_file, "w") as fh:
        fh.writelines("#!/bin/bash\n")
        fh.writelines(f"#SBATCH --job-name={config_dict['numeric_settings']['slurm_job_name']}\n")
        fh.writelines(f"#SBATCH --partition=kurruf_gpu\n")  # Always to be executed on the GPU node
        fh.writelines(f"#SBATCH --nodes={config_dict['numeric_settings']['number_of_nodes']}\n")
        fh.writelines(f"#SBATCH --ntasks-per-node={config_dict['numeric_settings']['number_of_tasks']}\n")
        # fh.writelines(f"#SBATCH --mem={config_dict['numeric_settings']['max_memory']}\n")
        fh.writelines(f"#SBATCH --gres=gpu:A100:{config_dict['numeric_settings']['number_of_gpus']}\n")
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

    msg = f"""\033[92mInput configuration saved to:\033[0m
{final_input_file}

\033[92mTo execute the simulation, just run:\033[0m
sbatch {script_file}
"""

    # Conditionally write post-analysis batch file based on `run_analysis` in config file
    analysis_val = config_dict["post_analysis"]["run_analysis"]
    if analysis_val == 0:
        msg += """
\033[93mWARNING:\033[0m
Post-analysis is turned off in the config file!"""
    else:
        msg += """
\033[93mWORK-IN-PROGRESS!\033[0m
No actual analysis will be run..."""
        with open("submit_analysis.sh", "w") as fa:
            fa.writelines("#!/bin/bash\n")
            fa.writelines(f"#SBATCH --job-name=athenapk_analysis\n")
            fa.writelines(f"#SBATCH --partition=mapu\n")  # Always to be executed on a CPU node
            fa.writelines(f"\n")
            fa.writelines(f"# Load modules\n")
            fa.writelines(f"module load openmpi/4.1.5 fftw/3.3.10_openmpi-4.1.5 hdf5/1.14.1-2_openmpi-4.1.5_parallel Python/3.11.4\n")
            fa.writelines(f"\n")
            fa.writelines(f"# Set directory names\n")
            fa.writelines(f"PRJDIR={os.environ['HOME']}/athenapk_kultrun\n")
            fa.writelines(f"RUNDIR={out_dir}\n")
            fa.writelines("OUTDIR=outputs/${RUNDIR}\n")
            fa.writelines("cd $PRJDIR\n")
            fa.writelines("\n")
            if analysis_val == 1:
                fa.writelines("# Run simple post-analysis\n")
                fa.writelines("python3 scripts/mdl_analyse.py ${OUTDIR}\n")
                # fa.writelines('srun -N 1 -n 1 python3 scripts/mdl_analyse.py ${OUTDIR} --weight=config_dict["post_analysis"]["weight"]\n')
            elif analysis_val == 2:
                fa.writelines("# Run P.G.'s flow analysis (requires the repository!)\n")
                # fa.writelines('for X in `seq -w 00001 00049`; do srun -n 2 python3 ~/energy-transfer-analysis/mdl_analyse.py --res 256 --data_path ${OUTDIR}/$X.phdf --data_type AthenaPK --type flow --eos adiabatic --gamma 1.0001 --outfile ${OUTDIR}/flow-$X.hdf5 -forced; done\n')
            elif analysis_val == 3:
                fa.writelines("# Run P.G.'s energy transfer analysis (requires the repository!)\n")
                # fa.writelines('for X in `seq -w 00001 00049`; do srun -n 2 python3 ~/energy-transfer-analysis/mdl_analyse.py --res 256 --data_path ${OUTDIR}/$X.phdf --data_type AthenaPK --type flow --eos adiabatic --gamma 1.0001 --outfile ${OUTDIR}/flow-$X.hdf5 -forced; done\n')
            else:
                raise ValueError(f"Non-valid post-analysis method ({analysis_val}), please refer to the config file.")

    print(msg)
