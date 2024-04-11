import os
from src.commons import load_config_file
from src.mdl_files import (
    custom_column_widths,
    output_directory_name,
    read_input_file,
    write_input_file,
    modify_input_file
)

def make_output_dir(run_dir):
    """
    Check if the directory exists, and if not, create it.

    :param run_dir: str, name of the run directory.
    """
    out_dir = os.path.join("outputs", run_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    else:
        print(f"\033[93m[WARNING]\033[0m\nDirectory '{run_dir}' already exists.\n")
        # raise FileExistsError(f"\033[91m[ERROR]\033[0m Directory '{run_dir}' already exists.")

def prepare_run(
    input_file: str,
    config_file: str,
    script_file: str
) -> None:
    """
    Main function for preparing the run directory, input file, and submission script.

    :param input_file: str, path to the template input file.
    :param config_file: str, path to the configuration file.
    :param script_file: str, path to the submission script file.
    :return: None
    """
    # Load configuration from the config file
    config_dict = load_config_file(f"config/{config_file}")
    initial_conditions = config_dict["initial_conditions"]

    # Specify the directory path
    out_dir = output_directory_name(config_dict)  # , suffix=["MFM_0"])

    # Create output directory
    make_output_dir(out_dir)

    # Read the template configuration file
    run_input = read_input_file("inputs/" + input_file, skip_header=False)

    # List of modifications to apply
    mods = [
        ("hydro", "eos", initial_conditions["equation_of_state_type"]),
        ("hydro", "gamma", initial_conditions["equation_of_state_gamma"]),
        ("problem/turbulence", "b0", initial_conditions["initial_magnetic_field"]),
        ("problem/turbulence", "rho0", initial_conditions["initial_mean_density"]),
        ("problem/turbulence", "corr_time", initial_conditions["correlation_time"]),
        ("problem/turbulence", "accel_rms", initial_conditions["acceleration_field_rms"]),
        ("problem/turbulence", "solenoidal_weight", initial_conditions["solenoidal_weight"])
    ]

    # Apply modifications to the configuration and write it down
    modified_input_data = modify_input_file(run_input, mods)
    modified_input_path = os.path.join("outputs", out_dir, "turbulence_philipp.in")
    write_input_file(
        modified_input_path,
        modified_input_data,
        custom_column_widths,
        config_dict["numeric_settings"]["number_of_cells"]
    )

    # Write the batch file for running the simulation
    write_run_batch_file(
        out_dir,
        script_file,
        config_dict
    )
    msg = f"\033[92m[Input configuration saved to]\033[0m\n"
    msg += f"{modified_input_path}\n\n"
    msg += f"\033[92m[To execute the simulation, just run]\033[0m\n"
    msg += f"$ sbatch scripts/{script_file}\n\n"

    # Conditionally write post-analysis batch file based on `run_analysis` in config file
    if config_dict["post_analysis"]["run_analysis"] == 0:
        msg += f"\033[93m[WARNING]\033[0m\n"
        msg += f"Post-analysis is turned off in the config file!"
    else:
        write_analysis_batch_file(out_dir, config_dict)
        msg += f"\033[93m[WARNING]\033[0m\n"
        msg += f"Post-analysis is turned on in the config file! However, this is a work in progress\n"
        msg += f"and nothing will be run immediately after the simulation. To run it manually, use:\n"
        msg += f"sbatch scripts/submit_analysis.sh\n"

    # Finalise
    print(msg)

def write_run_batch_file(
    out_dir: str,
    script_file: str,
    config_dict: dict
):
    """Write the batch file for running the simulation."""
    numeric_settings = config_dict["numeric_settings"]
    kultrun_modules = [
        "openmpi/4.1.5",
        "gcc/12.2.0",
        "hdf5/1.14.1-2_openmpi-4.1.5_parallel",
        "cuda/12.2",
        "Python/3.11.4"
    ]

    with open(script_file, "w") as fh:
        fh.writelines("#!/bin/bash\n")
        fh.writelines(f"#SBATCH --job-name={numeric_settings['slurm_job_name']}\n")
        fh.writelines(f"#SBATCH --partition=kurruf_gpu\n")  # Always to be executed on the GPU node
        fh.writelines(f"#SBATCH --nodes={numeric_settings['number_of_nodes']}\n")
        fh.writelines(f"#SBATCH --ntasks-per-node={numeric_settings['number_of_tasks']}\n")
        fh.writelines(f"#SBATCH --gres=gpu:A100:{numeric_settings['number_of_gpus']}\n\n")
        # fh.writelines(f"#SBATCH --mem={numeric_settings['max_memory']}\n")
        fh.writelines("# Load modules\n")
        fh.writelines("module load " + " ".join(kultrun_modules) + "\n\n")
        fh.writelines("# Set directory names\n")
        fh.writelines(f"HOME_DIR={os.environ['HOME']}\n")
        fh.writelines(f"REPO_DIR={os.environ['HOME']}/athenapk_kultrun\n")
        fh.writelines(f"SIM_DIR={out_dir}\n")
        fh.writelines("OUT_DIR=outputs/${SIM_DIR}\n\n")
        fh.writelines("# Run the sim\n")
        fh.writelines("cd $REPO_DIR\n")
        fh.writelines("mpirun ./athenapk/build-host/bin/athenaPK -i ./${OUT_DIR}/turbulence_philipp.in -d\
./${OUT_DIR}/ > './${OUT_DIR}/turbulence_philipp.out'\n\n")

def write_analysis_batch_file(
    out_dir: str,
    config_dict: dict
):
    """Write the batch file for running the post-analysis."""
    analysis_val = config_dict["post_analysis"]["run_analysis"]
    with open("submit_analysis.sh", "w") as fa:
        fa.writelines("#!/bin/bash\n")
        fa.writelines(f"#SBATCH --job-name=athenapk_analysis\n")
        fa.writelines(f"#SBATCH --partition=mapu\n\n")  # Always to be executed on a CPU node
        fa.writelines("# Load modules\n")
        fa.writelines("module load openmpi/4.1.5\n")
        fa.writelines("module load gcc/12.2.0\n")
        fa.writelines("module load hdf5/1.14.1-2_openmpi-4.1.5_parallel\n")
        fa.writelines("module load cuda/12.2\n\n")
        fa.writelines(f"# Set directory names\n")
        fa.writelines(f"REPO_DIR={os.environ['HOME']}/athenapk_kultrun\n")
        fa.writelines(f"SIM_DIR={out_dir}\n")
        fa.writelines("OUT_DIR=outputs/${SIM_DIR}\n")
        fa.writelines("cd $REPO_DIR\n\n")
        if analysis_val == 1:
            fa.writelines("# Run simple post-analysis\n")
            fa.writelines("python3 scripts/mdl_analyse.py ${OUT_DIR}\n")
        elif analysis_val == 2:
            fa.writelines("# Run P.G.'s flow analysis (requires the repository!)\n")
        elif analysis_val == 3:
            fa.writelines("# Run P.G.'s energy transfer analysis (requires the repository!)\n")
        else:
            raise ValueError(f"Non-valid post-analysis method ({analysis_val}), please refer to the config file.")

# fa.writelines('srun -N 1 -n 1 python3 scripts/mdl_analyse.py ${OUT_DIR} --weight=config_dict["post_analysis"]["weight"]\n')
# fa.writelines('for X in `seq -w 00001 00049`; do srun -n 2 python3 ~/energy-transfer-analysis/mdl_analyse.py --res 256 --data_path ${OUT_DIR}/$X.phdf --data_type AthenaPK --type flow --eos adiabatic --gamma 1.0001 --outfile ${OUT_DIR}/flow-$X.hdf5 -forced; done\n')
# fa.writelines('for X in `seq -w 00001 00049`; do srun -n 2 python3 ~/energy-transfer-analysis/mdl_analyse.py --res 256 --data_path ${OUT_DIR}/$X.phdf --data_type AthenaPK --type flow --eos adiabatic --gamma 1.0001 --outfile ${OUT_DIR}/flow-$X.hdf5 -forced; done\n')
