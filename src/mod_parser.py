import argparse

def parser():
    """Parse command line arguments."""
    parse = argparse.ArgumentParser(description="AthenaPK preparation and analysis helper for Kultrun.")

    parse.add_argument("--exec-mode", choices=["prepare", "analyse"], required=True, default="prepare",
                       help="Execution mode: prepare or analyse")

    parse.add_argument("--input", type=str, metavar="FILE", default="turbulence_philipp.in",
                       help="Template input file")

    parse.add_argument("--config", type=str, metavar="FILE", default="config.yaml",
                       help="Configuration file for the run")

    parse.add_argument("--slurm_name",
                       help="Specify the SLURM job name")

    parse.add_argument("--number_of_gpus", type=int,
                       help="Specify the number of GPUs")

    parse.add_argument("--number_of_nodes", type=int,
                       help="Specify the number of nodes per GPU")

    parse.add_argument("--number_of_tasks", type=int,
                       help="Specify the number of tasks per node")

    parse.add_argument("--number_of_cells", type=int,
                       help="Specify the number of cells")

    parse.add_argument("--max_memory",
                       help="Specify the maximum memory")

    parse.add_argument("--equation_of_state_type",
                       help="Specify the equation of state type")

    parse.add_argument("--equation_of_state_gamma", type=float,
                       help="Specify the equation of state gamma")

    parse.add_argument("--initial_mean_density", type=float,
                       help="Specify the initial mean density")

    parse.add_argument("--initial_mean_pressure", type=float,
                       help="Specify the initial mean pressure")

    parse.add_argument("--initial_magnetic_field", type=float,
                       help="Specify the initial magnetic field")

    parse.add_argument("--magnetic_field_mode", type=int,
                       help="Specify the magnetic field mode")

    parse.add_argument("--acceleration_field_rms", type=float,
                       help="Specify the acceleration field rms")

    parse.add_argument("--solenoidal_weight", type=float,
                       help="Specify the solenoidal weight")

    parse.add_argument("--correlation_time", type=float,
                       help="Specify the correlation time")

    parse.add_argument("--script", type=str, metavar="FILE", default="submit_run.sh",
                       help="[UNUSED] Submission script file")

    parse.add_argument("--name", type=str, metavar="NAME", default="run",
                       help="[UNUSED] Run name")

    parse.add_argument("--rundir", type=str, metavar="DIR", default="run",
                       help="Simulation run directory")

    parse.add_argument("--weight", type=float, metavar="FLOAT", default=None,
                       help="Weight for the average")

    parse.add_argument("--output", type=str, metavar="FILE", default="analysis.h5",
                       help="Output file with results")

    parse.add_argument("--overwrite", action="store_true",
                       help="[UNUSED] Overwrite output directory")

    parse.add_argument("--verbose", action="store_true",
                       help="[UNUSED] Verbose output")

    return parser
