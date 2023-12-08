import argparse
from src.model import LoadAthenaPKRun
from src.commons import load_config_file

def main():
    parser = argparse.ArgumentParser(description="Perform a simple analysis routine on a run.")
    parser.add_argument("run", help="Simulation run directory")
    parser.add_argument("--weight", help="Weight for the average", default=None)
    args = parser.parse_args()

    # Load run and configuration
    run = args.run
    sim = LoadAthenaPKRun(run)

    # # Todo : Is this working as intended??
    # corr_time = sim.get_run_integral_time()

    # Print run statistics
    sim.get_run_statistics()

    # # Get average value of desired fields
    # config_dict = load_config_file("config.yaml")
    # field_weight = args.weight if args.weight else config_dict["post_analysis"]["weight"]
    # sim.get_run_average_fields(
    #     config_dict["post_analysis"]["fields_for_analysis"],
    #     weight=field_weight,
    #     verbose=True,
    #     in_time=True,
    #     save_data=True
    # )

if __name__ == "__main__":
    main()

