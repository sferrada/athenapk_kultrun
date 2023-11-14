import argparse
from src.model import LoadAthenaPKRun
from src.commons import load_config_file

def main():
    parser = argparse.ArgumentParser(description="Get average value of a field.")
    parser.add_argument("run", help="AthenaPK run directory")
    parser.add_argument("--weight", help="Weight for the average", default=None)
    args = parser.parse_args()

    # Load run and configuration
    run = args.run
    sim = LoadAthenaPKRun(run)

    # # Todo : Is this working as intended??
    # corr_time = sim.get_run_integral_time()

    # # Todo : Finalise this function!
    # sim.get_run_statistics()

    # Get average value of desired fields
    # # Todo : add `field_weight = ("gas", config_file["post_analysis"]["weight"])`
    config_file = load_config_file("config.yaml")
    for field in config_file["post_analysis"]["fields_for_analysis"]:
        print(f"Running analysis for field: {field}")
        sim.get_run_average_fields(args.field, in_time=True)  # Todo : add `weight=field_weight`

if __name__ == "__main__":
    main()
