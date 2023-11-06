import argparse
from src.model import LoadAthenaPKRun
from src.commons import load_config_file

def main():
    parser = argparse.ArgumentParser(description="Get average value of a field.")
    parser.add_argument("run", help="AthenaPK run directory")
    args = parser.parse_args()

    # Load run and configuration
    run = args.run
    sim = LoadAthenaPKRun(run)

    # corr_time = sim.get_run_integral_time()
    # print(corr_time)

    sim.get_run_statistics()

    # Get average value of desired fields
    # config_file = load_config_file("config.yaml")
    # for field in config_file["fields_for_analysis"]:
    #     print(f"Running analysis for field: {field}")
    #     sim.get_run_average_fields(args.field, in_time=True)

if __name__ == "__main__":
    main()
