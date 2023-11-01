import argparse
from src.model import LoadAthenaPKRun

def main():
    parser = argparse.ArgumentParser(description="Get average value of a field.")
    parser.add_argument("run", help="AthenaPK run directory")
    parser.add_argument("field", help="Field to calculate average for")
    parser.add_argument("--weight", help="Weight for the average (default: None)", default=None)
    args = parser.parse_args()

    weight = None
    if args.weight is not None:
        weight = ("gas", args.weight)

    # Get average value, non-parallelised version
    run = args.run
    sim = LoadAthenaPKRun(run)
    sim.get_run_average_fields(args.field, weight=weight, in_time=True)

if __name__ == "__main__":
    main()
