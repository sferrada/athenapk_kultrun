import sys
from src.model import LoadAthenaPKRun

# Non-parallelized version
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python get_average_value.py <run> <field>")
        sys.exit(1)

    run = sys.argv[1]
    sim = LoadAthenaPKRun(run)

    # Get average value
    sim.get_all_average_field(sys.argv[2])

