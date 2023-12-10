import h5py
import numpy as np
import argparse as ap
from src.model import SimAthenaPK
from src.commons import load_config_file

def main():
    parser = ap.ArgumentParser(description="Perform a simple analysis routine on a run.")
    parser.add_argument("run", help="Simulation run directory")
    parser.add_argument("--weight", help="Weight for the average", default=None)
    parser.add_argument("--output", help="Output file name for analysis results", default="analysis.h5")
    args = parser.parse_args()

    # Load run and configuration
    sim = SimAthenaPK(args.run)
    config_dict = load_config_file("config.yaml")

    # # Calculate correlation time
    # corr_time_vector = sim.get_run_integral_times()

    # # Get final-snapshot time scales
    # times_dict = sim.get_snapshot_timescales("final")

    # # Print final-snapshot time scales
    # print("Final snapshot time scales:")
    # for key, value in times_dict.items():
    #     print("{}: {}".format(key, value))

    # # Print run statistics
    # corr_times = sim.get_run_statistics()

    # Get average value of desired fields
    field_weight = args.weight if args.weight else config_dict["post_analysis"]["weight"]
    average_values = sim.get_run_average_fields(
        config_dict["post_analysis"]["fields_for_analysis"],
        weight=field_weight,
        verbose=True,
        in_time=True
    )

    # Save configuration dictionary, mean correlation time, and average values to an HDF5 file
    with h5py.File(args.output, "w") as f:
        # Save `config_dict` as attributes of the root group
        for key, value in config_dict.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    f.attrs[subkey] = subvalue
            else:
                f.attrs[key] = value

        # # Save `corr_times` as attributes of the root group
        # f.attrs["target_corr_time"] = corr_times[0]
        # f.attrs["actual_corr_time"] = corr_times[1]
        # f.attrs["corr_time_std"] = corr_times[2]

        # Save `average_values` columns as datasets of a group
        average_values_group = f.create_group("average_values")
        for column in average_values.columns:
            average_values_group.create_dataset(column, data=average_values[column].values)

if __name__ == "__main__":
    main()

