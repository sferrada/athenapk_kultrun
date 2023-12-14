import os
import h5py
import argparse as ap
from src.commons import load_config_file
from src.simclass_athenapk import SimAthenaPK

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
    # print("Final snapshot time scales:")
    # for key, value in times_dict.items():
    #     print("{}: {}".format(key, value))

    # Print run statistics
    run_statistics = sim.get_run_statistics()

    # Get average value of desired fields
    field_weight = args.weight if args.weight else config_dict["post_analysis"]["weight"]
    fields_for_analysis = config_dict["post_analysis"]["fields_for_analysis"]
    average_values = sim.get_run_average_fields(
        fields_for_analysis,
        weight=field_weight,
        verbose=True,
        in_time=True
    )

    # Save everything in an HDF5 file
    output_path = os.path.join(args.run, args.output)
    with h5py.File(output_path, "w") as f:
        # Save `config_dict` as attributes of the root group
        for key, value in config_dict.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    f.attrs[subkey] = subvalue
            else:
                f.attrs[key] = value

        # Save the run statistics dictionary
        for key, value in run_statistics.items():
            group = f.create_group(key)
            group.attrs["target"] = value["target"]
            group.attrs["actual"] = value["actual"]
            group.attrs["stand_dev"] = value["std"]

        # Save `average_values` columns as datasets of a group
        average_values_group = f.create_group("average_values")
        for column in average_values.columns:
            average_values_group.create_dataset(column, data=average_values[column].values)

if __name__ == "__main__":
    main()

