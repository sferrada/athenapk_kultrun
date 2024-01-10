import os
import h5py
import argparse
from src.commons import read_athenapk_input_file
from src.simclass_athenapk import SimAthenaPK

def parse_input_file(input_file: str) -> dict:
    """
    Parse and filter an AthenaPK input file to a dictionary with only the relevant parameters.

    Parameters:
        input_file (str): The path to the input file.

    Returns:
        dict: The parsed dictionary.
    """
    input_dict = read_athenapk_input_file(input_file)

    parsed_dict = {}
    for section, options in input_dict.items():
        parsed_dict[section] = {}
        for key, value, _ in options:
            parsed_dict[section][key] = value

    filtered_dict = {
        "time_limit": parsed_dict["parthenon/time"]["tlim"],
        "cycle_limit": parsed_dict["parthenon/time"]["nlim"],
        "cells_number": parsed_dict["parthenon/mesh"]["nx1"],
        "box_length_x1": parsed_dict["parthenon/mesh"]["x1max"],
        "box_length_x2": parsed_dict["parthenon/mesh"]["x2max"],
        "box_length_x3": parsed_dict["parthenon/mesh"]["x3max"],
        "eos_type": parsed_dict["hydro"]["eos"],
        "eos_gamma": parsed_dict["hydro"]["gamma"],
        "initial_density": parsed_dict["problem/turbulence"]["rho0"],
        "initial_pressure": parsed_dict["problem/turbulence"]["p0"],
        "initial_magnetic_field": parsed_dict["problem/turbulence"]["b0"],
        "magnetic_field_configuration": parsed_dict["problem/turbulence"]["b_config"],
        "correlation_time": parsed_dict["problem/turbulence"]["corr_time"],
        "solenoidal_weight": parsed_dict["problem/turbulence"]["sol_weight"],
        "acceleration_field_rms": parsed_dict["problem/turbulence"]["accel_rms"]
    }

    return filtered_dict

def main():
    parser = argparse.ArgumentParser(description="Perform a simple analysis routine on a run.")
    parser.add_argument("run", help="Simulation run directory")
    parser.add_argument("--weight", help="Weight for the average", default=None)
    parser.add_argument("--output", help="Output file name for analysis results", default="analysis.h5")
    args = parser.parse_args()

    # Load run
    sim = SimAthenaPK(args.run)

    # Parse input file as a dictionary
    input_file = os.path.join(args.run, "turbulence_philipp.in")
    config_dict = parse_input_file(input_file)
    # for key, value in config_dict.items():
    #     print(f"{key:>30}: {value}")

    # # Calculate correlation time
    # corr_time_vector = sim.get_run_integral_times()

    # Get run statistics
    run_statistics = sim.get_run_statistics()

    # Get average value of desired fields
    field_weight = args.weight if args.weight else config_dict["post_analysis"]["weight"]
    fields_for_analysis = config_dict["post_analysis"]["fields_for_analysis"]
    average_values = sim.get_run_average_fields(
        fields_for_analysis,
        weight=field_weight,
        in_time=True,
        verbose=True
    )

    # Save everything in an HDF5 file
    output_path = os.path.join(args.run, args.output)
    with h5py.File(output_path, "w") as f:
        # Save `config_dict` as attributes of the root group
        for key, value in config_dict.items():
            f.attrs[key] = value

        # Save the run statistics dictionary
        for key, value in run_statistics.items():
            group = f.create_group(key)
            group.attrs["target"] = float(value["target"])
            group.attrs["actual"] = float(value["actual"])
            group.attrs["stand_dev"] = float(value["std"])

        # Save `average_values` columns as datasets of a group
        average_values_group = f.create_group("average_values")
        for column in average_values.columns:
            average_values_group.create_dataset(column, data=average_values[column].values)

if __name__ == "__main__":
    main()

