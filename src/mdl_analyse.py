import os
import h5py
from src.commons import (load_config_file,
                         read_athenapk_input_file)
from src.mdl_simcls_athenapk import SimAthenaPK

def parse_input_file(input_file: str) -> dict:
    """
    Parse and filter an AthenaPK input file to a dictionary with only the relevant parameters.

    :param input_file: str, path to the input file.
    :return: dict, parsed and filttered dictionary. """
    input_dict = read_athenapk_input_file(input_file)

    parsed_dict = {}
    for section, options in input_dict.items():
        parsed_dict[section] = {}
        for key, value, _ in options:
            parsed_dict[section][key] = value

    filtered_dict = {
        "time_limit": float(parsed_dict["parthenon/time"]["tlim"]),
        "cycle_limit": int(parsed_dict["parthenon/time"]["nlim"]),
        "cells_number": int(parsed_dict["parthenon/mesh"]["nx1"]),
        "box_length_x1": float(parsed_dict["parthenon/mesh"]["x1max"]),
        "box_length_x2": float(parsed_dict["parthenon/mesh"]["x2max"]),
        "box_length_x3": float(parsed_dict["parthenon/mesh"]["x3max"]),
        "eos_type": str(parsed_dict["hydro"]["eos"]),
        "eos_gamma": float(parsed_dict["hydro"]["gamma"]),
        "initial_density": float(parsed_dict["problem/turbulence"]["rho0"]),
        "initial_pressure": float(parsed_dict["problem/turbulence"]["p0"]),
        "initial_magnetic_field": float(parsed_dict["problem/turbulence"]["b0"]),
        "magnetic_field_configuration": float(parsed_dict["problem/turbulence"]["b_config"]),
        "correlation_time": float(parsed_dict["problem/turbulence"]["corr_time"]),
        "solenoidal_weight": float(parsed_dict["problem/turbulence"]["sol_weight"]),
        "acceleration_field_rms": float(parsed_dict["problem/turbulence"]["accel_rms"])
    }

    return filtered_dict

def analyse_run(
        run_dir: str,
        weight: float,
        output_file: str
    ) -> None:    
    """
    Main function for analyzing a run.

    :param run_dir: str, path to the run directory.
    :param weight: float, weight for the average.
    :param output_file: str, path to the output file.
    :return: None """    
    # Load run
    sim = SimAthenaPK(run_dir)

    # Parse input file as a dictionary
    input_file = os.path.join(run_dir, "turbulence_philipp.in")
    input_dict = parse_input_file(input_file)

    # Get and add only post-analysis parameters from the config file
    config_dict = load_config_file("config.yaml")
    input_dict["post_analysis"] = config_dict["post_analysis"]

    # Get run statistics
    run_statistics = sim.get_run_statistics()

    # Get average value of desired fields
    field_weight = weight if weight else input_dict["post_analysis"]["weight"]
    fields_for_analysis = input_dict["post_analysis"]["fields_for_analysis"]
    average_values = sim.get_run_average_fields(
        fields_for_analysis,
        weight=field_weight,
        in_time=True,
        verbose=True
    )

    # Save everything in an HDF5 file
    output_path = os.path.join(run_dir, output_file)
    with h5py.File(output_path, "w") as f:
        # Save `input_dict` as attributes of the root group
        for key, value in input_dict.items():
            if isinstance(value, str):
                f.attrs[key] = value
            else:
                f.attrs[key] = str(value)

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


