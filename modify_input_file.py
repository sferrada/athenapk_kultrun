import os
import sys
from src.commons import (
    load_config_file,
    output_directory_name,
    read_athenapk_config_file,
    write_athenapk_config_file,
    modify_athenapf_config_file
)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <input_config_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    config_file = load_config_file('config.yaml')
    output_directory = output_directory_name(config_file)

    # Specify the directory path you want to create
    directory_path = os.path.join("outputs", output_directory)

    # Check if the directory exists, and if not, create it
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    else:
        print(f"ERROR: directory '{directory_path}' already exists.")
        sys.exit(1)

    # Read the configuration file
    config = read_athenapk_config_file(input_file, skip_header=False)

    # Custom column widths for specific sections
    custom_column_widths = {
        'global': (10, 10, 10),  # General column widths
        'comment': (9, 10, 10),  # Custom widths for the '<comment>' section
        'hydro': (1, 14, 14),  # Custom widths for the '<hydro>' section
        'problem/turbulence': (12, 8, 10),  # Custom widths for the '<problem/turbulence>' section
        'modes': (7, 2, 10)  # Custom widths for the '<modes>' section
    }

    # List of modifications to apply
    modifications = [
        ('hydro', 'eos', config_file["equation_of_state"]),  # Modify the equation of stat in <hydro>
        ('problem/turbulence', 'b0', config_file["initial_magnetic_field"]),
        ('problem/turbulence', 'accel_rms', config_file["acceleration_field_rms"])
    ]

    # Apply modifications to the configuration
    modified_config = modify_athenapf_config_file(config, modifications)

    # Write the configuration file with custom column widths
    output_file = os.path.join("outputs", output_directory, "turbulence_philipp.in")
    write_athenapk_config_file(
        output_file,
        config_dict=modified_config,
        column_widths=custom_column_widths,
        number_of_cells=int(config_file["number_of_cells"])
    )

    print(f"Modified configuration saved to {os.path.join(output_file)}")

