import os

def output_dir(run_name: str) -> str:
    """
    Get the output directory path for a specific run.

    :param run_name: str, the name of the run or simulation.
    :return: str, the output directory path for the specified run. """
    return os.path.join('outputs', run_name)

def output_directory_name(config_dict: dict, prefix: str = "", suffix=None) -> str:
    """
    Generate a unique output directory name based on the configuration settings.

    :param config_dict: dict, a dictionary containing configuration settings.
    :param prefix: str, a prefix to be added to the directory name.
    :param suffix: list, a list of suffixes to be added to the directory name.
    :return: str, a unique directory name based on the configuration. """
    base_name = prefix

    base_name += f"NG_{config_dict['numeric_settings']['number_of_gpus']}-"
    base_name += f"NC_{config_dict['numeric_settings']['number_of_cells']:03d}-"
    base_name += f"TCOR_{config_dict['initial_conditions']['correlation_time']:1.2f}-"
    base_name += f"SOLW_{config_dict['initial_conditions']['solenoidal_weight']:1.2f}-"
    base_name += f"ARMS_{config_dict['initial_conditions']['acceleration_field_rms']:1.2f}-"
    base_name += f"BINI_{config_dict['initial_conditions']['initial_magnetic_field']:1.2f}-"
    base_name += f"EOSG_{config_dict['initial_conditions']['equation_of_state_gamma']:1.2f}"

    if suffix is None:
        pass
    else:
        for s in suffix:
            base_name += f"-{s.upper()}"

    return base_name

def read_input_file(filename: str, skip_header: bool = True) -> dict:
    """
    Read and parse an AthenaPK configuration file.

    :param filename: str, the name of the configuration file to read.
    :param skip_header: bool, whether to skip the first section (e.g., <comment>).
                        Default is True.
    :return: dict, a dictionary representing the parsed configuration. """
    config_dict = {}
    current_section = None
    skip_lines = False

    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if skip_header and line.startswith("<comment>"):
                skip_lines = True
                continue
            if skip_lines and line == "":
                skip_lines = False

            if not line or line.startswith("#") or (skip_header and skip_lines):
                continue
            elif line.startswith("<") and line.endswith(">"):
                current_section = line[1:-1]
                config_dict[current_section] = []
            elif current_section is not None:
                parts = line.split('#', 1)
                if len(parts) == 2:
                    key_value_part, comment = map(str.strip, parts)
                else:
                    key_value_part, comment = line, None
                key, value = map(str.strip, key_value_part.split('='))
                config_dict[current_section].append((key, value, comment))

    return config_dict

def modify_input_file(config: dict, modifications: list) -> dict:
    """
    Modify a configuration dictionary based on a list of specified modifications.

    :param config: dict, the original configuration dictionary to be modified.
    :param modifications: list, a list of modifications, where each modification is a tuple
                          (section, key, new_value) specifying the section, key, and the new value to be set.
    :return: dict, the modified configuration dictionary. """
    modified_config = config.copy()
    for section, key, new_value in modifications:
        if section in modified_config:
            for i, (config_key, value, comment) in enumerate(modified_config[section]):
                if config_key == key:
                    modified_config[section][i] = (key, new_value, comment)
    return modified_config

def write_input_file(filename: str, config_dict: dict, column_widths: dict = None,
                              number_of_cells: int = 512) -> None:
    """
    Write an AthenaPK configuration to a file with custom column padding widths and substitute `number_of_cells`.

    :param filename: str, the name of the output configuration file.
    :param config_dict: dict, the configuration dictionary to write.
    :param column_widths: dict, dictionary specifying custom column widths for sections.
    :param number_of_cells: int, the number of cells to substitute in the configuration.
    :return: None """
    with open(filename, 'w') as file:
        for section, options in config_dict.items():
            file.write(f"<{section}>\n")

            if column_widths and section in column_widths:
                key_width, value_width, comment_width = column_widths[section]
            else:
                key_width, value_width, comment_width = 10, 10, 10

            for key, value, comment in options:
                key = str(key)
                value = str(value)
                comment = str(comment) if comment is not None else ""

                if section == "parthenon/mesh" and key.startswith("nx"):
                    value = str(number_of_cells)
                elif section == "parthenon/meshblock" and key.startswith("nx"):
                    if key == "nx1" or key == "nx2":
                        value = str(number_of_cells // 4)
                    elif key == "nx3":
                        value = str(number_of_cells // 2)

                key_padding = max(0, key_width - len(key))
                value_padding = max(0, value_width - len(value) if comment else 0)
                comment_padding = max(0, comment_width - len(comment) if comment else 0)

                if section == "parthenon/mesh":
                    if key == "nx2" or key == "nx3":
                        formatted_line = "\n"
                    else:
                        formatted_line = ""
                    formatted_line += f"{key}{' ' * key_padding} = {value}{' ' * value_padding}{' ' * comment_padding}{' # ' + comment if comment else ''}"
                else:
                    formatted_line = f"{key}{' ' * key_padding} = {value}{' ' * value_padding}{' ' * comment_padding}{' # ' + comment if comment else ''}"
                formatted_line += "\n"

                file.write(formatted_line)

            file.write("\n")

def format_input_file(config_dict: dict, print_formatted: bool = False) -> str:
    """
    Format an AthenaPK configuration dictionary as a string.

    :param config_dict: dict, the configuration dictionary to format.
    :param print_formatted: bool, whether to print the formatted configuration to the console.
    :return: str, the formatted configuration as a string. """
    formatted_config = ""
    for section, options in config_dict.items():
        formatted_config += f"<{section}>\n"
        for key, value, comment in options:
            if comment:
                formatted_config += f"{key} = {value}        # {comment}\n"
            else:
                formatted_config += f"{key} = {value}\n"
        formatted_config += "\n"

    if print_formatted:
        print(formatted_config)
    else:
        return formatted_config


custom_column_widths = {
    "global": (10, 10, 10),  # General column widths
    "modes": (7, 2, 10),  # Widths for the `<modes>` section
    "hydro": (1, 14, 14),  # Widths for the `<hydro>` section
    "comment": (9, 10, 10),  # Widths for the `<comment>` section
    "problem/turbulence": (12, 8, 10),  # Widths for the `<problem/turbulence>` section
}
