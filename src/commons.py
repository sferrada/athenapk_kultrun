import os
import yaml
from typing import (Union)

def custom_column_widths():
    """ Custom widths for specific input file sections """
    return {
        "modes": (7, 2, 10),  # Custom widths for the `<modes>` section
        "hydro": (1, 14, 14),  # Custom widths for the `<hydro>` section
        "global": (10, 10, 10),  # General column widths
        "comment": (9, 10, 10),  # Custom widths for the `<comment>` section
        "problem/turbulence": (12, 8, 10),  # Custom widths for the `<problem/turbulence>` section
    }


def output_dir(run_name: str) -> str:
    """
    Get the output directory path for a specific run.

    Args:
        run_name (str): The name of the run or simulation.

    Returns:
        str: The output directory path for the specified run.
    """
    return os.path.join('outputs', run_name)


def validate_parameter(param_to_validate,
                       default):
    """
    Validate and substitute a parameter with a default value if it is set to None.
    
    Args:
        param_to_validate: The input parameter to validate.
        default: The default value to use if the parameter is set to None.
    
    Returns:
        The `param_to_validate` if it is not None, otherwise, the `default` value.
    """
    return param_to_validate if param_to_validate is not None else default


def load_config_file(config_file_path: str,
                     override_config: Union[dict, None] = None) -> dict:
    """
    Load information from a YAML configuration file into a Python dictionary,
    optionally allowing parameter overrides.

    Args:
        config_file_path (str): The path to the configuration file.
        override_config (Union[dict, None], optional): Parameters to override in the loaded configuration.
            Defaults to None.

    Returns:
        dict: A dictionary with the parsed configuration information.
    """
    _override_config = validate_parameter(override_config, default={})
    with open(config_file_path) as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    for key in _override_config:
        if key not in config.keys():
            config[key] = {}
        for subkey in _override_config[key]:
            try:
                config[key][subkey] = _override_config[key][subkey]
            except KeyError:
                config[key] = {subkey: _override_config[key][subkey]}

    try:
        if config['density_powerlaw_idx'] == 0:
            config['central_density'] = config['density_at_reference']
            config['density_at_reference'] = None
    except KeyError:
        pass

    try:
        if config['dust_temperature_powerlaw_idx'] == 0:
            config['dust_temperature'] = config['dust_temperature_at_reference']
            config['dust_temperature_at_reference'] = None
    except KeyError:
        pass
    return config


def output_directory_name(config_dict: dict,
                          prefix: str = "",
                          suffix: list = []) -> str:
    """
    Generate a unique output directory name based on the configuration settings.

    Args:
        config_dict (dict): A dictionary containing configuration settings.
        prefix (str, optional): A prefix to be added to the directory name. Defaults to an empty string.
        suffix (list, optional): A list of suffixes to be added to the directory name. Defaults to an empty list.

    Returns:
        str: A unique directory name based on the configuration.
    """
    base_name = prefix

    base_name += f"NG_{config_dict['numeric_settings']['number_of_gpus']}-"
    base_name += f"NC_{config_dict['numeric_settings']['number_of_cells']:03d}-"
    base_name += f"TCOR_{config_dict['initial_conditions']['correlation_time']:1.2f}-"
    base_name += f"SOLW_{config_dict['initial_conditions']['solenoidal_weight']:1.2f}-"
    base_name += f"ARMS_{config_dict['initial_conditions']['acceleration_field_rms']:1.2f}-"
    base_name += f"BINI_{config_dict['initial_conditions']['initial_magnetic_field']:1.2f}-"
    base_name += f"EOSG_{config_dict['initial_conditions']['equation_of_state_gamma']:1.2f}"

    if suffix:
        for s in suffix:
            base_name += f"-{s.upper()}"

    return base_name


def read_athenapk_input_file(filename: str, 
                             skip_header: bool = True) -> dict:
    """
    Read and parse an AthenaPK configuration file.

    Parameters:
        filename (str): The name of the configuration file to read.
        skip_header (bool): Whether to skip the first section (e.g., <comment>). Default is True.

    Returns:
        dict: A dictionary representing the parsed configuration.
    """
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


def modify_athenapk_input_file(config: dict,
                               modifications: list) -> dict:
    """
    Modify a configuration dictionary based on a list of specified modifications.

    Args:
        config (dict): The original configuration dictionary to be modified.
        modifications (list): A list of modifications, where each modification is a tuple
        (section, key, new_value) specifying the section, key, and the new value to be set.

    Returns:
        dict: The modified configuration dictionary.
    """
    modified_config = config.copy()
    for section, key, new_value in modifications:
        if section in modified_config:
            for i, (config_key, value, comment) in enumerate(modified_config[section]):
                if config_key == key:
                    modified_config[section][i] = (key, new_value, comment)
    return modified_config
                

def write_athenapk_input_file(filename: str,
                              config_dict: dict,
                              number_of_cells: int = 512,
                              column_widths: dict = None) -> None:
    """
    Write an AthenaPK configuration to a file with custom column padding widths and substitute `number_of_cells`.

    Parameters:
        filename (str): The name of the output configuration file.
        config_dict (dict): The configuration dictionary to write.
        number_of_cells (int, optional): The number of cells to substitute in the configuration.
        column_widths (dict, optional): Dictionary specifying custom column widths for sections.

    Returns:
        None
    """
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
                    formatted_line  = f"{key}{' ' * key_padding} = {value}{' ' * value_padding}{' ' * comment_padding}{' # ' + comment if comment else ''}"
                formatted_line += "\n"

                file.write(formatted_line)

            file.write("\n")


def format_athenapk_input_file(config_dict: dict,
                               print_formatted: bool = False) -> str:
    """
    Format an AthenaPK configuration dictionary as a string.

    Parameters:
        config_dict (dict): The configuration dictionary to format.

    Returns:
        str: The formatted configuration as a string.
    """
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
