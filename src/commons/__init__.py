def read_athenapk_config_file(filename: str) -> dict:
    """
    Read and parse an AthenaPK configuration file.

    Parameters:
        filename (str): The name of the configuration file to read.

    Returns:
        dict: A dictionary representing the parsed configuration.
    """
    config = {}
    current_section = None

    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            elif line.startswith("<") and line.endswith(">"):
                current_section = line[1:-1]
                config[current_section] = []
            elif current_section is not None:
                parts = line.split('#', 1)  # Split at the first '#' to separate key and comment
                if len(parts) == 2:
                    key_value_part, comment = map(str.strip, parts)
                else:
                    key_value_part, comment = line, None
                key, value = map(str.strip, key_value_part.split('='))
                config[current_section].append((key, value, comment))

    return config


def write_athenapk_config_file(filename: str,
                               config: dict,
                               min_column_width: int = 10) -> None:
    """
    Write an AthenaPK configuration to a file with optional column padding.

    Parameters:
        filename (str): The name of the output configuration file.
        config (dict): The configuration dictionary to write.
        min_column_width (int, optional): Minimum width for each column. Default is 10.

    Returns:
        None
    """
    with open(filename, 'w') as file:
        for section, options in config.items():
            file.write(f"<{section}>\n")
            for key, value, comment in options:
                key_padding = max(0, min_column_width - len(key))
                value_padding = max(0, min_column_width - len(value) if comment else 0)
                comment_padding = max(0, min_column_width - len(comment) if comment else 0)

                formatted_line = f"{key}{' ' * key_padding} = {value}{' ' * value_padding}"
                if comment:
                    formatted_line += f"{' ' * comment_padding} # {comment}"
                formatted_line += "\n"

                file.write(formatted_line)
            file.write("\n")


def format_athenapk_config_file(config: dict) -> str:
    """
    Format an AthenaPK configuration dictionary as a string.

    Parameters:
        config (dict): The configuration dictionary to format.

    Returns:
        str: The formatted configuration as a string.
    """
    formatted_config = ""
    for section, options in config.items():
        formatted_config += f"<{section}>\n"
        for key, value, comment in options:
            if comment:
                formatted_config += f"{key} = {value}        # {comment}\n"
            else:
                formatted_config += f"{key} = {value}\n"
        formatted_config += "\n"
    return formatted_config
