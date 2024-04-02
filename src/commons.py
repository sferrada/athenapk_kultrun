import yaml

def validate_parameter(
        param_to_validate,
        default
    ):
    """
    Validate and substitute a parameter with a default value if it is set to None.

    :param param_to_validate: The input parameter to validate.
    :param default: The default value to use if the parameter is set to None.
    :return: The `param_to_validate` if it is not None, otherwise, the `default` value. """
    return param_to_validate if param_to_validate is not None else default

def load_config_file(
        config_file_path: str,
        override_config: dict = None
    ) -> dict:
    """
    Load information from a YAML configuration file into a Python dictionary,
    optionally allowing parameter overrides.

    :param config_file_path: str, the path to the configuration file.
    :param override_config: dict, parameters to override in the loaded configuration.
    :return: dict, a dictionary with the parsed configuration information. """
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

    return config

