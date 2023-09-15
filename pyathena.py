import sys
from src.commons import (read_athenapk_config_file,
                         write_athenapk_config_file,
                         format_athenapk_config_file)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <input_config_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = input_file.replace('.in', '_modified.in')

    # Read the configuration file
    config = read_athenapk_config_file(input_file)

    # Format the config dictionary to be printed
    config_formatted = format_athenapk_config_file(config)
    print(config_formatted)

    # Modify the configuration if needed
    config_mod = {
        'parthenon/output2': [
            ('file_type', 'h5', None),
            ('variables', 'acc', None),
            ('dt', '666', None),
            ('id', '420', None),
            ('single_precision_output', 'false', 'lalalala'),
        ],
    }

    # Write the modified configuration back to the file
    write_athenapk_config_file(output_file, config | config_mod)

    print(f"Modified configuration saved to {output_file}")

