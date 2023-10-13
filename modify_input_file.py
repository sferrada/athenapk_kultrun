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
    print(config['parthenon/mesh'])

    # # Format the config dictionary to be printed
    # format_athenapk_config_file(config, print_formatted=True)

    # Modify the configuration if needed
    number_of_cells = 128
    config['parthenon/mesh'][1]  = ('nx1', number_of_cells, 'Number of zones in X1-direction')
    config['parthenon/mesh'][6]  = ('nx2', number_of_cells, 'Number of zones in X2-direction')
    config['parthenon/mesh'][11] = ('nx3', number_of_cells, 'Number of zones in X3-direction')
    config_mod = {
        # Todo : If I do what's below, it replaces all the non-modified fields of the section
        # Todo cont'd: thus, I need to do something about it, it should only replace the wanted
        # Todo cont'd: fields and leave the rest untouched... for the moment I settled with the above
        # 'parthenon/mesh': [
        #     ('nx1', number_of_cells, 'Number of zones in X1-direction'),
        #     ('nx2', number_of_cells, 'Number of zones in X2-direction'),
        #     ('nx3', number_of_cells, 'Number of zones in X3-direction')
        # ],
        'parthenon/meshblock': [
            ('nx1', number_of_cells // 4, None),
            ('nx2', number_of_cells // 4, None),
            ('nx3', number_of_cells // 2, None)
        ]
    }

    # Write the modified configuration back to the file
    write_athenapk_config_file(output_file, config | config_mod)

    print(f"Modified configuration saved to {output_file}")

