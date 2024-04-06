import os
import re
import datetime
import pandas as pd

map_attributes = {
    "NG": "numberOfGPUs",
    "NC": "numberOfCells",
    "TCOR": "correrationTime",
    "SOLW": "solenoidalWeight",
    "ARMS": "accelerationFieldRMS",
    "BINI": "initialMagneticField",
    "EOSG": "equationOfStateGamma",
    "MFM": "magneticFieldMode",
}

def extract_attributes(folder_name):
    """Extract attributes from folder name."""
    attributes = {}
    regex = re.compile(r'([^_]+)_([^_-]+)')
    mfm_default = '4'  # Default value for MFM
    mfm_found = False  # Flag to check if MFM is explicitly mentioned
    for match in regex.finditer(folder_name):
        key = match.group(1).lstrip('-')  # Remove leading "-" if present
        value = match.group(2)
        # Handle special case for "-MFM"
        if key == "MFM":
            mfm_found = True
            if not value:
                value = int(mfm_default)
        # Convert value to int or float if possible
        if '.' in value:
            try:
                value = float(value)
            except ValueError:
                pass
        else:
            try:
                value = int(value)
            except ValueError:
                pass
        attributes[map_attributes[key]] = value
    
    # If MFM is not explicitly mentioned, set default value
    if not mfm_found:
        attributes[map_attributes["MFM"]] = mfm_default
    
    return attributes

def get_creation_date(file_path):
    """Get the creation date of a file."""
    if os.path.exists(file_path):
        creation_timestamp = os.path.getctime(file_path)
        creation_datetime = datetime.datetime.fromtimestamp(creation_timestamp)
        return creation_datetime.strftime("%Y/%m/%d[%H:%M:%S]")
    else:
        return None

def determine_status(files):
    """Determine the status of a simulation based on the files present."""
    if not files['parthenon.prim.00000.phdf']:
        return "NOT-RUN"
    elif not files['parthenon.prim.final.phdf']:
        return "INCOMPLETE"
    # elif files['analysis.h5'] and files['analysis.h5'] < files['parthenon.prim.00000.phdf']:
    #     return "OUTDATED"
    else:
        return "COMPLETE"

def check_simulation_status(root_folder):
    """Check the status of all simulations in a folder."""
    results = []
    for folder_name in os.listdir(root_folder):
        simulation_folder = os.path.join(root_folder, folder_name)
        if os.path.isdir(simulation_folder):
            attributes = extract_attributes(folder_name)
            files = {
                'parthenon.prim.00000.phdf': None,
                'parthenon.prim.final.phdf': None,
                'turbulence_philipp.out': None,
                'analysis.h5': None
            }
            for file_name, creation_date in files.items():
                file_path = os.path.join(simulation_folder, file_name)
                files[file_name] = get_creation_date(file_path)
            status = determine_status(files)
            results.append({'run': folder_name,
                            **attributes,
                            **files,
                            'status': status})
    return results

# Entry point of the script
if __name__ == "__main__":
    root_folder = os.path.dirname(os.path.realpath(__file__))
    outputs_folder = os.path.join(root_folder, 'outputs')
    results = check_simulation_status(outputs_folder)
    
    # Display results in tabular format
    df = pd.DataFrame(results)
    print(df)

    # Save results to CSV file
    csv_file_path = os.path.join(outputs_folder, 'simulation_results.csv')
    df.to_csv(csv_file_path, index=False)
    print(f"\nResults saved to {csv_file_path}")

