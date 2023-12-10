import h5py

# Open the HDF5 file
file_path = "analysis.h5"
with h5py.File(file_path, "r") as file:
    # Print the attributes
    print("Attributes:")
    for attr_name, attr_value in file.attrs.items():
        print(f"{attr_name}: {attr_value}")

    # Get the available groups
    groups = list(file.keys())
    print(f"\nGroups: {groups}")
    # Print the datasets of each group
    for group in groups:
        print(f"\nDatasets of group '{group}':")
        for dataset_name, dataset_value in file[group].items():
            print(f"{dataset_name}: {dataset_value}")
            # Print the dataset attributes
            for attr_name, attr_value in dataset_value.attrs.items():
                print(f"    {attr_name}: {attr_value}")
            # Print the dataset values
            print(f"    {dataset_value.name}: {dataset_value[()]}")
