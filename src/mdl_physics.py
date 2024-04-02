import numpy as np

def magnetic_energy_1(field, data):
    """ Calculate the magnetic energy density """
    return (data["gas", "magnetic_energy_density"])  # * data["gas", "cell_volume"])

def magnetic_energy_2(field, data):
    """ Calculate the magnetic energy. Is this, however, the magnetic energy density instead? """
    mu_0 = 4 * np.pi * 1e-7  # Vacuum permeability
    return 0.5 * (data["gas", "magnetic_field_strength"]**2 / mu_0)

def turbulent_energy(field, data):
    """ Calculate the turbulent energy """
    return 0.5 * data["gas", "density"] * data["gas", "velocity_magnitude"]**2

def kinetic_energy_density(field, data):
    """ Calculate the kinetic energy density """
    return (data["gas", "kinetic_energy_density"] * data["gas", "cell_volume"])

def velocity_rms(field, data):
    """ Calculate the root mean square velocity """
    return np.sqrt(
        data["gas", "velocity_x"]**2 +
        data["gas", "velocity_y"]**2 +
        data["gas", "velocity_z"]**2
    )

def mean_squared_ratio(field1, field2):
    pass

def add_field(data, func, sampling_type="local"):
    """
    Add a field to the dataset
    
    :param data: The dataset to add the field to
    :param func: The function that calculates the field
    :param sampling_type: The sampling type of the field
    :return: The field that was added to the dataset """
    units = {
        "magnetic_energy_1": "dyne/cm**2",
        "magnetic_energy_2": "G**2",
        "turbulent_energy": "dyne*cm",
        "kinetic_energy_density": "erg/cm**3",
        "velocity_rms": "cm/s",
        "mean_squared_ratio": "dimensionless"
    }

    data.add_field(("gas", func.__name__), function=func,
        sampling_type=sampling_type, units=units[func.__name__])

    return data.all_data()[("gas", func.__name__)]

