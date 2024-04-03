import numpy as np

def magnetic_energy_1(field, data):
    """ Calculate the magnetic energy density """
    return data["gas", "magnetic_energy_density"]  # * data["gas", "cell_volume"])

def magnetic_energy_2(field, data):
    """ Calculate the magnetic energy. Is this, however, the magnetic energy density instead? """
    mu_0 = 4 * np.pi * 1e-7  # Vacuum permeability
    return 0.5 * (data["gas", "magnetic_field_strength"]**2 / mu_0)

def turbulent_energy(field, data):
    """ Calculate the turbulent energy """
    return 0.5 * data["gas", "density"] * data["gas", "velocity_magnitude"]**2

def kinetic_energy_density(field, data):
    """ Calculate the kinetic energy density """
    return data["gas", "kinetic_energy_density"] * data["gas", "cell_volume"]

def velocity_rms(field, data):
    """ Calculate the root mean square velocity """
    return np.sqrt(
        data["gas", "velocity_x"]**2
        + data["gas", "velocity_y"]**2
        + data["gas", "velocity_z"]**2
    )

def mean_squared_ratio(field1, field2, data):
    """ Calculate the mean squared ratio """
    ad = data.ds.all_data()
    field1_mean = ad.quantities.weighted_average_quantity(field1, ("index", "volume"))
    field2_mean = ad.quantities.weighted_average_quantity(field2, ("index", "volume"))

    return field1_mean / field2_mean

def add_field(data, func, sampling_type="local"):
    """
    Add a field to the dataset

    :param data: The dataset to add the field to
    :param func: The function that calculates the field
    :param sampling_type: The sampling type of the field
    :return: The field that was added to the dataset """
    unit_map = {
        "magnetic_energy_1": "dyne/cm**2",  # "*cm",
        "magnetic_energy_2": "G**2",  # "dyne*cm",
        "turbulent_energy": "dyne*cm",
        "kinetic_energy_density": "erg/cm**3",
        "velocity_rms": "cm/s",
        "mean_squared_ratio": "dimensionless",
    }

    data.add_field(
        ("gas", func.__name__),
        function=func,
        sampling_type=sampling_type,
        units=unit_map[func.__name__],
    )

    return data.all_data()[("gas", func.__name__)]

# # Get snapshot's average kinetic energy density
# kinetic_energy_density = self.get_snapshot_field_average(snap_id, ("gas", "kinetic_energy_density"))
# 
# # Calculate the turbulent energy from the average kinetic energy density
# turbulent_energy = kinetic_energy_density * ds.domain_width[0] ** 3

