import yt
import matplotlib.pyplot as plt

def plot_snapshot_field_map(self,
                            snap_id: int | str,
                            field: tuple[str, str],
                            normal: str = "z",
                            method: str = "slice",
                            color_map: str = "viridis",
                            overplot_velocity: bool = False,
                            **kwargs: dict) -> None:
    """
    This function creates slice or projection plots of simulation data using yt,
    returning the resulting plot object.

     Args:
        snap_id (int or str): The snapshot ID to plot.
        field (tuple of str): A tuple specifying the field to plot (e.g., ('gas', 'density')).
        normal (str, optional): The axis for slicing or project (e.g., 'z'). Defaults to 'z'.
        method (str, optional): The plotting method ('slice' or 'projection'). Defaults to 'slice'.
        color_map (str, optional): The colormap to use for visualization. Defaults to 'viridis'.
        overplot_velocity (bool, optional): If True, overplot the velocity field. Defaults to False.
        **kwargs (dict, optional): Additional keyword arguments to pass to the yt plot.

     Returns:
        yt.SlicePlot or yt.ProjectionPlot: The yt plot object representing the field slice or projection."""
    ds = self.__load_snapshot_data__(snap_id)

    if method.lower() == "slice":
        _plot = yt.SlicePlot(ds, normal, field, **kwargs)
    elif method.lower() == "projection":
        _plot = yt.ProjectionPlot(ds, normal, field, **kwargs)
    else:
        raise ValueError("Invalid method. Supported methods are 'slice' and 'projection'.")

    _plot.set_cmap(field=field, cmap=color_map)

    if overplot_velocity:
        coords = "xyz"
        indices = coords.index(normal)
        velocity_coords = coords[:indices] + coords[indices + 1:]
        _plot.annotate_quiver(
            ("gas", f"velocity_{velocity_coords[0]}"),
            ("gas", f"velocity_{velocity_coords[1]}"),
            color='green',
            factor=16
        )

    return _plot

def plot_snapshot_power_spectra(self, snap_id: int | str) -> None:
    """
    Plot the power spectra of the velocity and magnetic fields of a snapshot.

    Args:
        snap_id (int or str): The snapshot ID to plot."""
    ds = self.__load_snapshot_data__(snap_id)
    ad = ds.all_data()

    # Calculate the power spectra of the velocity, kinetic and magnetic fields
    # using the yt built-in function.
    kinetic_power_spectrum = yt.create_profile(ad, "kinetic_energy", "cell_volume",
                                               logs={"kinetic_energy": True},
                                               n_bins=64, weight_field=None)
    velocity_power_spectrum = yt.create_profile(ad, "velocity_magnitude", "cell_volume",
                                                logs={"velocity_magnitude": True},
                                                n_bins=64, weight_field=None)
    magnetic_power_spectrum = yt.create_profile(ad, "magnetic_field_strength", "cell_volume",
                                                logs={"magnetic_field_strength": True},
                                                n_bins=64, weight_field=None)

    # Calculate forcing spectrum

    # Plot the power spectra.
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].plot(kinetic_power_spectrum.x, kinetic_power_spectrum["kinetic_energy"])
    ax[0].set_xlabel(r"$k$")
    ax[0].set_ylabel(r"$E(k)$")
    ax[0].set_title("Kinetic Power Spectrum")
    ax[1].plot(velocity_power_spectrum.x, velocity_power_spectrum["velocity_magnitude"])
    ax[1].set_xlabel(r"$k$")
    ax[1].set_ylabel(r"$E(k)$")
    ax[1].set_title("Velocity Power Spectrum")
    ax[2].plot(magnetic_power_spectrum.x, magnetic_power_spectrum["magnetic_field_strength"])
    ax[2].set_xlabel(r"$k$")
    ax[2].set_ylabel(r"$E(k)$")
    ax[2].set_title("Magnetic Power Spectrum")
    plt.show()
