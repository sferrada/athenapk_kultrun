import os
import yt
import numpy as np
from typing import Union

class Snapshot:
    def __init__(self, index, outdir):
        """
            Initialize a single snapshot instance
        """
        self.id = index  # Snapshot identifier
        self.outdir = outdir  # Path to the simulation directory

        # Get the snapshot file path
        self._path = self._get_path()

        # Instantiate snapshot data
        self.data = None

    def _get_path(self):
        """
            Return the path to the snapshot file
        """
        num_id = str(self.id).zfill(5)
        num_id += ".phdf" if not num_id.endswith(".phdf") else ""

        return os.path.join(self.outdir, f"parthenon.prim.{num_id}")

    def load(self):
        """
            Load and return snapshot data
        """
        try:
            self.data = yt.load(self._path)
        except:
            raise FileNotFoundError("")

    # def __get_snapshot_file_path__(
    #         self,
    #         snap_id: Union[int, str]
    #     ) -> str:
    #     """
    #     Get the file path for a snapshot based on its number or string representation.

    #     :param snap_id: int or str, the snapshot ID.
    #     :return: str, the file path to the snapshot. """
    #     snapshot_number_str = str(snap_id).zfill(5)

    #     if not snapshot_number_str.endswith('.phdf'):
    #         snapshot_number_str += '.phdf'

    #     # Return the full path to the snapshot file.
    #     return os.path.join(self.outdir, f'parthenon.prim.{snapshot_number_str}')

    # def __load_snapshot_data__(
    #         self,
    #         snap_id: Union[int, str]
    #     ) -> yt.data_objects.static_output.Dataset:
    #     """
    #     Load and return the data from a snapshot file.

    #     :param snap_id: int or str, the snapshot ID to load.
    #     :return: yt.data_objects.static_output.Dataset, a yt dataset containing the snapshot data. """
    #     snapshot_path = self.__get_snapshot_file_path__(snap_id)
    #     if not os.path.exists(snapshot_path):
    #         raise FileNotFoundError(f'Snapshot not found in the current simulation directory: {snapshot_path}')

    #     try:
    #         ds = yt.load(snapshot_path)
    #         return ds
    #     except Exception as e:
    #         raise RuntimeError(f'Error loading snapshot data: {str(e)}')

    def get_field_data(
            self,
            snap_id: Union[int, str],
            field: tuple[str, str]
        ) -> np.ndarray:
        """
        Get the data of a field in a snapshot.

        :param snap_id: int or str, the snapshot ID to analyze.
        :param field: tuple of str, field to analyze, e.g., ('gas', 'density').
        :return: np.ndarray, the field data. """
        ds = self.__load_snapshot_data__(snap_id)
        ad = ds.all_data()

        try:
            return ad[field].value
        except Exception as e:
            raise RuntimeError(f"Error loading snapshot data: {str(e)}")

    def get_field_average(
            self,
            snap_id: Union[int, str],
            field: tuple[str, str],
            weight: Union[tuple[str, str], None] = None
        ) -> float:
        """
        Get the average value of a field in a snapshot.

        :param snap_id: int or str, the snapshot ID to analyze.
        :param field: tuple of str, field to analyze, e.g., ('gas', 'density').
        :param weight: tuple of str, optional, weight field. Default is None.
        :return: float, the average value of the field. """
        ds = self.__load_snapshot_data__(snap_id)
        ad = ds.all_data()

        if weight is None or weight == "None":
            weight = ("index", "ones")

        return ad.quantities.weighted_average_quantity(field, weight)

    def get_snapshot_magnetic_energy(
            self,
            snap_id: Union[int, str]
        ) -> float:
        """
        Get the magnetic energy at each cell from their magnetic energy density.

        Args:
            snap_id (int or str): The snapshot ID to analyze.

        Returns:
            float: The magnetic energy of the snapshot."""
        def _magnetic_energy_test1(field, data):
            return (data["gas", "magnetic_energy_density"])  # * data["gas", "cell_volume"])

        def _magnetic_energy_test2(field, data):
            mu_0 = 4 * np.pi * 1e-7  # Vacuum permeability
            return 0.5 * (data["gas", "magnetic_field_strength"]**2 / mu_0)

        ds = self.__load_snapshot_data__(snap_id)

        ds.add_field(
            ("gas", "magnetic_energy_test1"),
            units="dyne/cm**2",  # "*cm",
            function=_magnetic_energy_test1,
            sampling_type="local"
        )

        ds.add_field(
            ("gas", "magnetic_energy_test2"),
            units="G**2",  # "dyne*cm",
            function=_magnetic_energy_test2,
            sampling_type="local"
        )

        magnetic_energy_test1 = ds.all_data()[("gas", "magnetic_energy_test1")]
        magnetic_energy_test2 = ds.all_data()[("gas", "magnetic_energy_test2")]

        return magnetic_energy_test1, magnetic_energy_test2

    def get_snapshot_turbulent_energy(
            self,
            snap_id: Union[int, str]
        ) -> float:
        """
        Get the turbulent energy at each cell from their kinetic energy density.

        Args:
            snap_id (int or str): The snapshot ID to analyze.
        
        Returns:
            float: The turbulent energy of the snapshot."""
        def _turbulent_energy(field, data):
            return (data["gas", "kinetic_energy_density"] * data["gas", "cell_volume"])

        ds = self.__load_snapshot_data__(snap_id)
        ad = ds.all_data()

        # Get snapshot's average kinetic energy density
        kinetic_energy_density = self.get_snapshot_field_average(snap_id, ("gas", "kinetic_energy_density"))

        # Calculate the turbulent energy from the average kinetic energy density
        turbulent_energy = kinetic_energy_density * ds.domain_width[0] ** 3

        return float(turbulent_energy)

    def get_snapshot_turbulent_to_magnetic_energy_ratio(
            self,
            snap_id: Union[int, str]
        ) -> float:
        """
        Get the ratio of turbulent to magnetic energy in a snapshot.

        Args:
            snap_id (int or str): The snapshot ID to analyze.

        Returns:
            float: The ratio of turbulent to magnetic energy."""
        def _turbulent_energy(field, data):
            return 0.5 * data["gas", "density"] * data["gas", "velocity_magnitude"] ** 2

        def _magnetic_energy(field, data):
            mu_0 = 4 * np.pi * 1e-7  # Vacuum permeability
            return 0.5 * (data["gas", "magnetic_field_strength"]**2 / mu_0)

        ds = self.__load_snapshot_data__(snap_id)
        ds.add_field(
            ("gas", "turbulent_energy"),
            units="dyne*cm",
            function=_turbulent_energy,
            sampling_type="local"
        )
        ds.add_field(
            ("gas", "magnetic_energy"),
            units="dyne*cm",
            function=_magnetic_energy,
            sampling_type="local"
        )

        # Turbulent energy (kinetic energy)
        turbulent_energy = ds.all_data()[("gas", "turbulent_energy")]

        # Magnetic energy
        magnetic_energy = ds.all_data()[("gas", "magnetic_energy")]

        # Get snapshot averages
        turbulent_energy = self.get_snapshot_field_average(snap_id, ("gas", "turbulent_energy"))
        magnetic_energy = self.get_snapshot_field_average(snap_id, ("gas", "magnetic_energy"))            

        # Turbulent to magnetic energy ratio
        return float(turbulent_energy) / float(magnetic_energy)

    def get_snapshot_timescales(
            self,
            snap_id: Union[int, str]
        ) -> dict[str, float]:
        """
        Get various time-related information of a snapshot in the simulation.

        Args:
            snap_id (int or str): The snapshot ID or identifier.

        Returns:
            dict[str, float]: A dictionary containing the current physical time, crossing
                              time, and eddy turnover time of the specified snapshot."""
        ds = self.__load_snapshot_data__(snap_id)

        # Get current physical time
        current_time = float(ds.current_time)

        # Calculate crossing time
        domain_width = ds.domain_width[0]
        domain_dimensions = ds.domain_dimensions[0]
        crossing_time = float(domain_width / domain_dimensions / current_time)

        # Calculate eddy turnover time
        mach_number = self.get_snapshot_field_average(snap_id, ("gas", "mach_number"))
        sound_speed = self.get_snapshot_field_average(snap_id, ("gas", "sound_speed"))
        eddy_turnover_time = domain_width / (2 * mach_number * sound_speed)

        # Calculate sound crossing time
        sound_crossing_time = domain_width / sound_speed

        # Calculate turbulence crossing time (turbulence formation characteristic time)
        def _velocity_rms(field, data):
            return np.sqrt(
                data[("gas", "velocity_x")] ** 2 +
                data[("gas", "velocity_y")] ** 2 +
                data[("gas", "velocity_z")] ** 2
            )
        ds.add_field(("gas", "velocity_rms"), function=_velocity_rms, sampling_type="local",  units="cm/s")
        velocity_rms = self.get_snapshot_field_average(snap_id, ("gas", "velocity_rms"))
        turbulence_crossing_time = domain_width / velocity_rms

        # # Calculate Alfven crossing time
        # magnetic_field_strength = self.get_snapshot_field_average(snap_id, ("gas", "magnetic_field_strength"))
        # alfven_crossing_time = domain_width / magnetic_field_strength

        return {
            "current_time": float(current_time),
            "crossing_time": float(crossing_time),
            "eddy_turnover_time": float(eddy_turnover_time),
            "sound_crossing_time": float(sound_crossing_time),
            "turbulence_crossing_time": float(turbulence_crossing_time),
            # "alfven_crossing_time": float(alfven_crossing_time)
        }

