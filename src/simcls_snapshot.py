import os
import yt
import numpy as np
from src.mdl_physics import add_field

class Snapshot:
    def __init__(self, index, outdir):
        """Initialize a single snapshot instance."""
        self.id = index  # Snapshot identifier
        self.outdir = outdir  # Path to the simulation directory

        # Get the snapshot file path
        self._path = self._get_path()

        # Instantiate snapshot data
        self.data = None
        self.fields = None
        self.timescales = None

    def _get_path(self):
        """Return the path to the snapshot file."""
        num_id = str(self.id).zfill(5)
        num_id += ".phdf" if not num_id.endswith(".phdf") else ""

        return os.path.join(self.outdir, f"parthenon.prim.{num_id}")

    def _define_fields(self):
        """Define extra physical fields."""
        self.fields = {}

        for field in [
            ("gas", "magnetic_energy_test1"),
            ("gas", "magnetic_energy_test2"),
            ("gas", "turbulent_energy"),
            ("gas", "magnetic_energy"),
            ("gas", "velocity_rms"),
            ("gas", "mean_squared_ratio")
        ]:
            self.fields[field[1]] = add_field(self.data, field)

    def load(self):
        """Load and return snapshot data"""
        try:
            self.data = yt.load(self._path)

            # # Define extra physical fields
            # self._define_fields()

        except FileNotFoundError:
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

    def get_field_data(self, field: tuple[str, str]) -> np.ndarray:
        """
        Get the data of a field in a snapshot.

        :param field: tuple of str, field to analyze, e.g., ('gas', 'density').
        :return: np.ndarray, the field data. """
        try:
            ds = self.data
            ad = ds.all_data()

            return ad[field].value
        except Exception as e:
            raise RuntimeError(f"Error loading snapshot data: {str(e)}")

    def get_field_average(self, field: tuple[str, str], weight: tuple[str, str] = None) -> float:
        """
        Get the average value of a field in a snapshot.

        :param field: tuple of str, field to analyze, e.g., ('gas', 'density').
        :param weight: tuple of str, optional, weight field. Default is None.
        :return: float, the average value of the field. """
        try:
            ds = self.data
            ad = ds.all_data()

            if weight is None:
                weight = ("index", "ones")

            return ad.quantities.weighted_average_quantity(field, weight)
        except Exception as e:
            raise RuntimeError(f"Error loading snapshot data: {str(e)}")

    def get_snapshot_timescales(self) -> None:
        """Get various time-related information of a snapshot in the simulation."""
        ds = self.data

        # Get current physical time
        current_time = float(ds.current_time)

        # Calculate crossing time
        domain_width = ds.domain_width[0]
        domain_dimensions = ds.domain_dimensions[0]
        crossing_time = float(domain_width / domain_dimensions / current_time)

        # Calculate eddy turnover time
        mach_number = self.get_field_average(("gas", "mach_number"))
        sound_speed = self.get_field_average(("gas", "sound_speed"))
        eddy_turnover_time = domain_width / (2 * mach_number * sound_speed)

        # Calculate sound crossing time
        sound_crossing_time = domain_width / sound_speed

        # Calculate turbulence crossing time (turbulence formation characteristic time)
        velocity_rms = self.get_field_average(("gas", "velocity_rms"))
        turbulence_crossing_time = domain_width / velocity_rms

        # # Calculate Alfven crossing time
        # magnetic_field_strength = self.get_field_average(("gas", "magnetic_field_strength"))
        # alfven_crossing_time = domain_width / magnetic_field_strength

        self.timescales = {
            "current_time": float(current_time),
            "crossing_time": float(crossing_time),
            "eddy_turnover_time": float(eddy_turnover_time),
            "sound_crossing_time": float(sound_crossing_time),
            "turbulence_crossing_time": float(turbulence_crossing_time),
            # "alfven_crossing_time": float(alfven_crossing_time)
        }
