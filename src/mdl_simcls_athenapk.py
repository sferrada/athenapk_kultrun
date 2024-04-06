import os
import re
import yt
import numpy as np
import scipy as sp
import pandas as pd
from typing import Union
from collections import deque
from src.mdl_files import read_athenapk_input_file
yt.funcs.mylog.setLevel("ERROR")

class Snapshot:
    def __init__(self, index, outdir):
        """
            Initialize a single snapshot instance
        """
        self.id = index  # Snapshot identifier
        self.outdir = outdir  # Path to the simulation directory

        # Get the snapshot file path
        self._path = self._get_path()

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
            return yt.load(self._path)
        except:
            raise FileNotFoundError("")

class SimAthenaPK:
    # Regular expressions for log file parsing
    T_LIMIT_REGEX = r"tlim=(\S+)"
    N_LIMIT_REGEX = r"nlim=(\S+)"
    WALLTIME_REGEX = r"walltime used = (\S+)"
    CYCLES_PER_WALLSECOND_REGEX = r"zone-cycles/wallsecond = (\S+)"

    def __init__(self, folder_path: str) -> None:
        """
        Initialize an instance of an AthenaPK run with the folder path containing simulation data.

        :param folder_path: str, the path to the folder containing AthenaPK simulation data. """
        self.outdir = folder_path  # Path to the simulation folder
        self.snapshots = {}  # Dictionary of snapshots
        self.input_attrs = {}  # Dictionary of input file attributes

        # Input file parameters
        self.cells_number = None  # Number of cells in the simulation
        self.dump_code_time = None  # Physical time between code dumps
        self.correlation_time = None  # Correlation time between acceleration fields
        self.solenoidal_weight = None  # Relative power of solenoidal modes
        self.acceleration_field_rms = None  # RMS acceleration field
        self.initial_magnetic_field = None  # Initial magnetic field strength

        # Log file regular expressions
        self.walltime = None  # Walltime used by the simulation
        self.time_limit = None  # Physical maximum time limit
        self.cycle_limit = None  # Maximum cycle limit
        self.cycles_per_walls = None  # Cycles per walltime in seconds

        self.parse_infile()
        self.parse_logfile()
        self.get_snapshots()

    def __get_snapshot_file_path__(
            self,
            snap_id: Union[int, str]
        ) -> str:
        """
        Get the file path for a snapshot based on its number or string representation.

        :param snap_id: int or str, the snapshot ID.
        :return: str, the file path to the snapshot. """
        snapshot_number_str = str(snap_id).zfill(5)

        if not snapshot_number_str.endswith('.phdf'):
            snapshot_number_str += '.phdf'

        # Return the full path to the snapshot file.
        return os.path.join(self.outdir, f'parthenon.prim.{snapshot_number_str}')

    def __load_snapshot_data__(
            self,
            snap_id: Union[int, str]
        ) -> yt.data_objects.static_output.Dataset:
        """
        Load and return the data from a snapshot file.

        :param snap_id: int or str, the snapshot ID to load.
        :return: yt.data_objects.static_output.Dataset, a yt dataset containing the snapshot data. """
        snapshot_path = self.__get_snapshot_file_path__(snap_id)
        if not os.path.exists(snapshot_path):
            raise FileNotFoundError(f'Snapshot not found in the current simulation directory: {snapshot_path}')

        try:
            ds = yt.load(snapshot_path)
            return ds
        except Exception as e:
            raise RuntimeError(f'Error loading snapshot data: {str(e)}')

    def __load_all_snapshot_data__(self) -> dict[str, yt.data_objects.static_output.Dataset]:
        """
        Load and return the data from all snapshot files.

        Returns:
            dict[str, yt.data_objects.static_output.Dataset]: A dictionary containing all the loaded datasets."""
        ds_dict = {}
        for snapshot_index, snapshot_file_name in enumerate(self.snapshot_list):
            ds_dict[snapshot_index] = self.__load_snapshot_data__(snapshot_file_name)
        
        return ds_dict

    def parse_infile(self) -> None:
        """
            Read and return the input attributes from the input file if it exists.
        """
        infile_name = next(
            (f for f in os.listdir(self.outdir) if f.endswith('.in')
        ), None)

        try:
            infile_dict = read_athenapk_input_file(
                os.path.join(self.outdir, infile_name)
            )

            # Extract the relevant parameters from the input file
            if "parthenon/mesh" in infile_dict:
                self.cells_number = int(infile_dict["parthenon/mesh"][1][1])
            if "parthenon/output2" in infile_dict:
                self.dump_code_time = float(infile_dict["parthenon/output2"][2][1])
            if "problem/turbulence" in infile_dict:
                turb_params = infile_dict["problem/turbulence"]
                self.correlation_time = float(turb_params[6][1])
                self.solenoidal_weight = float(turb_params[8][1])
                self.initial_magnetic_field = float(turb_params[2][1])
                self.acceleration_field_rms = float(turb_params[9][1])

            # Get other parameters from the input file as a dictionary
                self.input_attrs = infile_dict

        except Exception as e:
            print(f"Error reading input file: {str(e)}")

    def parse_logfile(self) -> None:
        """
            Extract walltime, time limit, cycle limit, and cycles per wallsecond from the log
        file in the simulation folder.
        """
        logfile_name = next(
            (f for f in os.listdir(self.outdir) if f.endswith('.out')
        ), None)

        try:
            logfile_path = os.path.join(self.outdir, logfile_name)
            logfile_tail = deque(maxlen=5)

            with open(logfile_path, "r") as log:
                for line in log:
                    logfile_tail.append(line)

                    # Get key parameters from the log file
                    if "tlim=" in line:
                        self.time_limit = float(re.search(self.T_LIMIT_REGEX, line).group(1))
                    if "nlim=" in line:
                        self.cycle_limit = int(re.search(self.N_LIMIT_REGEX, line).group(1))
                    if "walltime used =" in line:
                        self.walltime = float(re.search(self.WALLTIME_REGEX, line).group(1))
                    if "zone-cycles/wallsecond =" in line:
                        self.cycles_per_walls = float(re.search(self.CYCLES_PER_WALLSECOND_REGEX, line).group(1))

        except Exception as e:
            print(f"Error reading log file: {str(e)}")

    def get_snapshots(self) -> None:
        """
            Get a list of snapshot files in the simulation folder.
        """
        snapshot_list = [f for f in os.listdir(self.outdir) if f.endswith('.phdf')]
        snapshot_list.sort()
        return snapshot_list

    def get_snapshot_field_data(
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

        try :
            return ad[field].value
        except Exception as e:
            raise RuntimeError(f"Error loading snapshot data: {str(e)}")

    def get_snapshot_field_average(
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

    def get_run_available_fields(self) -> None:
        """
        Get information about available fields in a snapshot.

        Args:
            snap_id (int or str): The snapshot ID to analyze."""
        # Search for the first available snapshot file in the simulation folder.
        snapshot_file_name = next((f for f in os.listdir(self.outdir) if f.endswith('.phdf')), None)
        if not snapshot_file_name:
            raise FileNotFoundError('No snapshot file found in the current simulation directory.')

        # Load the snapshot data and print the available fields.
        snapshot_number = snapshot_file_name.split('.')[2]
        ds = self.__load_snapshot_data__(snapshot_number)
        print('>> [Field] Gas:')
        for elem in dir(ds.fields.gas):
            print(elem)
        print('\n>> [Field] Index:')
        for elem in dir(ds.fields.index):
            print(elem)
        print('\n>> [Field] Parthenon:')
        for elem in dir(ds.fields.parthenon):
            print(elem)

    def get_run_integral_times(self) -> np.ndarray:
        """
        Calculate and return the correlation time between the acceleration fields for a series of simulation snapshots.

        Returns:
            np.ndarray: A 4D NumPy array containing the correlation time values.

        This function loads a sequence of simulation snapshots from the specified folder path and calculates the correlation time between the acceleration fields
        for all pairs of snapshots. The correlation time is computed for the full 3D acceleration field and its individual components (x, y, z).
        The results are returned as a 4D NumPy array, where the dimensions represent:
        - The acceleration component (0 for full field, 1 for x, 2 for y, 3 for z).
        - The first snapshot's index.
        - The second snapshot's index (offset by the first snapshot)."""
        ds_arr = yt.load(self.outdir + '/parthenon.prim.*.phdf')

        acc_arr = []
        for ds in ds_arr:    
            cg = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
            acc_arr.append(np.array([
                cg[("parthenon", "acc_0")],
                cg[("parthenon", "acc_1")],
                cg[("parthenon", "acc_2")],
            ]))

        acc_arr = np.array(acc_arr)
        num_snaps = acc_arr.shape[0]
        correlation_time = np.zeros((4, num_snaps, num_snaps))

        # We start i from 1 because the correlation time is zero for the first snapshot.
        for i in range(1, num_snaps):
            for j in range(num_snaps):
                if j < i:
                    continue
                correlation_time[0, i, j-i] = sp.stats.pearsonr(acc_arr[i, :, :, :, :].reshape(-1), acc_arr[j, :, :, :, :].reshape(-1))[0]
                correlation_time[1, i, j-i] = sp.stats.pearsonr(acc_arr[i, 0, :, :, :].reshape(-1), acc_arr[j, 0, :, :, :].reshape(-1))[0]
                correlation_time[2, i, j-i] = sp.stats.pearsonr(acc_arr[i, 1, :, :, :].reshape(-1), acc_arr[j, 1, :, :, :].reshape(-1))[0]
                correlation_time[3, i, j-i] = sp.stats.pearsonr(acc_arr[i, 2, :, :, :].reshape(-1), acc_arr[j, 2, :, :, :].reshape(-1))[0]

        return correlation_time

    def get_run_average_fields(
            self,
            fields: list[str],
            weight: tuple[str, str] = None,
            verbose: bool = False,
            in_time: bool = False,
            save_data: bool = False
        ) -> None | pd.DataFrame:
        """
        Calculate and save the density-weighted average values for specified fields.

        :param fields: list[str], the names of the fields for which to calculate averages.
        :param weight: tuple[str, str], the weight field to use for averaging. Default is None.
        :param verbose: bool, if True, print information about the field being processed.
        :param in_time: bool, if True, get snapshot's current time and save it with the averages.
        :param save_data: bool, if True, save the calculated values to a file. Default is False.
        :return: None or ndarray, if save_data is True, the function saves the calculated values
                    to a file and does not return a value. If save_data is False, the function
                    returns the calculated values as a NumPy array. """
        run_data = []

        for snapshot in self.snapshot_list:
            snapshot_id = snapshot.split('.')[2]
            snapshot_data = []

            if in_time:
                times_dict = self.get_snapshot_timescales(snapshot_id)
                snapshot_data.append(times_dict["current_time"])

            for field in fields:
                if verbose:
                    print(f"Running analysis for field: {field}")

                # Calculate the average field value
                average_field = self.get_snapshot_field_average(snapshot_id, ("gas", field), weight)
                snapshot_data.append(float(average_field))

            run_data.append(snapshot_data)

        header = ["time"] + fields if in_time else fields
        df = pd.DataFrame(run_data, columns=header)

        if save_data:
            fout = "average_values.tsv"
            df.to_csv(os.path.join(self.outdir, fout), index=False, sep="\t", float_format="%.10E")
        else:
            return df

    def get_run_statistics(self) -> dict:
        """
        Calculate and print statistics for the simulation.

        Returns:
            dict: A dictionary containing the calculated statistics.

        The function calculates statistics related to the simulation and prints the results. It extracts specific values
        from the provided identifier and performs calculations. The computed statistics include the following:

        - Forcing correlation time: Target and actual values, along with the standard deviation.
        - RMS acceleration: Target and actual values, along with the standard deviation.
        - Relative power of solenoidal modes: Target and actual values, along with the standard deviation."""
        # Extract target values from the input file
        target_correlation_time = self.correlation_time
        target_rms_acceleration = self.acceleration_field_rms
        target_solenoidal_weight = self.solenoidal_weight

        # Find the first zero crossing - we only integrate unitl that point as it is noise afterwards anyway
        # this also ensures that the t_corr from later snapshots is not included as too few snapshots would
        # follow to actually integrate for a full forcing correlation time t_corr.
        vector_size = 4
        correlation_time = self.get_run_integral_times()
        t_corr_values = np.zeros((vector_size, correlation_time.shape[1]))
        for i in range(correlation_time.shape[1]):
            for j in range(correlation_time.shape[2]):
                if correlation_time[0, i, j] < 0:
                    t_corr_values[0, i] = j
                    break
                if j == correlation_time.shape[2] - 1:
                    t_corr_values[0, i] = correlation_time.shape[2]

        # calculate the integral
        integral = np.zeros((vector_size, correlation_time.shape[1]))
        std = np.zeros((vector_size, correlation_time.shape[1]))
        for i in range(correlation_time.shape[1]):
            for j in range(int(t_corr_values[0, i])):
                for k in range(vector_size):
                    integral[k, i] += correlation_time[k, i, j]
                    std[k, i] += correlation_time[k, i, j] ** 2
        integral *= self.code_time_between_dumps

        # Calculate the correlation time
        t_corr_values = integral / correlation_time.shape[2]

        # Calculate the standard deviation
        std = np.sqrt(std / correlation_time.shape[2])

        return {
            "correlation_time": {
                "target": target_correlation_time,
                "actual": np.mean(t_corr_values[0, :]),
                "std": np.mean(std[0, :])
            },
            "rms_acceleration": {
                "target": target_rms_acceleration,
                "actual": target_rms_acceleration,  # placeholder
                "std": 0.0  # placeholder
            },
            "solenoidal_weight": {
                "target": target_solenoidal_weight,
                "actual": target_solenoidal_weight,  # placeholder
                "std": 0.0  # placeholder
            }
        }

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

    # def plot_snapshot_power_spectra(self, snap_id: int | str) -> None:
    #     """
    #     Plot the power spectra of the velocity and magnetic fields of a snapshot.
    # 
    #     Args:
    #         snap_id (int or str): The snapshot ID to plot."""
    #     ds = self.__load_snapshot_data__(snap_id)
    #     ad = ds.all_data()
    # 
    #     # Calculate the power spectra of the velocity, kinetic and magnetic fields
    #     # using the yt built-in function.
    #     kinetic_power_spectrum = yt.create_profile(ad, "kinetic_energy", "cell_volume",
    #                                                logs={"kinetic_energy": True},
    #                                                n_bins=64, weight_field=None)
    #     velocity_power_spectrum = yt.create_profile(ad, "velocity_magnitude", "cell_volume",
    #                                                 logs={"velocity_magnitude": True},
    #                                                 n_bins=64, weight_field=None)
    #     magnetic_power_spectrum = yt.create_profile(ad, "magnetic_field_strength", "cell_volume",
    #                                                 logs={"magnetic_field_strength": True},
    #                                                 n_bins=64, weight_field=None)
    # 
    #     # Calculate forcing spectrum
    # 
    #     # Plot the power spectra.
    #     fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    #     ax[0].plot(kinetic_power_spectrum.x, kinetic_power_spectrum["kinetic_energy"])
    #     ax[0].set_xlabel(r"$k$")
    #     ax[0].set_ylabel(r"$E(k)$")
    #     ax[0].set_title("Kinetic Power Spectrum")
    #     ax[1].plot(velocity_power_spectrum.x, velocity_power_spectrum["velocity_magnitude"])
    #     ax[1].set_xlabel(r"$k$")
    #     ax[1].set_ylabel(r"$E(k)$")
    #     ax[1].set_title("Velocity Power Spectrum")
    #     ax[2].plot(magnetic_power_spectrum.x, magnetic_power_spectrum["magnetic_field_strength"])
    #     ax[2].set_xlabel(r"$k$")
    #     ax[2].set_ylabel(r"$E(k)$")
    #     ax[2].set_title("Magnetic Power Spectrum")
    #     plt.show()

def get_run_statistics_old(self) -> None:
        """
        Calculate and print statistics for the simulation.

        Returns:
            None

        The function calculates statistics related to the simulation and prints the results. It extracts specific values
        from the provided identifier and performs calculations. The printed statistics include the following:

        - Forcing correlation time: Both target and actual values are printed, along with the standard deviation.
        - RMS acceleration: Both target and actual values are printed, along with the standard deviation.
        - Relative power of solenoidal modes: Both target and actual values are printed, along with the standard deviation.
        """
        # Extract target values from the input file
        target_correlation_time = self.correlation_time
        target_rms_acceleration = self.acceleration_field_rms
        target_solenoidal_weight = self.solenoidal_weight

        # Calculate the forcing correlation time, method 1
        correlation_time = self.get_run_integral_times()
        correlation_time = correlation_time[0, :, :]
        correlation_time = correlation_time[correlation_time != 0]
        correlation_time_mean = np.mean(correlation_time)
        correlation_time_std = np.std(correlation_time)

        # Calculate the forcing correlation time, method 2
        corr_time_target = self.correlation_time
        corr_time_actuals = []
        all_data = self.get_run_integral_times()
        for this_data in all_data:
            num_points = this_data.shape[1]
            for i in range(num_points):
                this_slice = this_data[i, :]
                idx_0 = np.argwhere(np.array(this_slice) < 0)
                if len(idx_0) == 0:
                    continue
                corr_time_actuals.append(np.trapz(this_slice[:idx_0[0][0]], dx=self.code_time_between_dumps))

        # Calculate the RMS acceleration
        rms_acceleration = self.get_run_average_fields(['acc_0', 'acc_1', 'acc_2'])
        rms_acceleration = rms_acceleration.to_numpy()
        rms_acceleration = rms_acceleration[:, 1:]
        rms_acceleration = rms_acceleration[rms_acceleration != 0]
        rms_acceleration_mean = np.mean(rms_acceleration)
        rms_acceleration_std = np.std(rms_acceleration)

        # Calculate the relative power of solenoidal modes, method 1
        solenoidal_weight = self.get_run_average_fields(['solenoidal_weight'])
        solenoidal_weight = solenoidal_weight.to_numpy()
        solenoidal_weight = solenoidal_weight[:, 1:]
        solenoidal_weight = solenoidal_weight[solenoidal_weight != 0]
        solenoidal_weight_mean = np.mean(solenoidal_weight)
        solenoidal_weight_std = np.std(solenoidal_weight)

        # Calculate the relative power of solenoidal modes, method 2
        def get_mean_squared_ratio(field1, field2):
            ds = yt.load(self.outdir + '/parthenon.prim.*.phdf')
            ad = ds.all_data()
            field1 = ad.quantities.weighted_average_quantity(field1, ('index', 'volume'))
            field2 = ad.quantities.weighted_average_quantity(field2, ('index', 'volume'))
            return field1 / field2

        id_split = self.outdir.split('/')[-1].split('-')
        ζ = float(id_split[2].split('_')[1])
        sol_weight_actual = get_mean_squared_ratio('a_s_mag' + '/moments/' + 'rms', 'a' + '/moments/' + 'rms')
        sol_weight = 1.0 - ((1 - ζ) ** 2 / (1 - 2 * ζ + 3 * ζ ** 2))

        # Print the statistics
        print(f"> Forcing correlation time:")
        print(f"    Method 1: {correlation_time_mean:.2f} +/- {correlation_time_std:.2f} (target: {target_correlation_time:.2f})")
        print(f"    Method 2: {np.mean(corr_time_actuals):.2f} +/- {np.std(corr_time_actuals):.2f} (target: {target_correlation_time:.2f})")
        print(f"> RMS acceleration: {rms_acceleration_mean:.2f} +/- {rms_acceleration_std:.2f} (target: {target_rms_acceleration:.2f})")
        print(f"> Relative power of solenoidal modes:")
        print(f"    Method 1: {solenoidal_weight_mean:.2f} +/- {solenoidal_weight_std:.2f} (target: {target_solenoidal_weight:.2f})")
        print(f"    Method 2: {sol_weight_actual[0]:.2f} +/- {sol_weight_actual[1]:.2f} (target: {sol_weight:.2f})")

