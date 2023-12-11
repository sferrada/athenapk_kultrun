import os
import re
import yt
import numpy as np
import scipy as sp
import pandas as pd
from collections import deque
from src.commons import read_athenapk_input_file
yt.funcs.mylog.setLevel("ERROR")

class SimAthenaPK:
    T_LIMIT_PATTERN = r'tlim=(\S+)'
    N_LIMIT_PATTERN = r'nlim=(\S+)'
    WALLTIME_PATTERN = r'walltime used = (\S+)'
    CYCLES_PER_WALLSECOND_PATTERN = r'zone-cycles/wallsecond = (\S+)'

    def __init__(self, folder_path):
        """
        Initialize an instance of an AthenaPK run with the folder path containing simulation data.

        Args:
            folder_path (str): The path to the folder containing AthenaPK simulation data.
        """
        self.number_of_cells = None
        self.correlation_time = None
        self.solenoidal_weight = None
        self.acceleration_field_rms = None
        self.initial_magnetic_field = None
        self.code_time_between_dumps = None

        self.folder_path = folder_path
        self.input_attrs = self.read_input_attrs()
        self.snapshot_list = self.get_snapshot_list()

        self.walltime = None
        self.time_limit = None
        self.cycle_limit = None
        self.cycles_per_wallsecond = None
        self.extract_log_file_info()

    def read_input_attrs(self):
        """
        Read and return the input attributes from the input file if it exists.

        Returns:
            dict or None: The input attributes as a dictionary, or None if the input file doesn't exist.
        """
        input_file_name = next((f for f in os.listdir(self.folder_path) if f.endswith('.in')), None)
        if input_file_name:
            input_file_path = os.path.join(self.folder_path, input_file_name)
            input_file_dict = read_athenapk_input_file(input_file_path)

            if "parthenon/mesh" in input_file_dict:
                self.number_of_cells = int(input_file_dict["parthenon/mesh"][1][1])
            if "parthenon/output2" in input_file_dict:
                self.code_time_between_dumps = float(input_file_dict["parthenon/output2"][2][1])
            if "problem/turbulence" in input_file_dict:
                turbulence_params = input_file_dict["problem/turbulence"]
                self.solenoidal_weight = float(turbulence_params[8][1])
                self.acceleration_field_rms = float(turbulence_params[9][1])
                self.initial_magnetic_field = float(turbulence_params[2][1])
                self.correlation_time = float(turbulence_params[6][1])

            return input_file_dict
        return None

    def extract_log_file_info(self):
        """
        Extract information from the log file in the simulation folder.

        This method extracts walltime, time limit, cycle limit, and cycles per wallsecond from the log file.
        """
        log_file_name = next((f for f in os.listdir(self.folder_path) if f.endswith('.out')), None)
        if log_file_name:
            log_file_path = os.path.join(self.folder_path, log_file_name)
            last_lines = deque(maxlen=5)

            with open(log_file_path, 'r') as file:
                for line in file:
                    last_lines.append(line)
                    if 'tlim=' in line:
                        self.time_limit = float(re.search(self.T_LIMIT_PATTERN, line).group(1))
                    if 'nlim=' in line:
                        self.cycle_limit = int(re.search(self.N_LIMIT_PATTERN, line).group(1))
                    if 'walltime used =' in line:
                        self.walltime = float(re.search(self.WALLTIME_PATTERN, line).group(1))
                    if 'zone-cycles/wallsecond =' in line:
                        self.cycles_per_wallsecond = float(re.search(self.CYCLES_PER_WALLSECOND_PATTERN, line).group(1))

    def get_snapshot_list(self) -> None:
        """
        Get a list of snapshot files in the simulation folder.
        """
        snapshot_list = [f for f in os.listdir(self.folder_path) if f.endswith('.phdf')]
        snapshot_list.sort()
        return snapshot_list

    def __get_snapshot_file_path__(self,
                                   n_snap: int | str) -> str:
        """
        Get the file path for a snapshot based on its number or string representation.

        Args:
            n_snap (int or str): The snapshot number or its string representation.

        Returns:
            str: The file path to the snapshot.
        """
        snapshot_number_str = str(n_snap).zfill(5)

        if isinstance(n_snap, str):
            if not n_snap.endswith('.phdf'):
                snapshot_number_str += '.phdf'

        if isinstance(n_snap, int):
            snapshot_number_str += '.phdf'

        # Return the full path to the snapshot file.
        return os.path.join(self.folder_path, f'parthenon.prim.{snapshot_number_str}')

    def __load_snapshot_data__(self,
                               n_snap: int | str) -> yt.data_objects.static_output.Dataset:
        """
        Load and return the data from a snapshot file.

        Args:
            n_snap (int or str): The snapshot number or its string representation.

        Returns:
            yt.data_objects.static_output.Dataset: A dataset containing the snapshot data.
        """
        snapshot_file_path = self.__get_snapshot_file_path__(n_snap)
        if not os.path.exists(snapshot_file_path):
            raise FileNotFoundError(f'Snapshot not found in the current simulation directory: {snapshot_file_path}')

        try:
            ds = yt.load(snapshot_file_path)
            return ds
        except Exception as e:
            raise RuntimeError(f'Error loading snapshot data: {str(e)}')

    def __load_all_snapshot_data__(self) -> dict[str, yt.data_objects.static_output.Dataset]:
        """
        Load and return the data from all snapshot files.

        Returns:
            dict[str, yt.data_objects.static_output.Dataset]: A dictionary containing all the loaded datasets.
        """
        ds_dict = {}
        for snapshot_index, snapshot_file_name in enumerate(self.snapshot_list):
            ds_dict[snapshot_index] = self.__load_snapshot_data__(snapshot_file_name)
        
        return ds_dict

    def get_snapshot_field_info(self) -> None:
        """
        Get information about available fields in a snapshot.

        Args:
            n_snap (int or str): The snapshot number to analyze.
        """
        # Search for the first available snapshot file in the simulation folder.
        snapshot_file_name = next((f for f in os.listdir(self.folder_path) if f.endswith('.phdf')), None)
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

    def get_snapshot_field_data(self, n_snap: int | str, field: tuple[str, str]) -> np.ndarray:
        """
        Get the data of a field in a snapshot.

        Args:
            n_snap (int or str): The snapshot number to analyze.
            field (tuple of str): A tuple specifying the field to analyze (e.g., ('gas', 'density')).
        
        Returns:
            np.ndarray: A NumPy array containing the field data.
        """
        ds = self.__load_snapshot_data__(n_snap)
        ad = ds.all_data()

        try :
            return ad[field].value
        except Exception as e:
            raise RuntimeError(f'Error loading snapshot data: {str(e)}')

    def get_snapshot_field_average(self,
                                   n_snap: int | str,
                                   field: tuple[str, str],
                                   weight: tuple[str, str] | None = None) -> float:
        """
        Get the average value of a field in a snapshot.

        Args:
            n_snap (int or str): The snapshot number to analyze.
            field (tuple of str): A tuple specifying the field to analyze (e.g., ('gas', 'density')).
            weight (tuple of str, optional): A tuple specifying the weight field to use for averaging (e.g., ('index', 'volume')).
        
        Returns:
            averaged_quantity (float): The average value of the field.
        """
        ds = self.__load_snapshot_data__(n_snap)
        ad = ds.all_data()

        print(f"\n ******* {weight} *******")

        if weight is None or weight == "None":
            weight = ('index', 'ones')

        print(f"\n ******* {weight} *******")

        return ad.quantities.weighted_average_quantity(field, weight)

    def get_snapshot_timescales(self,
                                n_snap: int | str) -> dict[str, float]:
        """
        Get various time-related information of a snapshot in the simulation.

        Args:
            n_snap (int or str): The snapshot number or identifier.

        Returns:
            dict[str, float]: A dictionary containing the current physical time, crossing time, and eddy turnover time of the specified snapshot.
        """
        ds = self.__load_snapshot_data__(n_snap)

        # Get current physical time
        current_time = float(ds.current_time)

        # Calculate crossing time
        domain_width = ds.domain_width[0]
        domain_dimensions = ds.domain_dimensions[0]
        crossing_time = float(domain_width / domain_dimensions / current_time)

        # Calculate eddy turnover time
        mach_number = self.get_snapshot_field_average(n_snap, ("gas", "mach_number"))
        sound_speed = self.get_snapshot_field_average(n_snap, ("gas", "sound_speed"))
        eddy_turnover_time = domain_width / (2 * mach_number * sound_speed)

        return {
            "current_time": float(current_time),
            "crossing_time": float(crossing_time),
            "eddy_turnover_time": float(eddy_turnover_time)
        }

    def plot_snapshot_field_map(self,
                                n_snap: int | str,
                                field: tuple[str, str],
                                normal: str = "z",
                                method: str = "slice",
                                color_map: str = "viridis",
                                overplot_velocity: bool = False,
                                **kwargs: dict) -> None:
        """
        This function is a convenient wrapper for creating and customizing slice or projection plots of simulation data using yt.
        Depending on the specified method, it creates either a slice plot or a projection plot of the given field along the chosen axis.
        The resulting plot object is returned.

        Args:
            n_snap (int or str): The snapshot number to plot.
            field (tuple of str): A tuple specifying the field to plot (e.g., ('gas', 'density')).
            normal (str, optional): The axis for slicing or project (e.g., 'z'). Defaults to 'z'.
            method (str, optional): The plotting method ('slice' or 'projection'). Defaults to 'slice'.
            color_map (str, optional): The colormap to use for visualization. Defaults to 'viridis'.
            overplot_velocity (bool, optional): If True, overplot the velocity field. Defaults to False.
            **kwargs (dict, optional): Additional keyword arguments to pass to the yt plot.

        Returns:
            yt.SlicePlot or yt.ProjectionPlot: The yt plot object representing the field slice or projection.
        """
        ds = self.__load_snapshot_data__(n_snap)

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

    # def plot_snapshot_power_spectra(self, n_snap: int | str) -> None:
    #     """
    #     Plot the power spectra of the velocity and magnetic fields of a snapshot.
    # 
    #     Args:
    #         n_snap (int or str): The snapshot number to plot.
    #     """
    #     ds = self.__load_snapshot_data__(n_snap)
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
        - The second snapshot's index (offset by the first snapshot).
        """
        ds_arr = yt.load(self.folder_path + '/parthenon.prim.*.phdf')

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

                # Print current status of the foor loops
                print(f"i = {i}, j = {j}, correlation_time = {correlation_time[0, i, j-i]}")

        return correlation_time

    def get_run_average_fields(self,
                               fields: list[str],
                               weight: tuple[str, str] | None = None,
                               verbose: bool = False,
                               in_time: bool = False,
                               save_data: bool = False) -> None:
        """
        Calculate and save the density-weighted average values for specified fields.

        Args:
            fields (list[str]): The names of the fields for which to calculate averages.
            weight (tuple[str, str] | None, optional): The weight field to use for averaging. Default is None.
            verbose (bool, optional): If True, print information about the field being processed.
            in_time (bool, optional): If True, get the snapshot's current time and save it along with the averages.
            save_data (bool, optional): If True, save the calculated values to a file.
                                        If False, return the calculated values.
                                        Default is True.

        Returns:
            None or ndarray: If save_data is True, the function saves the calculated values to a file and does not return a value.
                             If save_data is False, the function returns the calculated values as a NumPy array.
        """
        snapshots_data = []

        for sim in self.snapshot_list:
            i_snapshot = sim.split('.')[2]
            snapshot_data = []

            if in_time:
                times_dict = self.get_snapshot_timescales(i_snapshot)
                snapshot_data.append(times_dict["current_time"])

            for field in fields:
                if verbose:
                    print(f"Running analysis for field: {field}")
                average_field = self.get_snapshot_field_average(i_snapshot, ('gas', field), weight)
                snapshot_data.append(float(average_field))

            snapshots_data.append(snapshot_data)

        header = ["time"] + fields if in_time else fields
        df = pd.DataFrame(snapshots_data, columns=header)

        if save_data:
            fout = "average_values.tsv"
            df.to_csv(os.path.join(self.folder_path, fout), index=False, sep="\t", float_format="%.10E")
        else:
            return df

    def get_run_statistics(self) -> None:
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
        # target_rms_acceleration = self.acceleration_field_rms
        # target_solenoidal_weight = self.solenoidal_weight

        # Calculate the forcing correlation time
        # find the first zero crossing - we only integrate till that point as it's noise afterwards anyway
        # this also ensures that the t_corr from later snapshots is not included as too few snapshots would
        # follow to actually integrate for a full t_corr
        correlation_time = self.get_run_integral_times()
        t_corr_values = np.zeros((4, correlation_time.shape[1]))
        for i in range(correlation_time.shape[1]):
            for j in range(correlation_time.shape[2]):
                if correlation_time[0, i, j] < 0:
                    t_corr_values[0, i] = j
                    break
                if j == correlation_time.shape[2] - 1:
                    t_corr_values[0, i] = correlation_time.shape[2]

        # calculate the integral
        integral = np.zeros((4, correlation_time.shape[1]))
        std = np.zeros((4, correlation_time.shape[1]))
        for i in range(correlation_time.shape[1]):
            for j in range(int(t_corr_values[0, i])):
                integral[0, i] += correlation_time[0, i, j]
                integral[1, i] += correlation_time[1, i, j]
                integral[2, i] += correlation_time[2, i, j]
                integral[3, i] += correlation_time[3, i, j]
                std[0, i] += (correlation_time[0, i, j] - t_corr_values[0, i]) ** 2
                std[1, i] += (correlation_time[1, i, j] - t_corr_values[1, i]) ** 2
                std[2, i] += (correlation_time[2, i, j] - t_corr_values[2, i]) ** 2
                std[3, i] += (correlation_time[3, i, j] - t_corr_values[3, i]) ** 2
        integral *= self.code_time_between_dumps

        # calculate the correlation time
        t_corr_values = integral / correlation_time.shape[2]

        # calculate the standard deviation
        std = np.sqrt(std / correlation_time.shape[2])

        # return the target, actual and standard deviation values
        return target_correlation_time, np.mean(t_corr_values[0, :]), np.mean(std[0, :])

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
            ds = yt.load(self.folder_path + '/parthenon.prim.*.phdf')
            ad = ds.all_data()
            field1 = ad.quantities.weighted_average_quantity(field1, ('index', 'volume'))
            field2 = ad.quantities.weighted_average_quantity(field2, ('index', 'volume'))
            return field1 / field2

        id_split = self.folder_path.split('/')[-1].split('-')
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

