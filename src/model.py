import os
import re
import yt
import numpy as np
import scipy as scipy
from collections import deque
from src.commons import read_athenapk_input_file
yt.funcs.mylog.setLevel("ERROR")

class LoadAthenaPKRun:
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
        self.equation_of_state = None
        self.solenoidal_weight = None
        self.acceleration_field_rms = None
        self.initial_magnetic_field = None
        self.correlation_time = None
        self.code_time_between_dumps = None

        # Instantiation of the class
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
            if "hydro" in input_file_dict:
                self.equation_of_state = str(input_file_dict["hydro"][1][1]).capitalize()
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

    def _get_snapshot_file_path(self,
                                n_snap: int | str) -> str:
        """
        Get the file path for a snapshot based on its number or string representation.

        Args:
            n_snap (int or str): The snapshot number or its string representation.

        Returns:
            str: The file path to the snapshot.
        """
        snapshot_number_str = str(n_snap).zfill(5)
        return os.path.join(self.folder_path, f'parthenon.prim.{snapshot_number_str}.phdf')

    def _load_snapshot_data(self,
                            n_snap: int | str) -> yt.data_objects.static_output.Dataset:
        """
        Load and return the data from a snapshot file.

        Args:
            n_snap (int or str): The snapshot number or its string representation.

        Returns:
            yt.data_objects.static_output.Dataset: A dataset containing the snapshot data.
        """
        snapshot_file_path = self._get_snapshot_file_path(n_snap)
        if not os.path.exists(snapshot_file_path):
            raise FileNotFoundError(f'Snapshot not found in the current simulation directory: {snapshot_file_path}')

        try:
            ds = yt.load(snapshot_file_path)
            return ds
        except Exception as e:
            raise RuntimeError(f'Error loading snapshot data: {str(e)}')

    def get_snapshot_field_info(self,
                                n_snap: int | str) -> None:
        """
        Get information about available fields in a snapshot.

        Args:
            n_snap (int or str): The snapshot number to analyze.
        """
        ds = self._load_snapshot_data(n_snap)
        print('>> [Field] Gas:')
        for elem in dir(ds.fields.gas):
            print(elem)
        print('\n>> [Field] Index:')
        for elem in dir(ds.fields.index):
            print(elem)
        print('\n>> [Field] Parthenon:')
        for elem in dir(ds.fields.parthenon):
            print(elem)

    def get_snapshot_current_time(self,
                                  n_snap: int | str) -> float:
        """
        Get the current time of a snapshot in the simulation.

        Args:
            n_snap (int or str): The snapshot number or identifier.

        Returns:
            float: The current time of the specified snapshot.
        """
        ds = self._load_snapshot_data(n_snap)
        return float(ds.current_time)

    def get_snapshot_field_average(self,
                                   n_snap: int | str,
                                   field: tuple[str, str],
                                   weight: tuple[str, str] | None = None) -> float:
        """
        Get the average value of a field in a snapshot.

        Args:
            n_snap (int or str): The snapshot number to analyze.
            field (tuple of str): A tuple specifying the field to analyze (e.g., ('gas', 'density')).
        
        Returns:
            averaged_quantity (float): The average value of the field.
        """
        ds = self._load_snapshot_data(n_snap)
        ad = ds.all_data()

        if weight is None:
            weight = ('index', 'ones')

        return ad.quantities.weighted_average_quantity(field, weight)

    def plot_snapshot_field(self,
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
        ds = self._load_snapshot_data(n_snap)

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
                ("gas", "velocity_%s" % velocity_coords[0]),
                ("gas", "velocity_%s" % velocity_coords[1]),
                color='green',
                factor=16
            )

        return _plot

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
        for i in range(num_snaps):
            for j in range(num_snaps):
                if j < i:
                    continue
                correlation_time[0, i, j-i] = scipy.stats.pearsonr(acc_arr[i, :, :, :, :].reshape(-1), acc_arr[j, :, :, :, :].reshape(-1))[0]
                correlation_time[1, i, j-i] = scipy.stats.pearsonr(acc_arr[i, 0, :, :, :].reshape(-1), acc_arr[j, 0, :, :, :].reshape(-1))[0]
                correlation_time[2, i, j-i] = scipy.stats.pearsonr(acc_arr[i, 1, :, :, :].reshape(-1), acc_arr[j, 1, :, :, :].reshape(-1))[0]
                correlation_time[3, i, j-i] = scipy.stats.pearsonr(acc_arr[i, 2, :, :, :].reshape(-1), acc_arr[j, 2, :, :, :].reshape(-1))[0]

        return correlation_time

    def get_run_average_fields(self,
                               field: str,
                               weight: tuple[str, str] | None = None,
                               in_time: bool = True) -> None:
        """
        Calculate and save the density-weighted averages values for a specified field.

        Args:
            field (str): The name of the field for which to calculate averages.
            weight (tuple[str, str] | None, optional): The weight field to use for averaging. Default is None.
            in_time (bool, optional): If True, get the snapshot's current time and save it along with the averages.

        Returns:
            None: The function saves the calculated values to a file and does not return a value.
        """
        times = []
        fields = []

        for sim in self.snapshot_list:
            i_snapshot = sim.split('.')[2]
            average_field = self.get_snapshot_field_average(i_snapshot, ('gas', field), weight)
            fields.append(float(average_field))

            if in_time:
                current_time = self.get_snapshot_current_time(i_snapshot)
                times.append(current_time)

        data = np.column_stack((times, fields)) if in_time else fields
        header = f'current_time {field}' if in_time else field
        out_name = f'average_{field}_in_time.txt' if in_time else f'average_{field}.txt'

        np.savetxt(os.path.join(self.folder_path, out_name), data, header=header)

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
        corr_time_target = self.correlation_time
        corr_time_actuals = []
        code_time_between_dumps = self.input_attrs["parthenon/output2"][2][1]
        print(code_time_between_dumps)

        # Note : `this_data` should be the vector for a single snapshot, not for the whole run
        all_data = self.get_run_integral_times()
        for this_data in all_data:
            num_points = this_data.shape[1]
            for i in range(num_points):
                this_slice = this_data[i, :]
                idx_0 = np.argwhere(np.array(this_slice) < 0)
                if len(idx_0) == 0:
                    continue
                corr_time_actuals.append(np.trapz(this_slice[:idx_0[0][0]], dx=code_time_between_dumps))

        print(corr_time_actuals)

        # corr_time_actual = (np.mean(corr_time_actuals), np.std(corr_time_actuals))

        # # a_rms = float(id_split[1].split('_')[1])
        # # a_rms_actual = get_mean('a' + '/moments/' + 'rms')

        # # ζ = float(id_split[2].split('_')[1])
        # # sol_weight_actual = get_mean_squared_ratio('a_s_mag' + '/moments/' + 'rms', 'a' + '/moments/' + 'rms')
        # # sol_weight = 1.0 - ((1 - ζ) ** 2 / (1 - 2 * ζ + 3 * ζ ** 2))

        # msg = (
        #     f"{'Forcing correlation time':30} target: {corr_time_target:.2f} actual: {corr_time_actual[0]:.2f}+/-{corr_time_actual[1]:.3f}\n"
        #     # f"{'RMS acceleration':30} target: {a_rms:.2f} actual: {a_rms_actual[0]:.2f}+/-{a_rms_actual[1]:.3f}\n"
        #     # f"{'Rel power of sol. modes':30} target: {sol_weight:.2f}"
        #     # f" actual: {sol_weight_actual[0]:.2f}+/-{sol_weight_actual[1]:.3f}"
        # )

        # print(msg)

