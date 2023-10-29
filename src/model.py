import os
import re
import yt
import numpy as np
import scipy as scipy
from collections import deque
from src.commons import read_athenapk_config_file
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
        self.folder_path = folder_path
        self.input_attrs = self.read_input_attrs()
        self.snapshot_list = self.get_snapshot_list()

        self.walltime = None
        self.time_limit = None
        self.cycle_limit = None
        self.cycles_per_wallsecond = None
        self.extract_log_file_info()

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

    def read_input_attrs(self):
        """
        Read and return the input attributes from the input file if it exists.

        Returns:
            dict or None: The input attributes as a dictionary, or None if the input file doesn't exist.
        """
        input_file_name = next((f for f in os.listdir(self.folder_path) if f.endswith('.in')), None)
        if input_file_name:
            input_file_path = os.path.join(self.folder_path, input_file_name)
            return read_athenapk_config_file(input_file_path)
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

    def get_all_average_field(self,
                              field: str,
                              in_time: bool = True) -> None:
        """
        Calculate and save the density-weighted averages values for a specified field.

        Args:
            field (str): The name of the field for which to calculate averages.
            in_time (bool, optional): If True, get also the snapshot's current time and save. Default is True.

        Returns:
            None: The function saves the calculated values to a file and does not return a value.
        """
        fields = []
        times = []

        for sim in self.snapshot_list:
            i_snapshot = sim.split('.')[2]
            average_field = self.get_snapshot_field_average(i_snapshot, ('gas', field))
            fields.append(float(average_field))

            if in_time:
                current_time = self.get_snapshot_current_time(i_snapshot)
                times.append(current_time)

        data = np.column_stack((times, fields)) if in_time else fields
        header = f'current_time {field}' if in_time else field
        out_name = f'average_{field}_in_time.txt' if in_time else f'average_{field}.txt'

        np.savetxt(os.path.join(self.folder_path, out_name), data, header=header)

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
                                   weight: str | None = None) -> float:
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
            # Compute non-weighted average using the 'mean' method
            average_quantity = ad[field].mean()
        else:
            # Compute weighted average based on the specified weight field
            average_quantity = ad[field].mean(weight=weight)

        return average_quantity

    def plot_snapshot_field_slice(self,
                                  n_snap: int | str,
                                  field: tuple[str, str],
                                  axis: str = 'z',
                                  cmap: str = 'viridis') -> None:
        """
        Plot a slice of a snapshot field.

        Args:
            n_snap (int or str): The snapshot number to plot.
            field (tuple of str): A tuple specifying the field to plot (e.g., ('gas', 'density')).
            axis (str, optional): The axis for slicing (e.g., 'z'). Defaults to 'z'.
            cmap (str, optional): The colormap to use for visualization. Defaults to 'viridis'.
        """
        ds = self._load_snapshot_data(n_snap)
        _plot = yt.SlicePlot(ds, axis, field)
        _plot.set_cmap(field=field, cmap=cmap)
        return _plot

    def get_integral_time(self,
                          n_snap: int | str):
        """
        Get the integral time of a snapshot.

        Args:
            n_snap (int or str): The snapshot number to analyze.
        """
        pass  # Placeholder for future implementation

