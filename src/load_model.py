import os
import re
import yt
import numpy as np
import scipy as scipy
from collections import deque
from src.commons import read_athenapk_config_file
yt.funcs.mylog.setLevel("ERROR")

class LoadAthenaPKRun:
    def __init__(self, folder_path):
        """
        Initialize an instance of an AthenaPK run with the folder path containing simulation data.

        Args:
            folder_path (str): The path to the folder containing AthenaPK simulation data.
        """
        self.folder_path = folder_path
        self.snapshot_list = None
        self.get_snapshot_list()

        # Input variables
        input_file_name = next((f for f in os.listdir(self.folder_path) if f.endswith('.in')), None)
        if input_file_name:
            input_file_path = os.path.join(folder_path, input_file_name)
            self.input_attrs = read_athenapk_config_file(os.path.join(input_file_path))

        # Output file attributes
        self.walltime = None
        self.time_limit = None
        self.cycle_limit = None
        self.cycles_per_wallsecond = None
        self.extract_log_file_info()


    def extract_log_file_info(self):
        """
        Extract information from the log file in the simulation folder.

        This method extracts walltime, time limit, cycle limit, and cycles per wallsecond from the log file.
        """
        # Define regular expressions for matching the desired patterns
        tlim_pattern = r'tlim=(\S+)'
        nlim_pattern = r'nlim=(\S+)'
        walltime_pattern = r'walltime used = (\S+)'
        zone_cycles_pattern = r'zone-cycles/wallsecond = (\S+)'

        log_file_name = next((f for f in os.listdir(self.folder_path) if f.endswith('.out')), None)
        if log_file_name:
            last_lines = deque(maxlen=5)
            log_file_path = os.path.join(self.folder_path, log_file_name)

            with open(log_file_path, 'r') as file:
                for line in file:
                    last_lines.append(line)

                    if 'tlim=' in line:
                        self.time_limit = float(re.search(tlim_pattern, line).group(1))
                    if 'nlim=' in line:
                        self.cycle_limit = int(re.search(nlim_pattern, line).group(1))
                    if 'walltime used =' in line:
                        self.walltime = float(re.search(walltime_pattern, line).group(1))
                    if 'zone-cycles/wallsecond =' in line:
                        self.cycles_per_wallsecond = float(re.search(zone_cycles_pattern, line).group(1))


    def get_snapshot_list(self) -> None:
        """
        Get a list of snapshot files in the simulation folder.
        """
        # List all files in the folder with a .phdf extension
        self.snapshot_list = [f for f in os.listdir(self.folder_path) if f.endswith('.phdf')]
        self.snapshot_list.sort()


    def get_snapshot_field_info(
            self,
            n_snap: int | str
        ) -> None:
        """
        Get information about available fields in a snapshot.

        Args:
            n_snap (int or str): The snapshot number to analyze.
        """
        snapshot_number_str = str(n_snap).zfill(5)

        try:
            ds = yt.load(os.path.join(self.folder_path, f'parthenon.prim.{snapshot_number_str}.phdf'))

            print('>> [Field] Gas:')
            for elem in dir(ds.fields.gas):
               print(elem)
            print('\n>> [Field] Index:')
            for elem in dir(ds.fields.index):
               print(elem)
            print('\n>> [Field] Parthenon:')
            for elem in dir(ds.fields.parthenon):
               print(elem)

        except FileNotFoundError:
            raise FileNotFoundError('Snapshot not found in the current simulation directory.')


    def get_snapshot_current_time(
            self,
            n_snap: int | str
        ) -> float:
        """
        """
        snapshot_number_str = str(n_snap).zfill(5)

        try:
            ds = yt.load(os.path.join(self.folder_path, f'parthenon.prim.{snapshot_number_str}.phdf'))
            return float(ds.current_time)

        except FileNotFoundError:
            raise FileNotFoundError('Snapshot not found in the current simulation directory.')


    def get_field_average(
            self,
            n_snap: int | str,
            field: tuple[str, str]
        ) -> float:
        """
        Get the average value of a field in a snapshot.

        Args:
            n_snap (int or str): The snapshot number to analyze.
            field (tuple of str): A tuple specifying the field to analyze (e.g., ('gas', 'density')).
        
        Returns:
            averaged_quantity (float): The average value of the field.
        """
        snapshot_number_str = str(n_snap).zfill(5)

        try:
            ds = yt.load(os.path.join(self.folder_path, f'parthenon.prim.{snapshot_number_str}.phdf'))
            ad = ds.all_data()
            averaged_quantity = ad.quantities.weighted_average_quantity(field, ('gas', 'density'))
            return averaged_quantity

        except FileNotFoundError:
            raise FileNotFoundError('Snapshot not found in the current simulation directory.')


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
        snapshot_number_str = str(n_snap).zfill(5)

        try:
            ds = yt.load(os.path.join(self.folder_path, f'parthenon.prim.{snapshot_number_str}.phdf'))

            _plot = yt.SlicePlot(ds, axis, field)
            _plot.set_cmap(field=field, cmap=cmap)
            return _plot

        except FileNotFoundError:
            raise FileNotFoundError('Snapshot not found in the current simulation directory.')


    def get_integral_time(self,
                          n_snap: int | str):
        """
        Get the integral time of a snapshot.

        Args:
            n_snap (int or str): The snapshot number to analyze.
        """
        pass  # Placeholder for future implementation
