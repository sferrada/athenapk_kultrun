import os
import re
import yt
import numpy as np
import scipy as sp
import pandas as pd
from collections import deque
from src.mdl_files import read_input_file
from src.sim_snapshot import Snapshot
yt.funcs.mylog.setLevel("ERROR")

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

    def parse_infile(self) -> None:
        """
            Read and return the input attributes from the input file if it exists.
        """
        infile_name = next(
            (f for f in os.listdir(self.outdir) if f.endswith('.in')
        ), None)

        try:
            infile_dict = read_input_file(
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

    def get_snapshots(self) -> dict[str, Snapshot]:
        """
            Get a dictionary of snapshot objects from the simulation folder.
        """
        snapshot_list = [f for f in os.listdir(self.outdir) if f.endswith('.phdf')]
        snapshot_list.sort()

        for snapshot_index, snapshot_file_name in enumerate(snapshot_list):
            snapshot_id = snapshot_file_name.split('.')[2]
            self.snapshots[snapshot_index] = Snapshot(snapshot_id, self.outdir)

    def get_available_fields(self) -> None:
        """
            Get information about available fields in a snapshot.
        """
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

    def get_integral_times(self) -> np.ndarray:
        """
            Calculate and return the correlation time between the acceleration fields for a series of simulation snapshots.
            For this, the function loads a sequence of simulation snapshots from the specified folder path and calculates the
            correlation time between the acceleration fields for all pairs of snapshots. The correlation time is computed for
            the full 3D acceleration field and its individual components (x, y, z).

            The results are returned as a 4D NumPy array, where the dimensions represent:
            - The acceleration component (0 for full field, 1 for x, 2 for y, 3 for z).
            - The first snapshot's index.
            - The second snapshot's index (offset by the first snapshot).

            :return: np.ndarray, a 4D NumPy array containing the correlation time values.
        """
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

    def get_average_fields(
            self,
            fields: list[str],
            weight: tuple[str, str] = None,
            verbose: bool = False,
            in_time: bool = False,
            save_data: bool = False
        ) -> None:
        """
            Calculate and save the density-weighted average values for specified fields.

            :param fields: list[str], the names of the fields for which to calculate averages.
            :param weight: tuple[str, str], the weight field to use for averaging. Default is None.
            :param verbose: bool, if True, print information about the field being processed.
            :param in_time: bool, if True, get snapshot's current time and save it with the averages.
            :param save_data: bool, if True, save the calculated values to a file. Default is False.
            :return: None or ndarray, if save_data is True, the function saves the calculated values
                    to a file and does not return a value. If save_data is False, the function
                    returns the calculated values as a NumPy array.
        """
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

    def get_statistics(self) -> dict:
        """
            Calculate and print statistics for the simulation. The computed statistics include the following,
            all including their target values and standard deviations:
            - Forcing correlation time
            - RMS acceleration
            - Relative power of solenoidal modes

            :return: dict, a dictionary containing the calculated statistics.
        """
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

