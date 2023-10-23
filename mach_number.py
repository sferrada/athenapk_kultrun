import os
import numpy as np
from src.load_model import LoadAthenaPKRun

# Usage example
run = 'turb_nGPU1_nc256_M##_B0.05'
sim = LoadAthenaPKRun(os.path.join('outputs', run))

# -----------------------------------------------
# Non-parallelized version
times = []
results = []

# Get average Mach number for each snapshot
for i in sim.snapshot_list:
    n_snapshot = i.split('.')[2]

    # Get and save snapshot's current time [s]
    current_time = sim.get_snapshot_current_time(n_snapshot)
    times.append(current_time)

    # Get and save snapshot's density-weighted average Mach number
    average_mach_number = sim.get_field_average(n_snapshot, ('gas', 'mach_number'))
    results.append(float(average_mach_number))

# Save the lists to a single file
data = np.column_stack((times, results))
np.savetxt('output_data.txt', data, header='Time(s) Average_Mach_Number', comments='')

# # -----------------------------------------------
# # Parallelized verion
# from multiprocessing import Pool
# 
# def process_snapshot(snapshot_name):
#     snapshot_number = snapshot_name.split('.')[2]
#     try:
#         average_mach_number = sim.get_field_average(snapshot_number, ('gas', 'mach_number'))
#         print(float(average_mach_number))
#         # return float(average_mach_number)
#     except FileNotFoundError as e:
#         print(f"Error processing snapshot {snapshot_number}: {e}")
#         return None

# # Number of processes to run in parallel
# num_processes = os.cpu_count()  # Use the number of available CPU cores

# # Create a pool of worker processes
# with Pool(num_processes) as pool:
#     results = pool.map(process_snapshot, sim.snapshot_list)

# # # Filter out any None results (snapshots not found)
# # results = [result for result in results if result is not None]

# # # Now, results will contain the average Mach number for each snapshot processed in parallel
# # print(results)
