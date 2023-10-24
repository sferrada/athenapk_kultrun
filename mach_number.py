from src.load_model import (output_dir,
                            LoadAthenaPKRun)

# Usage example
run = 'turb_nGPU1_nc256_M##_B0.05'
sim = LoadAthenaPKRun(output_dir(run))

# Non-parallelized version
sim.get_all_average_field('mach_number')

# # Parallelized verion
# import os
# import numpy as np
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
