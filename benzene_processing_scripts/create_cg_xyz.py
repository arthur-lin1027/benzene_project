"""
Generalizes the three create_{group_name}_xyz.py, takes in a command line argument.
"""
import sys
from ase.io import read, write
import numpy as np
from pathlib import Path
from anisoap.asecg.CGRep import CGInfo, coarsen_frame_manybeads

if len(sys.argv) != 4:
    print("Error: need to specify input xyz, output folder, and group name")
    print("Usage: python create_cg_xyz.py benzene_xyz/benzene_tagged_groups.xyz benzene_xyz/ GroupC2H2Adjacent")
    print("Input xyz must have names of tagged groups in arrays.")
    sys.exit(1)

input_fname = sys.argv[1]
output_folder = sys.argv[2]
group_name = sys.argv[3]
output_name = Path(sys.argv[2]).resolve()/f"benzene-{group_name}.xyz"
if output_name.exists():
    print(f"Error: {output_name} exists. Delete before running script")
    sys.exit(1)
print("Reading Input File: ", Path(input_fname).resolve())
print("Writing Output to: ", output_name)
frames = read(Path(input_fname).resolve(), ":")
print(f"Creating groups with {group_name=}")

for i, frame in enumerate(frames):
    if group_name not in frame.arrays.keys():
        print(f"Error, {group_name} not in frame {i}")
        sys.exit(1)
# %%
cg_frames = []
for frame in frames:
    cg_info_list = []
    for unique_id in np.unique(frame.arrays[group_name]):
        cg_info_list.append(
            CGInfo(
                cg_indices=np.where(frame.arrays[group_name] == unique_id)[0],
                name=group_name,
                symbol="X",
            )
        )
    cg_frame = coarsen_frame_manybeads(frame, cg_info_list, calc_geometries=[True] * len(cg_info_list))
    cg_frames.append(cg_frame)

#%%
# coarsen_frame_manybeads(frame, cg_info_list, calc_geometries=[True] * len(cg_info_list))
#%%
for frame in cg_frames:
    frame.arrays["orientation"] = np.roll(frame.arrays["c_q"], -1, axis=1)
    frame.arrays["aspherical_shape"] = np.vstack((frame.arrays["c_diameter[1]"], frame.arrays["c_diameter[2]"], frame.arrays["c_diameter[3]"])).T

# %%
write(output_name, cg_frames)