#%%
import numpy as np
from ase import neighborlist
from ase.io import read, write
from scipy import sparse
# BENZENE_FILE_TRAIN = "/Users/alin62/Documents/Research/edgar-benzene/dataset/benzene_train_FPS_AIMS_PBE0_MBD.xyz"
# BENZENE_FILE_TEST = "/Users/alin62/Documents/Research/edgar-benzene/dataset/benzene_test_AIMS_PBE0_MBD.xyz"
# BENZENE_FILE_VAL = "/Users/alin62/Documents/Research/edgar-benzene/dataset/benzene_val_AIMS_PBE0_MBD.xyz"

BENZENE_FILE_TRAIN = "/Users/alin62/Documents/Research/benzene_xyz/benzene_train_FPS_QE_PBE_TS.xyz"
BENZENE_FILE_TEST = "/Users/alin62/Documents/Research/benzene_xyz/benzene_test_FPS_QE_PBE_TS.xyz"
BENZENE_FILE_VAL = "/Users/alin62/Documents/Research/benzene_xyz/benzene_val_QE_PBE_TS.xyz"
NLIST_KWARGS = {
    "skin": 0.3,   # doesn't matter for this application.
    "sorted": False,
    "self_interaction": False,
    "bothways": True
}

def ensure_12atoms_per_component(n_components, component_list):
    for i in range(n_components):
        if (component_list == i).sum() != 12:
            return False
    return True

def ensure_6h_per_component(frame, n_components, component_list):
    for i in range(n_components):
        if ((component_list == i) & (frame.arrays['numbers'] == 1)).sum() != 6:
            return False
    return True

def ensure_6c_per_component(frame, n_components, component_list):
    for i in range(n_components):
        if ((component_list == i) & (frame.arrays['numbers'] == 6)).sum() != 6:
            return False
    return True

def tag_frames(frames):
    for i, frame in enumerate(frames):
        # Initialize all frames with a -1 particle identifier.
        frame.arrays["corresponding_ellipsoid"] = np.array(len(frame) * [-1])
    for i, frame in enumerate(frames):
        nl = neighborlist.build_neighbor_list(frame, **NLIST_KWARGS)
        matrix = nl.get_connectivity_matrix(sparse=False)
        n_components, component_list = sparse.csgraph.connected_components(matrix)
        if n_components * 12 != len(frame):
            print(f"frame {i} improper n_components, should have {len(frame)/12}, has {n_components}")
        elif not ensure_12atoms_per_component(n_components, component_list):
            print(f"frame {i} improper number of atoms")
        elif not ensure_6h_per_component(frame, n_components, component_list):
            print(f"frame {i} improper number of h")
        elif not ensure_6c_per_component(frame, n_components, component_list):
            print(f"frame {i} improper number of c")
        else:
            frame.arrays["corresponding_ellipsoid"] = component_list
#%%
frames = read(BENZENE_FILE_TRAIN, ":")
tag_frames(frames)
frames_clean = [frame for frame in frames if np.all(frame.arrays["corresponding_ellipsoid"] != -1)]
# %%
write("/Users/alin62/Documents/Research/benzene_xyz/edgarbenzene_train_tagged.xyz", frames_clean)
# %%
frames = read(BENZENE_FILE_TEST, ":")
tag_frames(frames)
frames_clean = [frame for frame in frames if np.all(frame.arrays["corresponding_ellipsoid"] != -1)]
# %%
write("/Users/alin62/Documents/Research/benzene_xyz/edgarbenzene_test_tagged.xyz", frames_clean)
# %%
frames = read(BENZENE_FILE_VAL, ":")
tag_frames(frames)
frames_clean = [frame for frame in frames if np.all(frame.arrays["corresponding_ellipsoid"] != -1)]
# %%
write("/Users/alin62/Documents/Research/benzene_xyz/edgarbenzene_val_tagged.xyz", frames_clean)
# %%
