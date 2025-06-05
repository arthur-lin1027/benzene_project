# %%
"""
Takes in benzenes.xyz
Adds additional fields specifying all possible schema with 3CG beads per benzene
"""
import numpy as np
from ase.io import read, write
from ase import Atom, Atoms
from ase.geometry import get_distances
from tqdm.auto import tqdm

# %%
def find_adjacent_H_from_C(c_index: int, frame: Atoms) -> int:
    """
    Given a Carbon Atom, find the bonded Hydrogen Atom.
    We extract the c_index'th row of a distance matrix and find the shortest
    nonzero distance in that row. The index corresponding with that will be the H
    because the CH bond is the shortest distance in the frame.
    """
    assert frame[c_index].symbol == "C"
    distances = frame.get_all_distances()
    distances_from_c = distances[c_index]
    # The 0th index of this argsort is itself, 1st index will be the adjacent H
    h_index = np.argsort(distances_from_c)[1]
    assert frame[h_index].symbol == "H"
    return h_index


def find_adjacent_C_from_H(h_index: int, frame: Atoms) -> int:
    """
    Symmetric Operation to find_adjacent_H_from_C function,
    with similar implementation.
    """
    assert frame[h_index].symbol == "H"
    distances = frame.get_all_distances()
    distances_from_h = distances[h_index]
    # The 0th index of this argsort is itself, 1st index will be the adjacent C
    c_index = np.argsort(distances_from_h)[1]
    assert frame[c_index].symbol == "C"
    return c_index


def find_adjacent_Cs_from_C(c_index: int, frame: Atoms) -> tuple[int, int]:
    """
    Return the two bonded Carbons to a given carbon in benzene.
    In finding C-C bonds, it's possible for atoms from another benzene molecule
    to be closer to a C than an adjacent bonded C.
    Hence, we need to explicitly consider atoms on a benzene-by-benzene basis
    """
    assert frame[c_index].symbol == "C"
    # identify atoms in the same benzene molecule
    ell_id = frame.arrays["corresponding_ellipsoid"][c_index]
    atom_indices = np.where(frame.arrays["corresponding_ellipsoid"] == ell_id)[0]  
    distances = frame.get_all_distances()
    # We only consider distances between atoms in the same benzene.
    # Index 2 and 3 correspond with bonded carbons
    distances_from_c = distances[c_index][atom_indices]    
    indices_subset = np.argsort(distances_from_c)
    c1_index_subset = indices_subset[2] 
    c2_index_subset = indices_subset[3]
    assert frame[atom_indices][c1_index_subset].symbol == "C" and frame[atom_indices][c2_index_subset].symbol, f"{atom_indices=}\n{c1_index_subset=}\n{c2_index_subset=}"
    # what we found above are indices within the subset of atoms.
    # convert these subset indices into global indicies (i.e. indices within the frame)
    c1_index = atom_indices[c1_index_subset]
    c2_index = atom_indices[c2_index_subset]
    assert frame[c1_index].symbol == "C" and frame[c2_index].symbol == "C"
    return (c1_index, c2_index)

def find_cross_Cs_from_C(c_index: int, frame: Atoms) -> int:
    """
    Return the index of the carbon sitting across another carbon, within a single molecule.
    """
    assert frame[c_index].symbol == "C"
    ell_id = frame.arrays["corresponding_ellipsoid"][c_index]
    atom_indices = np.where((frame.arrays["corresponding_ellipsoid"] == ell_id) & (frame.arrays['numbers'] == 6))[0]  
    distances = frame.get_all_distances()
    # We only consider distances between atoms in the same benzene.
    # Index 4 corresponds with the cross carbon 
    distances_from_c = distances[c_index][atom_indices]    
    c4_index_subset = np.argsort(distances_from_c)[-1]
    c4_index = atom_indices[c4_index_subset]
    assert frame[c4_index].symbol == "C", f"Error occured:\n{frame=}\n{c4_index=}\n{frame[c4_index]=}"
    return c4_index
# %%
frames = read("../benzene_xyz/edgarbenzene_val.xyz", ":")
frame = frames[0]
# %%
print(find_adjacent_H_from_C(0, frames[0]))
print(find_adjacent_C_from_H(6, frames[0]))
print(find_adjacent_Cs_from_C(0, frames[0]))
print(find_cross_Cs_from_C(0, frames[0]))
# %%
for i, frame in enumerate(frames[63:], start=63):
    frame.arrays["GroupCH"] = np.full(len(frame), np.nan)
    c_indices = [i for i, atom in enumerate(frame) if atom.symbol == "C"]
    group_id = 0
    print(i)
    for c_index in c_indices:
        h_index = find_adjacent_H_from_C(c_index, frame)
        assert np.isnan(frame.arrays["GroupCH"][c_index])
        assert np.isnan(frame.arrays["GroupCH"][h_index])
        frame.arrays["GroupCH"][c_index] = group_id
        frame.arrays["GroupCH"][h_index] = group_id
        group_id += 1
    assert not np.any(np.isnan(frame.arrays["GroupCH"]))

# %%
for i, frame in enumerate(frames):
    frame.arrays["GroupC2H2Adjacent"] = np.full(len(frame), np.nan)
    c_indices = [i for i, atom in enumerate(frame) if atom.symbol == "C"]
    group_id = 0
    for c_index in c_indices:
        if not np.isnan(frame.arrays["GroupC2H2Adjacent"][c_index]):
            continue
        c_index1, c_index2 = find_adjacent_Cs_from_C(c_index, frame)
        if np.isnan(frame.arrays["GroupC2H2Adjacent"][c_index1]):
            h_index = find_adjacent_H_from_C(c_index, frame)
            h_index1 = find_adjacent_H_from_C(c_index1, frame)
            frame.arrays["GroupC2H2Adjacent"][c_index] = group_id
            frame.arrays["GroupC2H2Adjacent"][c_index1] = group_id
            frame.arrays["GroupC2H2Adjacent"][h_index] = group_id
            frame.arrays["GroupC2H2Adjacent"][h_index1] = group_id
            group_id += 1
        elif np.isnan(frame.arrays["GroupC2H2Adjacent"][c_index2]):
            h_index = find_adjacent_H_from_C(c_index, frame)
            h_index2 = find_adjacent_H_from_C(c_index2, frame)
            frame.arrays["GroupC2H2Adjacent"][c_index] = group_id
            frame.arrays["GroupC2H2Adjacent"][c_index2] = group_id
            frame.arrays["GroupC2H2Adjacent"][h_index] = group_id
            frame.arrays["GroupC2H2Adjacent"][h_index2] = group_id
            group_id += 1
        else:
            print(f"Unexpected Error at frame {i}, {c_index=}")
    assert not np.any(np.isnan(frame.arrays["GroupC2H2Adjacent"]))

# %% 
# Create Group C2H2 Across
for i, frame in enumerate(frames):
    frame.arrays["GroupC2H2Across"] = np.full(len(frame), np.nan)
    c_indices = [i for i, atom in enumerate(frame) if atom.symbol == "C"]
    group_id = 0
    for c_index in c_indices:
        if not np.isnan(frame.arrays["GroupC2H2Across"][c_index]):
            continue
        c_index2 = find_cross_Cs_from_C(c_index, frame)
        h_index = find_adjacent_H_from_C(c_index, frame)
        h_index2 = find_adjacent_H_from_C(c_index2, frame)
        frame.arrays["GroupC2H2Across"][c_index] = group_id
        frame.arrays["GroupC2H2Across"][h_index] = group_id
        frame.arrays["GroupC2H2Across"][c_index2] = group_id
        frame.arrays["GroupC2H2Across"][h_index2] = group_id
        group_id += 1
    assert not np.any(np.isnan(frame.arrays["GroupC2H2Across"]))

# %%
write("benzene_tagged_groups.xyz", frames)
# %%
