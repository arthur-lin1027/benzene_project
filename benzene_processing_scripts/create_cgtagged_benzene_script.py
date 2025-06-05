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
import sys
from pathlib import Path

# %%
def find_adjacent_H_from_C(c_index: int, frame: Atoms) -> int:
    """
    Given a Carbon Atom, find the bonded Hydrogen Atom.
    We extract the c_index'th row of the distance matrix and extract the rows of the distance matrix that correspond with the hydrogens. find the shortest
    nonzero distance in that row. The index corresponding with that will be the H
    because the CH bond is the shortest distance in the frame.
    """
    ell_id = frame.arrays["corresponding_ellipsoid"][c_index]
    hydrogen_indices = np.where((frame.arrays["corresponding_ellipsoid"] == ell_id) & (frame.arrays["numbers"] == 1))[0]
    distances = frame.get_all_distances()
    distances_from_c = distances[c_index][hydrogen_indices]
    indices_subset = np.argsort(distances_from_c)
    h_index = hydrogen_indices[indices_subset[0]] 
    assert frame[h_index].symbol == "H"
    return h_index 

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
    ell_id = frame.arrays["corresponding_ellipsoid"][h_index]
    carbon_indices = np.where((frame.arrays["corresponding_ellipsoid"] == ell_id) & (frame.arrays["numbers"] == 6))[0]
    distances = frame.get_all_distances()
    distances_from_h = distances[h_index][carbon_indices]
    indices_subset = np.argsort(distances_from_h)
    c_index = carbon_indices[indices_subset[0]] 
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
    atom_indices = np.where((frame.arrays["corresponding_ellipsoid"] == ell_id) & (frame.arrays["numbers"] == 6))[0]  
    distances = frame.get_all_distances(mic=True)
    # We only consider distances between atoms in the same benzene.
    # Index 2 and 3 correspond with bonded carbons
    distances_from_c = distances[c_index][atom_indices]    
    indices_subset = np.argsort(distances_from_c)
    c1_index_subset = indices_subset[1] 
    c2_index_subset = indices_subset[2]
    assert frame[atom_indices][c1_index_subset].symbol == "C" and frame[atom_indices][c2_index_subset].symbol, f"{atom_indices=}\n{c1_index_subset=}\n{c2_index_subset=}"
    # what we found above are indices within the subset of atoms.
    # convert these subset indices into global indicies (i.e. indices within the frame)
    c1_index = atom_indices[c1_index_subset]
    c2_index = atom_indices[c2_index_subset]
    assert frame[c1_index].symbol == "C" and frame[c2_index].symbol == "C"
    return (c1_index, c2_index)

def generate_consec_pairs(c_index: int, frame: Atoms) -> list[int]:
    """
    Generate a list of integers that list consecutive pairs of carbon in a benzene.
    e.g if the returned list is [5, 8, 6, 2, 4, 7], then (5, 8), (6, 2) and (4, 7) are consectuive pairs. 
    """
    l = [c_index]
    c_index1, _ = find_adjacent_Cs_from_C(c_index, frame)
    l.append(c_index1)    # Found first pair, (c_index, c_index1)
    c_index1a, c_index1b = find_adjacent_Cs_from_C(c_index1, frame)    # one of these is c_index, the other generates the new pair.
    c_index2 = c_index1a if c_index1a not in l else c_index1b 
    l.append(c_index2)
    c_index2a, c_index2b = find_adjacent_Cs_from_C(c_index2, frame)    # One of these is c_index1, the other is c_index2's pair (i.e. c_index3)
    c_index3 = c_index2a if c_index2a not in l else c_index2b 
    l.append(c_index3)     # Found 2nd pair, (c_index2, c_index3)
    c_index3a, c_index3b = find_adjacent_Cs_from_C(c_index3, frame)    # One of these is c_index2, the other generates the new pair.
    c_index4 = c_index3a if c_index3a not in l else c_index3b 
    l.append(c_index4)
    c_index4a, c_index4b = find_adjacent_Cs_from_C(c_index4, frame)    # One of these is c_index3, the other is c_index4's pair (i.e. c_index5)
    c_index5 = c_index4a if c_index4a not in l else c_index4b
    l.append(c_index5)
    return (l[0], l[1]), (l[2], l[3]), (l[4], l[5])

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
if len(sys.argv) != 2:
    print("Error: Requires Input xyz and frames to filter")
    print("Example: python create_cg_tagged_benzene_script.py benzenes_xyz/benzenes.xyz")
    print("Example: python create_cg_tagged_benzene_script.py benzenes_xyz/edgarbenzene_train_tagged.xyz")
    sys.exit(1)

xyz_in = sys.argv[1]
assert Path(xyz_in).exists(), f"Error: {Path(xyz_in).resolve()} does not exist"
path_out = Path(xyz_in).parent/f"{xyz_in}_tagged_groups"
print(f"Reading {Path(xyz_in).resolve()}")
frames = read(Path(xyz_in), ":")
# frames = [frame for i, frame in enumerate(frames) if i not in filter]
frame = frames[0]
# %%
# print(find_adjacent_H_from_C(0, frames[0]))
# print(find_adjacent_C_from_H(6, frames[0]))
# print(find_adjacent_Cs_from_C(0, frames[0]))
# print(find_cross_Cs_from_C(0, frames[0]))
# %%
print("Assigning GroupCH")
for frame in tqdm(frames):
    frame.arrays["GroupCH"] = np.full(len(frame), np.nan)
    c_indices = [i for i, atom in enumerate(frame) if atom.symbol == "C"]
    group_id = 0
    for c_index in c_indices:
        h_index = find_adjacent_H_from_C(c_index, frame)
        assert np.isnan(frame.arrays["GroupCH"][c_index])
        assert np.isnan(frame.arrays["GroupCH"][h_index])
        frame.arrays["GroupCH"][c_index] = group_id
        frame.arrays["GroupCH"][h_index] = group_id
        group_id += 1
    assert not np.any(np.isnan(frame.arrays["GroupCH"]))

# %%
print("Assigning GroupC2H2Adjacent")
for i, frame in enumerate(tqdm(frames)):
    frame.arrays["GroupC2H2Adjacent"] = np.full(len(frame), np.nan)
    c_indices = [i for i, atom in enumerate(frame) if atom.symbol == "C"]
    group_id = 0
    ell_id_arr = np.unique(frame.arrays["corresponding_ellipsoid"])
    for ell_id in ell_id_arr:
        c_indices = np.where((frame.arrays["corresponding_ellipsoid"] == ell_id) & (frame.arrays["numbers"] == 6))[0]
        c_index = c_indices[0]
        pair1, pair2, pair3 = generate_consec_pairs(c_index, frame)
        h_index1_pair1 = find_adjacent_H_from_C(pair1[0], frame)
        h_index2_pair1 = find_adjacent_H_from_C(pair1[1], frame)
        frame.arrays["GroupC2H2Adjacent"][pair1[0]] = group_id
        frame.arrays["GroupC2H2Adjacent"][pair1[1]] = group_id
        frame.arrays["GroupC2H2Adjacent"][h_index1_pair1] = group_id
        frame.arrays["GroupC2H2Adjacent"][h_index2_pair1] = group_id
        group_id+=1

        h_index1_pair2 = find_adjacent_H_from_C(pair2[0], frame)
        h_index2_pair2 = find_adjacent_H_from_C(pair2[1], frame)
        frame.arrays["GroupC2H2Adjacent"][pair2[0]] = group_id
        frame.arrays["GroupC2H2Adjacent"][pair2[1]] = group_id
        frame.arrays["GroupC2H2Adjacent"][h_index1_pair2] = group_id
        frame.arrays["GroupC2H2Adjacent"][h_index2_pair2] = group_id
        group_id+=1
        
        h_index1_pair3 = find_adjacent_H_from_C(pair3[0], frame)
        h_index2_pair3 = find_adjacent_H_from_C(pair3[1], frame)
        frame.arrays["GroupC2H2Adjacent"][pair3[0]] = group_id
        frame.arrays["GroupC2H2Adjacent"][pair3[1]] = group_id
        frame.arrays["GroupC2H2Adjacent"][h_index1_pair3] = group_id
        frame.arrays["GroupC2H2Adjacent"][h_index2_pair3] = group_id
        group_id+=1

    assert not np.any(np.isnan(frame.arrays["GroupC2H2Adjacent"]))

# %% 
# Create Group C2H2 Across
print("Assigning GroupC2H2Across")
for i, frame in enumerate(tqdm(frames)):
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
print(f"Writing tagged xyz to {path_out}")
write("benzene_tagged_groups.xyz", frames)
# %%
