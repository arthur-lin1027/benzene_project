"""
Given some hypers and an xyz file, store the representation.
"""

import sys 
from ase.io import read
from anisoap.representations import EllipsoidalDensityProjection
from featomic import (
    SoapPowerSpectrum,
    SphericalExpansion,
    SphericalExpansionByPair,
    NeighborList,
)
import metatensor
from ase import Atom, Atoms
from ase.io import read, write
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from CGRep import find_gre
from skmatter.metrics import global_reconstruction_error as GRE
from pathlib import Path
import os

def create_key_from_tuple(my_tuple):
    assert len(my_tuple) == 4
    assert isinstance(my_tuple[0], int) 
    assert isinstance(my_tuple[1], int)
    assert isinstance(my_tuple[2], float)
    assert isinstance(my_tuple[3], str)
    return f"({my_tuple[0]},{my_tuple[1]},{my_tuple[2]:.3f},{my_tuple[3]})"

def create_featomic_descriptor(frames, lmax, nmax, rcut):
    HYPER_PARAMETERS = {
        "cutoff": {
            "radius": 7.0,
            "smoothing": {"type": "ShiftedCosine", "width": 0.5},
        },
        "density": {
            "type": "Gaussian",
            "width": 0.3,
        },
        "basis": {
            "type": "TensorProduct",
            "max_angular": 4,
            "radial": {"type": "Gto", "max_radial": 6},
        },
    }
    
    calculator_rascal = SoapPowerSpectrum(**HYPER_PARAMETERS)
    aa_desc_rascal = calculator_rascal.compute(frames)
    return aa_desc_rascal

def create_anisoap_descriptor(ell_frames, lmax, nmax, rcut):
    # Often drops the square brackets around c_diameter, so add again only if it exists.
    if "c_diameter[1]" in ell_frames[0].arrays.keys():
        pass   # We're all good
    elif "c_diameter1" in ell_frames[0].arrays.keys():
        for ell_frame in ell_frames:
            ell_frame.arrays["c_diameter[1]"] = ell_frame.arrays.pop("c_diameter1")
            ell_frame.arrays["c_diameter[2]"] = ell_frame.arrays.pop("c_diameter2")
            ell_frame.arrays["c_diameter[3]"] = ell_frame.arrays.pop("c_diameter3")
    else:
        print("c_diameters not defined in xyz files! Setting everything to 1")
        for ell_frame in ell_frames:
            ell_frame.arrays["c_diameter[1]"] = np.ones(len(ell_frame))
            ell_frame.arrays["c_diameter[2]"] = np.ones(len(ell_frame))
            ell_frame.arrays["c_diameter[3]"] = np.ones(len(ell_frame))
    
    # Create c_q if it doesn't exist.
    if "c_q" not in ell_frames[0].arrays.keys():
        for ell_frame in ell_frames:
            ell_frame.arrays["c_q"] = np.asarray([[1., 0, 0, 0]] * len(ell_frame))
    AniSOAP_HYPERS = {
        "max_angular": lmax,
        "max_radial": nmax,
        "radial_basis_name": "gto",
        "rotation_type": "quaternion",
        "rotation_key": "c_q",
        "cutoff_radius": rcut,
        "radial_gaussian_width": 1.5,
        "basis_rcond": 1e-8,
        "basis_tol": 1e-4,
    }

    # May need to create another set of hypers with larger cutoff radius for the cg descriptor?
    calculator = EllipsoidalDensityProjection(**AniSOAP_HYPERS)
    # 9 min for lmax=5, nmax=3.
    for frame in tqdm(ell_frames):
        cg_desc = calculator.power_spectrum([frame], show_progress=False, mean_over_samples=False)
    return cg_desc
if __name__ == '__main__':
    if len(sys.argv) != 5+1:
        print("Creates a representation based on the arguments in the arg list. Perfoms minimal type checking.")
        print("Error: Need to specify the following 5 options: {anisoap, featomic}, lmax:int, nmax:int, rcut:float, GroupID:str")
        print("Example Usage 1: python create_cg_rep.py featomic 5 3 7.0 AllAtom")
        print("Example Usage 2: python create_cg_rep.py anisoap 5 3 7.0 GroupC2H2Across")
        print("Example Usage 3: python create_cg_rep.py featomic 5 3 7.0 GroupC2H2Across", "This utilizes the groups assigned but does not calculate geometry")
        sys.exit(1)
    
    # Brief error checking: int and float conversions will fail if not passed properly.

    rep_type = sys.argv[1]
    lmax = int(sys.argv[2])
    nmax = int(sys.argv[3])
    rcut = float(sys.argv[4])
    group_name = sys.argv[5]
    fname_xyz = f"benzenes_{group_name}.xyz"
    assert rep_type in ("anisoap", "featomic")
    assert Path(fname_xyz).exists(), f"{group_name=} does not have a corresponding file {fname_xyz} in {os.cwd()=}."

    # Now do the work. See if file with these hypers exists
    key_tup = (lmax, nmax, rcut, group_name)
    key = create_key_from_tuple(key_tup)
    rep_fname = f"representations-2025/rep_{rep_type}_{key}.mts"
    print(rep_fname)
    if Path(rep_fname).exists():
        print(f"{Path(rep_fname).resolve()} exists!")
        sys.exit(0)
    
    # Create the rep and save it.
    frames = read(fname_xyz, ":")
    if rep_type == "anisoap":
        print("Creating AniSOAP descriptor")
        rep = create_anisoap_descriptor(frames, lmax, nmax, rcut)
    else:
        print("Creating featomic descriptor")
        rep = create_featomic_descriptor(frames, lmax, nmax, rcut)
    
    metatensor.save(Path(rep_fname).resolve(), rep)
        
    
