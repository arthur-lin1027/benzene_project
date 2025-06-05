# %%
"""
Reads the benzenes_3cg.xyz and analyzes it against the all atom frames.
"""
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
import sys


if len(sys.argv) != 2:
    print("Error, need to specify CG Group Name")
    sys.exit(1)

group_name = sys.argv[1]
print(f"Reading CG xyz with {group_name=}")
# %%
frames = read("benzenes.xyz", ":")
ell_frames = read(f"benzenes_{group_name}.xyz", ":")

#%%
# Often drops the square brackets around c_diameter, so add again.
for ell_frame in ell_frames:
    ell_frame.arrays["c_diameter[1]"] = ell_frame.arrays.pop("c_diameter1")
    ell_frame.arrays["c_diameter[2]"] = ell_frame.arrays.pop("c_diameter2")
    ell_frame.arrays["c_diameter[3]"] = ell_frame.arrays.pop("c_diameter3")

# Add the c_diameter fields to the all atom frame too.
for frame in frames:
    frame.arrays["c_diameter[1]"] = np.ones(len(frame))
    frame.arrays["c_diameter[2]"] = np.ones(len(frame))
    frame.arrays["c_diameter[3]"] = np.ones(len(frame))
    frame.arrays["c_q"] = np.array([[1.0, 0.0, 0.0, 0.0]] * len(frame))
# %%
lmax = 5
nmax = 3
rcut = 7.0

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
#%%
# 9 min.
cg_desc = calculator.power_spectrum(ell_frames, show_progress=True, mean_over_samples=False)
#%%
cg_desc_mean = metatensor.mean_over_samples(cg_desc, sample_names="center")
Xcg_raw = cg_desc_mean.block().values.squeeze()

#%%
# Rascaline AADesc
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
aa_desc_rascal = aa_desc_rascal.keys_to_samples("center_type")
aa_desc_rascal = aa_desc_rascal.keys_to_properties(
            ["neighbor_1_type", "neighbor_2_type"]
        )
aa_desc_rascal = metatensor.operations.mean_over_samples(
            aa_desc_rascal, sample_names=["center_type", "atom"]
        )
Xa_raw = aa_desc_rascal.block().values.squeeze()


#%%
print(GRE(cg_desc_mean.block().values.squeeze(), aa_desc_rascal.block().values.squeeze())) 
# Results in 0.21
