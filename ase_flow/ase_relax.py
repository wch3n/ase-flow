from jobflow import job
from ase import Atoms
from ase.optimize import BFGS, FIRE
from ase.io import read, write
from pymatgen.io.ase import AseAtomsAdaptor
import os, socket

def mace_calculator(atoms, model):
    from mace.calculators import MACECalculator
    if model is None:
         model = "/gpfs/scratch/acad/htbase/wchen/mxene/mlip/pbe0-rvv10_8/Ti/train/mace-omat-0-medium.model"
    calc = MACECalculator(model_paths=model, device='cuda')
    atoms.calc = calc
    return atoms

def orb_calculator(atoms, model):
    from orb_models.forcefield.calculator import ORBCalculator
    from orb_models.forcefield import pretrained
    if model is None:
        model = pretrained.orb_v3_conservative_inf_omat(device='cuda', precision="float32-high")
    calc = ORBCalculator(model, device='cuda')
    atoms.calc = calc
    return atoms

@job
def relax_ase(path_to_structure="POSCAR", atoms=None, forcefield="mace", model=None, alias=''):
    if atoms is not None:
        atoms = AseAtomsAdaptor.get_atoms(atoms)
    else:
        atoms = read(path_to_structure)
    
    if forcefield.lower() == "mace":
        atoms = mace_calculator(atoms, model)
    elif forcefield.lower() == "orb":
        atoms = orb_calculator(atoms, model)

    initial_atoms = atoms.copy()
    write("POSCAR", initial_atoms)
    dyn = FIRE(atoms, logfile="opt.log")
    dyn.run(fmax=0.01)
    write("CONTCAR", atoms)

    initial_structure = AseAtomsAdaptor.get_structure(initial_atoms)
    final_structure = AseAtomsAdaptor.get_structure(atoms)
    energy = atoms.get_potential_energy()

    output = {
        "output": {
        "structure": final_structure.as_dict(),
        "final_structure": final_structure.as_dict(),
        "energy": energy,
        "final_energy": energy,
        "energy_per_atom": energy / len(final_structure)
        }
    }

    results = {
        "dir_name": f"{socket.gethostname()}:{os.getcwd()}",
        "alias": alias,
        "nsites": len(final_structure),
        "formula_pretty": final_structure.composition.reduced_formula,
        "composition": final_structure.composition.as_dict(),
        "input": {
            "calculator": atoms.calc.name
        },
        "output": output
    }

    return results
