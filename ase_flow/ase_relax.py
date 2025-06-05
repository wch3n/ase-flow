from jobflow import job
from ase import Atoms
from ase.optimize import BFGS, LBFGS, FIRE
from ase.optimize.precon import PreconLBFGS
from ase.constraints import FixAtoms
from ase.io import read, write, Trajectory
from pymatgen.io.ase import AseAtomsAdaptor
from ase_flow.vibrations import calc_free_energy
import os, socket


def mace_calculator(atoms, model):
    from mace.calculators import MACECalculator

    if model is None:
        model = "/gpfs/scratch/acad/htbase/wchen/mxene/mlip/pbe0-rvv10_8/Ti/train/mace-omat-0-medium.model"
    calc = MACECalculator(model_paths=model, device="cuda")
    atoms.calc = calc
    return atoms


def orb_calculator(atoms, model):
    from orb_models.forcefield.calculator import ORBCalculator
    from orb_models.forcefield import pretrained

    if model is None:
        model = pretrained.orb_v3_conservative_inf_omat(
            device="cuda", precision="float32-high"
        )
    calc = ORBCalculator(model, device="cuda")
    atoms.calc = calc
    return atoms


@job
def relax_ase(
    path_to_structure="POSCAR",
    atoms=None,
    fix_below=None,
    rattle=None,
    forcefield="mace",
    model=None,
    alias="",
    free_energy=True,
    mode_fe="harmonic",
    temperature=298.15,
    pressure=101325,
    geometry="linear",
    symmetrynumber=2,
    spin=0
):
    if atoms is not None:
        atoms = AseAtomsAdaptor.get_atoms(atoms)
    else:
        atoms = read(path_to_structure)

    if forcefield.lower() == "mace":
        atoms = mace_calculator(atoms, model)
    elif forcefield.lower() == "orb":
        atoms = orb_calculator(atoms, model)

    if fix_below:
        fixed_indices = [i for i, pos in enumerate(atoms.positions) if pos[2] < fix_below]
        atoms.set_constraint(FixAtoms(indices=fixed_indices))

    if rattle is not None:
        atoms.rattle(stdev=rattle, seed=100)

    initial_atoms = atoms.copy()
    write("POSCAR", initial_atoms)
    """opt1 = PreconLBFGS(atoms, logfile="opt1.log")
    traj = Trajectory('opt1.traj', 'w', atoms)
    opt1.attach(traj.write, interval=10)
    opt1.run(fmax=0.02)"""
    #
    opt = PreconLBFGS(atoms, logfile="opt.log")
    traj = Trajectory("opt.traj", "w", atoms)
    opt.attach(traj.write, interval=10)
    opt.run(fmax=0.0095)
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
            "energy_per_atom": energy / len(final_structure),
        }
    }

    if free_energy:
        free_energy = calc_free_energy(
            atoms,
            temperature=temperature,
            pressure=pressure,
            mode=mode_fe,
            geometry=geometry,
            symmetrynumber=symmetrynumber,
            spin=spin,
        )
        output["output"]["free_energy"] = free_energy

    results = {
        "dir_name": f"{socket.gethostname()}:{os.getcwd()}",
        "alias": alias,
        "nsites": len(final_structure),
        "formula_pretty": final_structure.composition.reduced_formula,
        "composition": final_structure.composition.as_dict(),
        "input": {"calculator": atoms.calc.name},
        "output": output,
    }

    return results
