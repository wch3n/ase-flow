from jobflow import job, Flow
from ase.io import read
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp import Poscar
from ase.build import molecule, add_adsorbate
from ase import Atom
from ase_flow.ase_relax import relax_ase
from dataclasses import dataclass
import numpy as np
import os

@job
def load_surface(filename="POSCAR"):
    structure = Poscar.from_file(filename).structure
    return structure

@job
def adsorb(surface_structure, adsorbate, height, position):
    slab = AseAtomsAdaptor.get_atoms(surface_structure)
    if isinstance(adsorbate, str) and os.path.isfile(adsorbate):
        adsorbate = read(adsorbate)
    else:
        adsorbate = molecule(adsorbate)

    add_adsorbate(slab, adsorbate, height=height, position=position)
    return AseAtomsAdaptor.get_structure(slab)

@job 
def protonate(surface_structure, anchor_site: int, height=1.0, offset=(0.0,0.0)):
    slab = AseAtomsAdaptor.get_atoms(surface_structure)
    anchor_pos = slab.positions[anchor_site]
    h_pos = anchor_pos + np.array([offset[0], offset[1], height])
    slab.append(Atom("H", h_pos))
    return AseAtomsAdaptor.get_structure(slab)
    
@dataclass
class AdsorbateRelaxWorkflow:
    structure: object = None
    filename: str = "POSCAR"
    adsorbate: str = "CO2"
    height: float = 2.0
    position: tuple = (0,0),
    forcefield: str = "mace"
    model: str = None
    alias: str = ''
    job_name: str = "adsorbate relax"
    mode_fe = "harmonic"
    
    def build(self):
        if self.structure is not None:
            surface_structure = self.structure
            load = None
        else:
            load = load_surface(filename = self.filename)
            surface_structure = load.output
        add = adsorb(surface_structure, adsorbate = self.adsorbate, height=self.height, position=self.position)
        relax = relax_ase(atoms=add.output, forcefield=self.forcefield, model=self.model, alias=self.alias, mode_fe=self.mode_fe)
        relax.name = self.job_name
        jobs = [add, relax] if load is None else [load, add, relax]
        return Flow(jobs), relax

@dataclass
class ProtonationRelaxWorkflow:
    structure: object = None
    filename: str = "POSCAR"
    anchor_site: int = 0
    height: float = 1.0
    offset: tuple = (0.0, 0.0)
    forcefield: str = "mace"
    model: str = None
    alias: str = ''
    job_name: str = "adsorbate relax"
    
    def build(self):
        if self.structure is not None:
            surface_structure = self.structure
            load = None
        else:
            load = load_surface(filename = self.filename)
            surface_structure = load.output
        add = protonate(surface_structure, anchor_site=self.anchor_site, height=self.height, offset=self.offset)
        relax = relax_ase(atoms=add.output, forcefield=self.forcefield, model=self.model, alias=self.alias)
        relax.name = self.job_name
        jobs = [add, relax] if load is None else [load, add, relax]
        return Flow(jobs), relax
