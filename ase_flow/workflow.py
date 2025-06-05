from jobflow import job, Flow
from ase.io import read
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp import Poscar
from ase.build import molecule, add_adsorbate
from ase import Atom
from ase_flow.ase_relax import relax_ase
from dataclasses import dataclass, field
import numpy as np
import os


@job
def load_surface(filename="POSCAR"):
    structure = Poscar.from_file(filename).structure
    return structure


@job
def adsorb(surface_structure, adsorbate, rotation, height, position):
    slab = AseAtomsAdaptor.get_atoms(surface_structure)
    if isinstance(adsorbate, str) and os.path.isfile(adsorbate):
        adsorbate = read(adsorbate)
    else:
        adsorbate = molecule(adsorbate)
    adsorbate.rotate(rotation["x"], "x")
    adsorbate.rotate(rotation["y"], "y")
    adsorbate.rotate(rotation["z"], "z")
    add_adsorbate(slab, adsorbate, height=height, position=position)
    return AseAtomsAdaptor.get_structure(slab)


@job
def protonate(surface_structure, anchor_site: int, height=1.0, offset=(0.0, 0.0)):
    slab = AseAtomsAdaptor.get_atoms(surface_structure)
    anchor_pos = slab.positions[anchor_site]
    h_pos = anchor_pos + np.array([offset[0], offset[1], height])
    slab.append(Atom("H", h_pos))
    return AseAtomsAdaptor.get_structure(slab)


@job
def remove_indices(surface_structure, indices_to_remove):
    slab = AseAtomsAdaptor.get_atoms(surface_structure)
    for i in sorted(indices_to_remove, reverse=True):
        del slab[i]
    return AseAtomsAdaptor.get_structure(slab)


@dataclass
class RelaxWorkflow:
    structure: object = None
    filename: str = "POSCAR"
    fix_below: float = None
    forcefield: str = "mace"
    rattle: float = None
    model: str = None
    alias: str = ""
    free_energy: bool = True
    mode_fe: str = "harmonic"
    temperature: float = 298.15
    pressure: float = 101325
    geometry: str = "linear"
    symmetrynumber: int = 2
    spin: int = 0
    job_name: str = "substrate relax"

    def build(self):
        if self.structure is not None:
            structure = self.structure
            load = None
        else:
            load = load_surface(filename=self.filename)
            structure = load.output
        relax = relax_ase(
            atoms=structure,
            fix_below=self.fix_below,
            forcefield=self.forcefield,
            model=self.model,
            alias=self.alias,
            free_energy=self.free_energy,
            mode_fe=self.mode_fe,
            temperature=self.temperature,
            pressure=self.pressure,
            geometry=self.geometry,
            symmetrynumber=self.symmetrynumber,
            spin=self.spin
        )
        relax.name = self.job_name
        jobs = [relax] if load is None else [load, relax]
        return Flow(jobs), relax


@dataclass
class AdsorbateRelaxWorkflow:
    structure: object = None
    filename: str = "POSCAR"
    adsorbate: str = "CO2"
    rotation: dict = field(default_factory=lambda: {"x": 0, "y": 0, "z": 0})
    height: float = 2.0
    position: tuple = ((0, 0),)
    fix_below: float = None
    rattle: float = None
    forcefield: str = "mace"
    model: str = None
    alias: str = ""
    free_energy: bool = True
    mode_fe = "harmonic"
    job_name: str = "adsorbate relax"

    def build(self):
        if self.structure is not None:
            surface_structure = self.structure
            load = None
        else:
            load = load_surface(filename=self.filename)
            surface_structure = load.output
        add = adsorb(
            surface_structure,
            adsorbate=self.adsorbate,
            rotation=self.rotation,
            height=self.height,
            position=self.position,
        )
        relax = relax_ase(
            atoms=add.output,
            fix_below=self.fix_below,
            forcefield=self.forcefield,
            model=self.model,
            alias=self.alias,
            free_energy=self.free_energy,
            mode_fe=self.mode_fe,
        )
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
    fix_below: float = None
    rattle: float = None
    forcefield: str = "mace"
    model: str = None
    alias: str = ""
    free_energy: bool = True
    mode_fe = "harmonic"
    job_name: str = "adsorbate relax"

    def build(self):
        if self.structure is not None:
            surface_structure = self.structure
            load = None
        else:
            load = load_surface(filename=self.filename)
            surface_structure = load.output
        add = protonate(
            surface_structure,
            anchor_site=self.anchor_site,
            height=self.height,
            offset=self.offset,
        )
        relax = relax_ase(
            atoms=add.output,
            fix_below=self.fix_below,
            forcefield=self.forcefield,
            model=self.model,
            alias=self.alias,
            free_energy=self.free_energy,
            mode_fe=self.mode_fe
        )
        relax.name = self.job_name
        jobs = [add, relax] if load is None else [load, add, relax]
        return Flow(jobs), relax

@dataclass
class DesorbRelaxWorkflow:
    structure: object = None
    filename: str = "POSCAR"
    indices_to_remove: list = None
    fix_below: float = None
    rattle: float = None
    forcefield: str = "mace"
    model: str = None
    alias: str = ""
    free_energy: bool = True
    mode_fe = "harmonic"
    job_name: str = "adsorbate relax"

    def build(self):
        if self.structure is not None:
            surface_structure = self.structure
            load = None
        else:
            load = load_surface(filename=self.filename)
            surface_structure = load.output
        desorb = remove_indices(
            surface_structure,
            indices_to_remove=self.indices_to_remove
        )
        relax = relax_ase(
            atoms=desorb.output,
            fix_below=self.fix_below,
            rattle=self.rattle,
            forcefield=self.forcefield,
            model=self.model,
            alias=self.alias,
            free_energy=self.free_energy,
            mode_fe=self.mode_fe,
        )
        relax.name = self.job_name
        jobs = [desorb, relax] if load is None else [load, desorb, relax]
        return Flow(jobs), relax
