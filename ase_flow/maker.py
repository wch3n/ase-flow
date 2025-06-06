#!/usr/bin/env python3

from .workflow import (
    AdsorbateRelaxWorkflow,
    ProtonationRelaxWorkflow,
    RelaxWorkflow,
    DesorbRelaxWorkflow
)
from jobflow.managers.fireworks import flow_to_workflow
from fireworks import LaunchPad
from ase.build import molecule
from ase.io import read, write
from os.path import join
from os import getcwd
from jobflow import Flow
from ase.cluster import Icosahedron, Octahedron
from atomate2.vasp.jobs.core import StaticMaker
from atomate2.vasp.sets.core import StaticSetGenerator
from pymatgen.io.ase import AseAtomsAdaptor

lpad = LaunchPad.auto_load()

user_incar_settings = {
    "IVDW": 12,
    "GGA": "PE",
    "ISIF": 1,
    "KPAR": 4,
    "LASPH": "True",
    "NELM": 50,
    "ALGO": "Normal",
    "EDIFF": 1e-5,
    "EDIFFG": -0.01,
    "ISMEAR": 0,
    "SIGMA": 0.02,
    "ENCUT": 520,
    "PRECFOCK": "Fast",
    "ISPIN": 2,
    "NSW": 0,
    "LSORBIT": "False",
    "LWAVE": False,
    "LCHARG": False,
}
user_kpoints_settings = {"reciprocal_density": 250}

# cu_cluster = Icosahedron("Cu", 2)
# write('cu.xyz', cu_cluster)

model_default = "/gpfs/scratch/acad/htbase/wchen/mxene/mlip/pbe-d3/Ti_C_O_H_F_Cu/models/ft-omat_0-00_stagetwo.model"

def molecule_relax(molecule_name, temperature=298.15, pressure=101325, geometry="linear", symmetrynumber=2, spin=0, job_name="molecule relax", model=model_default):
    mol = molecule(molecule_name)
    mol.center(vacuum=10.0)
    mol.set_pbc(True)
    flow_relax, relax_job = RelaxWorkflow(
        structure=AseAtomsAdaptor.get_structure(mol),
        temperature=temperature,
        pressure=pressure,
        geometry="linear",
        symmetrynumber=symmetrynumber,
        spin=spin,
        mode_fe="ideal",
        forcefield="mace",
        model=model,
        free_energy=True,
        job_name=job_name
    ).build()

    flow = Flow(flow_relax)
    lpad.add_wf(flow_to_workflow(flow))

def substrate_relax(poscar, topdir, job_name="substrate relax", model=model_default):
    flow_relax, relax_job = RelaxWorkflow(
        filename=join(topdir, poscar),
        forcefield="mace",
        model=model,
        job_name=job_name,
        free_energy=True,
    ).build()

    flow = Flow(flow_relax)
    lpad.add_wf(flow_to_workflow(flow))

def vasp_relax():
    # vasp relax
    custom_generator = StaticSetGenerator(
        user_incar_settings=user_incar_settings,
        user_kpoints_settings=user_kpoints_settings,
        force_gamma=True,
    )
    static_maker = StaticMaker(input_set_generator=custom_generator)
    job_static = static_maker.make(relax_job.output["output"]["output"]["final_structure"])

    flow = Flow(job_static)
    lpad.add_wf(flow_to_workflow(flow))

def adsorbate_relax(topdir, job_name="adsorbate relax", model=model_default):
    # adsorbate 
    flow_adsorb, relax_job = AdsorbateRelaxWorkflow(
        filename=join(topdir,'calc/slab/CONTCAR'),
        adsorbate='CO2',
        rotation={'x': 90, 'y':0, 'z':0},
        forcefield = 'mace',
        model = model,
        height = 2.0,
        position = (4.8, 7.5),
        alias = '',
        job_name = job_name 
    ).build()

    flow = Flow(flow_adsorb)
    lpad.add_wf(flow_to_workflow(flow))

def protonation_from_job(relax_job, model=model_default):
    # protonation 1
    flow_proto, _ = ProtonationRelaxWorkflow(
        structure = relax_job.output["output"]["output"]["final_structure"],
        forcefield = 'mace',
        model = model,
        anchor_site = 51,
        height = 4.0,
    ).build()

    flow = Flow(flow_proto)
    lpad.add_wf(flow_to_workflow(flow))

def protonation_relax(rel_path, topdir, anchor_site, height=1.0, offset=(0,0), model=model_default):
    # protonation 1
    flow_proto, _ = ProtonationRelaxWorkflow(
        filename = join(topdir, rel_path),
        forcefield = 'mace',
        model = model,
        anchor_site = anchor_site,
        height = height,
        offset = offset,
        free_energy = True,
        job_name = 'adsorbate relax'
    ).build()

    flow = Flow(flow_proto)
    lpad.add_wf(flow_to_workflow(flow))
    
def desorption_relax(rel_path, indices, model=model_default):
    flow_desorb, _ = DesorbRelaxWorkflow(
        filename = join(topdir, rel_path),
        indices_to_remove = indices,
        forcefield = 'mace',
        model = model,
        free_energy = True,
        job_name = 'adsorbate relax'
    ).build()

    flow = Flow(flow_desorb)
    lpad.add_wf(flow_to_workflow(flow))

if __name__ == '__main__':
    topdir = '/gpfs/scratch/acad/htbase/wchen/mxene/co2rr/oh_f-term/ML_0_33/mace'
    molecule_relax('CO', temperature=298.15, pressure=101325, geometry='linear', symmetrynumber=2, spin=0)
    #substrate_relax('slab/POSCAR', job_name='substrate relax')
    #substrate_relax('co2/POSCAR', job_name="adsorbate relax")
    #protonation_relax('ch3o/calc/launcher_2025-06-05-10-18-52-681095/CONTCAR', anchor_site=167, height=1.0, offset=(0.0,0.0))
    #desorption_relax('ch3o-h/calc/launcher_2025-06-05-11-32-08-283160/CONTCAR', indices=[273,274,215,275,276,277])
