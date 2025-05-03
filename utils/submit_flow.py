#!/usr/bin/env python3

from ase_flow.workflow import AdsorbateRelaxWorkflow, ProtonationRelaxWorkflow
from jobflow.managers.local import run_locally
from jobflow.managers.fireworks import flow_to_workflow
from fireworks import LaunchPad
from ase.io import read, write
from os.path import join
from jobflow import Flow

from ase.cluster import Icosahedron, Octahedron

cu_cluster = Icosahedron("Cu", 2)
write('cu.xyz', cu_cluster)

topdir = '/gpfs/scratch/acad/htbase/wchen/test'
model = '/gpfs/scratch/acad/htbase/wchen/mxene/mlip/pbe-d3/Ti3C2-OH-F-Cu-CO2/models/ft-omat_0-00_stagetwo.model'

flow_adsorb, relax_job = AdsorbateRelaxWorkflow(
    filename=join(topdir,'Ti3C2_OH_F.POSCAR'),
    adsorbate=join(topdir,'cu.xyz'),
    forcefield = 'mace',
    model = model,
    height = 3.0,
    position = (5,5),
    alias = 'cu13_ico'
).build()

flow_proto, _ = ProtonationRelaxWorkflow(
    structure = relax_job.output["output"]["output"]["final_structure"],
    forcefield = 'mace',
    model = model,
    anchor_site = 51,
    height = 4.0,
).build()

flow = Flow(flow_adsorb + flow_proto)

lpad = LaunchPad.auto_load()
lpad.add_wf(flow_to_workflow(flow))
