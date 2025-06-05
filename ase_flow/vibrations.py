import os
from ase.vibrations import Vibrations
from ase.thermochemistry import HarmonicThermo, IdealGasThermo


def calc_free_energy(
    atoms,
    temperature=298.15,
    pressure=101325,
    mode="harmonic",
    geometry="nonlinear",
    symmetrynumber=2,
    spin=0,
):
    vib = Vibrations(atoms)
    vib.run()
    vib_freq = [
        freq * 1.23981e-4 for freq in vib.get_frequencies() if abs(freq.imag) < 1e-6
    ]  # cm^-1 to eV
    zpe = vib.get_zero_point_energy()
    if mode == "harmonic":
        thermo = HarmonicThermo(vib_freq, atoms.get_potential_energy())
        free_energy = thermo.get_helmholtz_energy(temperature)
    elif mode == "ideal":
        thermo = IdealGasThermo(
            vib_energies=vib_freq,
            potentialenergy=atoms.get_potential_energy(),
            atoms=atoms,
            geometry=geometry,
            symmetrynumber=symmetrynumber,
            spin=spin,
        )
        free_energy = thermo.get_gibbs_energy(temperature, pressure)

    vib.clean()
    return free_energy
