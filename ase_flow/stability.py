from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

from jobflow import Flow, job
from pymatgen.core import Composition
from pymatgen.io.vasp import Poscar

from ase_flow.ase_relax import relax_ase


def _result_payload(relax_result: dict) -> dict:
    return relax_result["output"]["output"]


def _energy_key(relax_result: dict, use_free_energy: bool) -> str:
    payload = _result_payload(relax_result)
    if use_free_energy and "free_energy" in payload:
        return "free_energy"
    return "energy"


def _reduced_formula_factor(relax_result: dict) -> Tuple[Composition, float]:
    composition = Composition(relax_result["composition"])
    reduced_composition, factor = composition.get_reduced_composition_and_factor()
    return reduced_composition, float(factor)


def _infer_mixing_elements(
    endmember_a_result: dict,
    endmember_b_result: dict,
) -> Tuple[str, str]:
    comp_a = Composition(endmember_a_result["composition"])
    comp_b = Composition(endmember_b_result["composition"])
    elements_a = {str(el) for el in comp_a.elements}
    elements_b = {str(el) for el in comp_b.elements}
    only_a = sorted(elements_a - elements_b)
    only_b = sorted(elements_b - elements_a)
    if len(only_a) != 1 or len(only_b) != 1:
        raise ValueError(
            "Could not infer the two mixing elements from the endpoint compositions. "
            "Please pass mixing_elements explicitly."
        )
    return only_a[0], only_b[0]


def _infer_mixing_elements_from_references(
    reference_results: Iterable[dict],
) -> Tuple[str, ...]:
    element_sets = [
        {str(el) for el in Composition(result["composition"]).elements}
        for result in reference_results
    ]
    if len(element_sets) < 2:
        raise ValueError("Need at least two reference structures.")
    common = set.intersection(*element_sets)
    varying = sorted(set.union(*element_sets) - common)
    if len(varying) < 2:
        raise ValueError(
            "Could not infer a mixing subspace from the reference compositions. "
            "Please pass mixing_elements explicitly."
        )
    return tuple(varying)


def _scaffold_signature(
    reduced_composition: Composition,
    mixing_elements: Iterable[str],
) -> dict:
    mixing_set = set(mixing_elements)
    return {
        el: float(amount)
        for el, amount in reduced_composition.get_el_amt_dict().items()
        if el not in mixing_set and amount > 0.0
    }


def _mixing_amounts(
    reduced_composition: Composition,
    mixing_elements: Iterable[str],
) -> dict:
    comp_dict = reduced_composition.get_el_amt_dict()
    return {el: float(comp_dict.get(el, 0.0)) for el in mixing_elements}


def _mixing_fractions(
    reduced_composition: Composition,
    mixing_elements: Iterable[str],
) -> dict:
    amounts = _mixing_amounts(reduced_composition, mixing_elements)
    total = float(sum(amounts.values()))
    if total <= 0.0:
        raise ValueError(
            f"Reduced composition {reduced_composition.reduced_formula} does not contain "
            f"mixing elements {tuple(mixing_elements)}."
        )
    return {el: amounts[el] / total for el in mixing_elements}


def _reference_pure_element(
    reduced_composition: Composition,
    mixing_elements: Iterable[str],
) -> str:
    fractions = _mixing_fractions(reduced_composition, mixing_elements)
    nonzero = [el for el, value in fractions.items() if value > 1e-12]
    if len(nonzero) != 1:
        raise ValueError(
            "Reference entries must be pure end members on the mixing sublattice. "
            f"Got {reduced_composition.reduced_formula} with fractions {fractions}."
        )
    return nonzero[0]


@job
def load_structure(filename: str) -> dict:
    return Poscar.from_file(filename).structure


@job
def analyze_endpoint_stability(
    endmember_a_result: dict,
    endmember_b_result: dict,
    alloy_result: dict,
    mixing_elements: Iterable[str] | None = None,
    use_free_energy: bool = True,
) -> dict:
    if mixing_elements is None:
        element_a, element_b = _infer_mixing_elements(
            endmember_a_result,
            endmember_b_result,
        )
    else:
        mixing_elements = tuple(mixing_elements)
        if len(mixing_elements) != 2:
            raise ValueError("mixing_elements must contain exactly two element symbols.")
        element_a, element_b = str(mixing_elements[0]), str(mixing_elements[1])

    key_a = _energy_key(endmember_a_result, use_free_energy=use_free_energy)
    key_b = _energy_key(endmember_b_result, use_free_energy=use_free_energy)
    key_alloy = _energy_key(alloy_result, use_free_energy=use_free_energy)

    reduced_a, factor_a = _reduced_formula_factor(endmember_a_result)
    reduced_b, factor_b = _reduced_formula_factor(endmember_b_result)
    reduced_alloy, factor_alloy = _reduced_formula_factor(alloy_result)

    energy_a_pf = float(_result_payload(endmember_a_result)[key_a]) / factor_a
    energy_b_pf = float(_result_payload(endmember_b_result)[key_b]) / factor_b
    energy_alloy_pf = float(_result_payload(alloy_result)[key_alloy]) / factor_alloy

    reduced_alloy_mixing_total = float(
        reduced_alloy.get_el_amt_dict().get(element_a, 0.0)
        + reduced_alloy.get_el_amt_dict().get(element_b, 0.0)
    )
    if reduced_alloy_mixing_total <= 0.0:
        raise ValueError(
            f"Alloy reduced composition does not contain mixing elements {element_a}, {element_b}."
        )
    x_b = float(reduced_alloy.get_el_amt_dict().get(element_b, 0.0)) / reduced_alloy_mixing_total
    x_a = 1.0 - x_b

    reference_pf = x_a * energy_a_pf + x_b * energy_b_pf
    delta_pf = energy_alloy_pf - reference_pf

    reduced_alloy_atoms = float(sum(reduced_alloy.get_el_amt_dict().values()))
    delta_per_atom = delta_pf / reduced_alloy_atoms if reduced_alloy_atoms > 0.0 else float("nan")

    return {
        "mixing_elements": [element_a, element_b],
        "energy_key_endmember_a": key_a,
        "energy_key_endmember_b": key_b,
        "energy_key_alloy": key_alloy,
        "endpoint_a": {
            "formula": endmember_a_result["formula_pretty"],
            "reduced_formula": reduced_a.reduced_formula,
            "formula_units": factor_a,
            "energy_per_reduced_formula": energy_a_pf,
        },
        "endpoint_b": {
            "formula": endmember_b_result["formula_pretty"],
            "reduced_formula": reduced_b.reduced_formula,
            "formula_units": factor_b,
            "energy_per_reduced_formula": energy_b_pf,
        },
        "alloy": {
            "formula": alloy_result["formula_pretty"],
            "reduced_formula": reduced_alloy.reduced_formula,
            "formula_units": factor_alloy,
            "energy_per_reduced_formula": energy_alloy_pf,
            "fraction_" + element_a: x_a,
            "fraction_" + element_b: x_b,
        },
        "reference_energy_per_reduced_formula": reference_pf,
        "delta_mix_per_reduced_formula": delta_pf,
        "delta_mix_per_atom": delta_per_atom,
        "stable_against_endmembers": bool(delta_pf <= 0.0),
    }


@job
def analyze_reference_stability(
    reference_results: list[dict],
    alloy_results: list[dict],
    mixing_elements: Iterable[str] | None = None,
    use_free_energy: bool = True,
) -> dict:
    if len(reference_results) < 2:
        raise ValueError("Need at least two reference entries.")
    if len(alloy_results) < 1:
        raise ValueError("Need at least one alloy entry.")

    if mixing_elements is None:
        mixing_elements = _infer_mixing_elements_from_references(reference_results)
    else:
        mixing_elements = tuple(str(el) for el in mixing_elements)
        if len(mixing_elements) < 2:
            raise ValueError("mixing_elements must contain at least two symbols.")

    reference_entries = []
    reference_energy_map = {}
    scaffold_signature = None
    for result in reference_results:
        key = _energy_key(result, use_free_energy=use_free_energy)
        reduced_comp, factor = _reduced_formula_factor(result)
        signature = _scaffold_signature(reduced_comp, mixing_elements)
        if scaffold_signature is None:
            scaffold_signature = signature
        elif signature != scaffold_signature:
            raise ValueError(
                "Reference entries are not on the same fixed scaffold. "
                f"Expected scaffold {scaffold_signature}, got {signature}."
            )

        pure_element = _reference_pure_element(reduced_comp, mixing_elements)
        energy_per_reduced_formula = float(_result_payload(result)[key]) / factor
        if pure_element in reference_energy_map:
            raise ValueError(f"Duplicate reference entry for mixing element {pure_element}.")
        reference_energy_map[pure_element] = energy_per_reduced_formula

        entry = {
            "formula": result["formula_pretty"],
            "reduced_formula": reduced_comp.reduced_formula,
            "formula_units": factor,
            "pure_mixing_element": pure_element,
            "energy_key": key,
            "energy_per_reduced_formula": energy_per_reduced_formula,
        }
        entry.update(
            {f"fraction_{el}": value for el, value in _mixing_fractions(reduced_comp, mixing_elements).items()}
        )
        reference_entries.append(entry)

    missing = [el for el in mixing_elements if el not in reference_energy_map]
    if missing:
        raise ValueError(
            f"Missing pure reference entries for mixing elements: {missing}."
        )

    alloy_entries = []
    for result in alloy_results:
        key = _energy_key(result, use_free_energy=use_free_energy)
        reduced_comp, factor = _reduced_formula_factor(result)
        signature = _scaffold_signature(reduced_comp, mixing_elements)
        if signature != scaffold_signature:
            raise ValueError(
                "Alloy entry is not on the same fixed scaffold as the references. "
                f"Expected scaffold {scaffold_signature}, got {signature}."
            )

        energy_per_reduced_formula = float(_result_payload(result)[key]) / factor
        fractions = _mixing_fractions(reduced_comp, mixing_elements)
        reference_energy = sum(
            fractions[el] * reference_energy_map[el] for el in mixing_elements
        )
        delta_pf = energy_per_reduced_formula - reference_energy
        reduced_atoms = float(sum(reduced_comp.get_el_amt_dict().values()))
        alloy_entry = {
            "formula": result["formula_pretty"],
            "reduced_formula": reduced_comp.reduced_formula,
            "formula_units": factor,
            "energy_key": key,
            "energy_per_reduced_formula": energy_per_reduced_formula,
            "reference_energy_per_reduced_formula": reference_energy,
            "delta_mix_per_reduced_formula": delta_pf,
            "delta_mix_per_atom": (
                delta_pf / reduced_atoms if reduced_atoms > 0.0 else float("nan")
            ),
            "stable_against_references": bool(delta_pf <= 0.0),
        }
        alloy_entry.update({f"fraction_{el}": value for el, value in fractions.items()})
        alloy_entries.append(alloy_entry)

    return {
        "mixing_elements": list(mixing_elements),
        "reference_entries": reference_entries,
        "alloys": alloy_entries,
    }


@dataclass
class EndpointStabilityWorkflow:
    endmember_a_structure: object = None
    endmember_b_structure: object = None
    alloy_structure: object = None
    endmember_a_filename: str = "POSCAR.endmember_a"
    endmember_b_filename: str = "POSCAR.endmember_b"
    alloy_filename: str = "POSCAR.alloy"
    mixing_elements: Iterable[str] | None = None
    forcefield: str = "mace"
    model: str | None = None
    free_energy: bool = True
    mode_fe: str = "harmonic"
    temperature: float = 298.15
    pressure: float = 101325
    geometry: str = "linear"
    symmetrynumber: int = 2
    spin: int = 0
    alias_endmember_a: str = "endmember_a"
    alias_endmember_b: str = "endmember_b"
    alias_alloy: str = "alloy"

    def _load_or_use(self, structure: object, filename: str):
        if structure is not None:
            return None, structure
        load_job = load_structure(filename=filename)
        return load_job, load_job.output

    def build(self):
        jobs = []

        load_a, structure_a = self._load_or_use(
            self.endmember_a_structure,
            self.endmember_a_filename,
        )
        load_b, structure_b = self._load_or_use(
            self.endmember_b_structure,
            self.endmember_b_filename,
        )
        load_alloy, structure_alloy = self._load_or_use(
            self.alloy_structure,
            self.alloy_filename,
        )

        for load_job in (load_a, load_b, load_alloy):
            if load_job is not None:
                jobs.append(load_job)

        relax_a = relax_ase(
            atoms=structure_a,
            forcefield=self.forcefield,
            model=self.model,
            alias=self.alias_endmember_a,
            free_energy=self.free_energy,
            mode_fe=self.mode_fe,
            temperature=self.temperature,
            pressure=self.pressure,
            geometry=self.geometry,
            symmetrynumber=self.symmetrynumber,
            spin=self.spin,
        )
        relax_a.name = "relax endmember a"
        jobs.append(relax_a)

        relax_b = relax_ase(
            atoms=structure_b,
            forcefield=self.forcefield,
            model=self.model,
            alias=self.alias_endmember_b,
            free_energy=self.free_energy,
            mode_fe=self.mode_fe,
            temperature=self.temperature,
            pressure=self.pressure,
            geometry=self.geometry,
            symmetrynumber=self.symmetrynumber,
            spin=self.spin,
        )
        relax_b.name = "relax endmember b"
        jobs.append(relax_b)

        relax_alloy = relax_ase(
            atoms=structure_alloy,
            forcefield=self.forcefield,
            model=self.model,
            alias=self.alias_alloy,
            free_energy=self.free_energy,
            mode_fe=self.mode_fe,
            temperature=self.temperature,
            pressure=self.pressure,
            geometry=self.geometry,
            symmetrynumber=self.symmetrynumber,
            spin=self.spin,
        )
        relax_alloy.name = "relax alloy"
        jobs.append(relax_alloy)

        analyze = analyze_endpoint_stability(
            relax_a.output,
            relax_b.output,
            relax_alloy.output,
            mixing_elements=self.mixing_elements,
            use_free_energy=self.free_energy,
        )
        analyze.name = "analyze endpoint stability"
        jobs.append(analyze)

        return Flow(jobs), analyze


@dataclass
class ReferenceStabilityWorkflow:
    reference_structures: Iterable[object] | None = None
    alloy_structures: Iterable[object] | None = None
    reference_filenames: Iterable[str] | None = None
    alloy_filenames: Iterable[str] | None = None
    mixing_elements: Iterable[str] | None = None
    forcefield: str = "mace"
    model: str | None = None
    free_energy: bool = True
    mode_fe: str = "harmonic"
    temperature: float = 298.15
    pressure: float = 101325
    geometry: str = "linear"
    symmetrynumber: int = 2
    spin: int = 0
    reference_aliases: Iterable[str] | None = None
    alloy_aliases: Iterable[str] | None = None

    def _normalize_inputs(
        self,
        structures: Iterable[object] | None,
        filenames: Iterable[str] | None,
        prefix: str,
    ):
        structures = [] if structures is None else list(structures)
        filenames = [] if filenames is None else list(filenames)
        if structures and filenames:
            raise ValueError(f"Provide either {prefix}_structures or {prefix}_filenames, not both.")
        if not structures and not filenames:
            raise ValueError(f"Need at least one {prefix} structure.")
        if structures:
            return list(structures), [f"{prefix}_{idx}" for idx in range(len(structures))]
        return list(filenames), [f"{prefix}_{idx}" for idx in range(len(filenames))]

    def _load_or_use_many(
        self,
        structures: list[object],
        names: list[str],
        from_files: bool,
    ):
        jobs = []
        outputs = []
        for structure, name in zip(structures, names):
            if from_files:
                load_job = load_structure(filename=str(structure))
                jobs.append(load_job)
                outputs.append(load_job.output)
            else:
                outputs.append(structure)
        return jobs, outputs

    def build(self):
        reference_items, reference_names = self._normalize_inputs(
            self.reference_structures,
            self.reference_filenames,
            "reference",
        )
        alloy_items, alloy_names = self._normalize_inputs(
            self.alloy_structures,
            self.alloy_filenames,
            "alloy",
        )

        reference_aliases = (
            list(self.reference_aliases)
            if self.reference_aliases is not None
            else reference_names
        )
        alloy_aliases = (
            list(self.alloy_aliases)
            if self.alloy_aliases is not None
            else alloy_names
        )
        if len(reference_aliases) != len(reference_items):
            raise ValueError("reference_aliases must match the number of reference inputs.")
        if len(alloy_aliases) != len(alloy_items):
            raise ValueError("alloy_aliases must match the number of alloy inputs.")

        jobs = []
        ref_load_jobs, ref_outputs = self._load_or_use_many(
            reference_items,
            reference_names,
            from_files=self.reference_filenames is not None,
        )
        alloy_load_jobs, alloy_outputs = self._load_or_use_many(
            alloy_items,
            alloy_names,
            from_files=self.alloy_filenames is not None,
        )
        jobs.extend(ref_load_jobs)
        jobs.extend(alloy_load_jobs)

        ref_relax_jobs = []
        for structure, alias in zip(ref_outputs, reference_aliases):
            relax_job = relax_ase(
                atoms=structure,
                forcefield=self.forcefield,
                model=self.model,
                alias=alias,
                free_energy=self.free_energy,
                mode_fe=self.mode_fe,
                temperature=self.temperature,
                pressure=self.pressure,
                geometry=self.geometry,
                symmetrynumber=self.symmetrynumber,
                spin=self.spin,
            )
            relax_job.name = f"relax {alias}"
            ref_relax_jobs.append(relax_job)
            jobs.append(relax_job)

        alloy_relax_jobs = []
        for structure, alias in zip(alloy_outputs, alloy_aliases):
            relax_job = relax_ase(
                atoms=structure,
                forcefield=self.forcefield,
                model=self.model,
                alias=alias,
                free_energy=self.free_energy,
                mode_fe=self.mode_fe,
                temperature=self.temperature,
                pressure=self.pressure,
                geometry=self.geometry,
                symmetrynumber=self.symmetrynumber,
                spin=self.spin,
            )
            relax_job.name = f"relax {alias}"
            alloy_relax_jobs.append(relax_job)
            jobs.append(relax_job)

        analyze = analyze_reference_stability(
            [job.output for job in ref_relax_jobs],
            [job.output for job in alloy_relax_jobs],
            mixing_elements=self.mixing_elements,
            use_free_energy=self.free_energy,
        )
        analyze.name = "analyze reference stability"
        jobs.append(analyze)

        return Flow(jobs), analyze
