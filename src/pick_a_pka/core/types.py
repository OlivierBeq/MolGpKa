from enum import Enum
from typing import TypedDict

from rdkit import Chem


class BackendType(str, Enum):
    MOLGPKA = "molgpka"
    PKALEARN = "pkalearn"
    QUPKAKE = "qupkake"


class LadderStep(TypedDict):
    smiles: str
    mol_a: Chem.Mol
    mol_b: Chem.Mol
    center: int
    pka: float


class StateDistribution(TypedDict):
    smiles: str
    mol: Chem.Mol
    abundance: float


class MicrostateResult(TypedDict):
    major_state: Chem.Mol
    major_abundance: float
    distribution: list[StateDistribution]
