import logging
from typing import Literal

from rdkit import Chem

from .backends.molgpka.model import MolGpKaModel
from .backends.pkalearn.model import PkaLearnModel
from .core.base import BasePKaModel
from .core.exceptions import InvalidBackendError
from .core.types import BackendType, MicrostateResult

logger = logging.getLogger(__name__)


def _generate_ordered_states(
    mol_no_hs: Chem.Mol,
    base_pka_dict: dict[int, float],
    acid_pka_dict: dict[int, float],
) -> list[Chem.Mol]:
    """Return protonation-state molecules in ascending pKa order.

    Produces one molecule per deprotonation step, from the fully protonated
    state (all basic sites charged) down to the fully deprotonated state.
    Backend-agnostic: works for any backend that returns the standard
    ``{base_pka, acid_pka, mol}`` dict.

    Args:
        mol_no_hs: heavy-atom-only RDKit molecule from ``predict_pka``.
        base_pka_dict: mapping of atom index → basic pKa.
        acid_pka_dict: mapping of atom index → acidic pKa.

    Returns:
        List of RDKit molecules, most protonated first. Always contains at
        least one entry even when both dicts are empty.
    """
    ionizable_sites: list[tuple[float, int, str]] = []
    for idx, pka in base_pka_dict.items():
        ionizable_sites.append((pka, idx, "base"))
    for idx, pka in acid_pka_dict.items():
        ionizable_sites.append((pka, idx, "acid"))
    ionizable_sites.sort(key=lambda x: x[0])

    if not ionizable_sites:
        return [mol_no_hs]

    unique_atoms = {idx for _, idx, _ in ionizable_sites}
    # Each basic pKa on an atom represents one proton that can be lost from
    # the fully protonated form, so the fully protonated charge equals the
    # count of basic sites on that atom.
    fully_protonated_charges = {
        atom_idx: sum(1 for _, i, t in ionizable_sites if i == atom_idx and t == "base")
        for atom_idx in unique_atoms
    }

    states: list[Chem.Mol] = []
    for k in range(len(ionizable_sites) + 1):
        rw = Chem.RWMol(mol_no_hs)
        try:
            Chem.Kekulize(rw, clearAromaticFlags=True)
        except Exception:
            pass

        for atom_idx in unique_atoms:
            atom = rw.GetAtomWithIdx(atom_idx)
            atom.SetNumExplicitHs(0)
            atom.SetNoImplicit(False)
            atom.SetFormalCharge(fully_protonated_charges[atom_idx])

        for i in range(k):
            _, idx, _ = ionizable_sites[i]
            atom = rw.GetAtomWithIdx(idx)
            atom.SetFormalCharge(atom.GetFormalCharge() - 1)

        try:
            rw.UpdatePropertyCache(strict=False)
            Chem.SanitizeMol(rw)
            states.append(rw.GetMol())
        except Exception:
            logger.warning(
                "_generate_ordered_states: sanitization failed for state k=%d; "
                "state skipped.",
                k,
                exc_info=True,
            )

    return states


class PKaPredictor(BasePKaModel):
    """Unified pKa predictor with pluggable backends.

    Wraps either the MolGpKa or pKaLearn GNN backend behind a single API.
    Call :meth:`dispose` explicitly when you are done to release GPU memory;
    do not rely on ``__del__`` for deterministic cleanup.

    Args:
        model: backend identifier — ``"molgpka"``, ``"pkalearn"``, or the
            corresponding :class:`BackendType` enum value. Defaults to
            ``BackendType.MOLGPKA``.
        device: PyTorch device string (e.g. ``"cpu"``, ``"cuda:0"``).
        allow_amphoteric: when ``True``, the pKaLearn backend runs an extra
            inference pass to identify atoms that are both acidic and basic
            (e.g. amino-acid side chains). Has no effect on MolGpKa.
    """

    def __init__(
        self,
        model: Literal["molgpka", "pkalearn"] | BackendType = BackendType.MOLGPKA,
        device: str = "cpu",
        allow_amphoteric: bool = False,
    ) -> None:
        try:
            self.model_name = BackendType(model)
        except ValueError:
            raise InvalidBackendError(
                f"Unknown backend: '{model}'. Choose from: {[b.value for b in BackendType]}"
            )
        super().__init__(device=device)
        self.allow_amphoteric = allow_amphoteric
        self.model = self._build_model()

    def _build_model(self) -> MolGpKaModel | PkaLearnModel:
        """Instantiate and return the backend model."""
        if self.model_name == BackendType.MOLGPKA:
            return MolGpKaModel(device=self.device)
        elif self.model_name == BackendType.PKALEARN:
            return PkaLearnModel(
                device=self.device,
                allow_amphoteric=self.allow_amphoteric,
            )
        # Unreachable: BackendType(model) above already rejects unknown values,
        # but kept for exhaustiveness.
        raise InvalidBackendError(f"Unhandled backend: {self.model_name}")  # pragma: no cover

    def dispose(self) -> None:
        """Release backend resources (e.g. GPU memory).

        Call this explicitly when you no longer need the predictor. Do not
        rely on garbage collection for timely GPU cleanup.
        """
        if hasattr(self, "model") and hasattr(self.model, "dispose"):
            self.model.dispose()

    def predict_pka(
        self,
        mol: Chem.Mol | list[Chem.Mol] | str | list[str],
    ) -> dict[str, object] | list[dict[str, object]]:
        """Predict pKa values for one or more molecules.

        Args:
            mol: a single molecule (RDKit Mol or SMILES string) or a list of
                either. Lists must be flat; nested lists are not supported.

        Returns:
            A single result dict when *mol* is not a list, or a list of result
            dicts when it is. Each dict has keys ``"acid_pka"``, ``"base_pka"``
            (both mapping atom index → float), and ``"mol"`` (the
            heavy-atom-only RDKit Mol used internally).
        """
        mols = self._to_mol(mol)
        results = [self.model.predict_pka(m) for m in mols]
        return results if isinstance(mol, list) else results[0]

    def predict_microstates(
        self,
        mol: Chem.Mol | list[Chem.Mol] | str | list[str],
        ph: float | None = 7.4,
        ph_range: tuple[float, float] | None = None,
        ph_step: float | None = None,
    ) -> MicrostateResult | dict[float, MicrostateResult] | list:
        """Predict microstate abundances at a given pH (or over a pH range).

        Args:
            mol: a single molecule (RDKit Mol or SMILES string) or a list of
                either. Lists must be flat; nested lists are not supported.
            ph: single pH value at which to evaluate abundances. Pass
                ``None`` together with *ph_range* to get a range sweep instead.
            ph_range: ``(ph_min, ph_max)`` tuple. Ignored when *ph* is not
                ``None``.
            ph_step: step size between pH values in the range sweep. Required
                when *ph_range* is given.

        Returns:
            When *mol* is not a list:
              - a :class:`~core.types.MicrostateResult` if *ph* is given, or
              - a ``dict[float, MicrostateResult]`` if *ph_range* is given.
            When *mol* is a list, a list of the above.
        """
        mols = self._to_mol(mol)
        results = [
            self.model.predict_microstates(m, ph=ph, ph_range=ph_range, ph_step=ph_step)
            for m in mols
        ]
        return results if isinstance(mol, list) else results[0]

    def protonation_ladder(
        self,
        mol: Chem.Mol | str,
        acid_first: bool = True,
    ) -> list[str]:
        """Return protonation states as canonical SMILES, ordered along the ladder.

        Derived from :meth:`predict_pka`; backend-agnostic.

        Args:
            mol: molecule or SMILES string.
            acid_first: if ``True`` (default), the list runs from the most
                protonated state (lowest pH / highest charge) to the most
                deprotonated state. Pass ``False`` to reverse the order.

        Returns:
            List of canonical SMILES strings, one per protonation state, in
            the requested order. Always contains at least one entry (the input
            molecule) even for non-ionisable structures.
        """
        input_mol = self._to_mol(mol)[0]
        pred = self.predict_pka(input_mol)
        mol_no_hs = pred["mol"]
        base_pka_dict = pred["base_pka"]
        acid_pka_dict = pred["acid_pka"]

        states = _generate_ordered_states(mol_no_hs, base_pka_dict, acid_pka_dict)

        # Deduplication preserves order; identical SMILES can arise when two
        # sites share a pKa value.
        seen: set[str] = set()
        smiles_list: list[str] = []
        for state_mol in states:
            smi = Chem.MolToSmiles(state_mol, isomericSmiles=True)
            if smi not in seen:
                seen.add(smi)
                smiles_list.append(smi)

        if not acid_first:
            smiles_list = list(reversed(smiles_list))

        return smiles_list
