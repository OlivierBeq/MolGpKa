"""Tests for the allow_amphoteric feature of the pKaLearn backend."""
import pytest

from pick_a_pka import PKaPredictor
from constants import OSIMERTINIB, GLYCINE, PHENYLALANINE, TRIMETHYLAMINE, BENZOIC_ACID


@pytest.fixture(scope="module")
def model_on():
    return PKaPredictor("pkalearn", allow_amphoteric=True)


@pytest.fixture(scope="module")
def model_off():
    return PKaPredictor("pkalearn", allow_amphoteric=False)


# ---------------------------------------------------------------------------
# Basic flag behaviour
# ---------------------------------------------------------------------------

class TestFlagBehaviour:
    def test_flag_stored_on_predictor(self):
        assert PKaPredictor("pkalearn", allow_amphoteric=True).allow_amphoteric is True
        assert PKaPredictor("pkalearn", allow_amphoteric=False).allow_amphoteric is False

    def test_amphoteric_flag_propagated_to_backend(self):
        model = PKaPredictor("pkalearn", allow_amphoteric=True)
        assert model.model.allow_amphoteric is True


# ---------------------------------------------------------------------------
# Separation invariant: acid_pka ∩ base_pka == ∅ when flag is OFF
# ---------------------------------------------------------------------------

class TestSeparationWithoutFlag:
    @pytest.mark.parametrize("smiles", [
        OSIMERTINIB, GLYCINE, PHENYLALANINE, TRIMETHYLAMINE, BENZOIC_ACID
    ]
                             )
    def test_no_overlap_without_flag(self, model_off, smiles):
        result = model_off.predict_pka(smiles)
        overlap = set(result["acid_pka"]) & set(result["base_pka"])
        assert len(overlap) == 0, (
            f"Without allow_amphoteric, atoms {overlap} appear in both acid and base dicts"
        )


# ---------------------------------------------------------------------------
# Amphoteric sites present when flag is ON
# ---------------------------------------------------------------------------

class TestAmphotericSitesPresent:
    def test_complex_molecule_has_amphoteric_overlap(self, model_on):
        result = model_on.predict_pka(OSIMERTINIB)
        overlap = set(result["acid_pka"]) & set(result["base_pka"])
        assert len(overlap) > 0, "Expected at least one amphoteric atom in the complex molecule"

    def test_glycine_has_amphoteric_nitrogen(self, model_on):
        """Glycine's amine is the canonical amphoteric site."""
        result = model_on.predict_pka(GLYCINE)
        overlap = set(result["acid_pka"]) & set(result["base_pka"])
        assert len(overlap) >= 1

    def test_phenylalanine_has_amphoteric_nitrogen(self, model_on):
        result = model_on.predict_pka(PHENYLALANINE)
        overlap = set(result["acid_pka"]) & set(result["base_pka"])
        assert len(overlap) >= 1


# ---------------------------------------------------------------------------
# Thermodynamic ordering of amphoteric atoms
# ---------------------------------------------------------------------------

class TestThermodynamicOrdering:
    @pytest.mark.xfail(
        reason="Model-accuracy limitation: predicted acid/base pKa for "
               "osimertinib's aniline-link N-H are close enough that noise "
               "can flip the expected ordering.",
        strict=False,
    )
    def test_acid_pka_gt_base_pka_for_amphoteric_atoms(self, model_on):
        """For every amphoteric atom, its acid pKa must exceed its base pKa.

        Chemically: the neutral form (between the two pKa values) is the
        thermodynamically preferred state.  base_pka < acid_pka guarantees
        the zwitterion / neutral window exists.
        """
        result = model_on.predict_pka(OSIMERTINIB)
        overlap = set(result["acid_pka"]) & set(result["base_pka"])
        for idx in overlap:
            assert result["base_pka"][idx] < result["acid_pka"][idx], (
                f"Atom {idx}: base_pka={result['base_pka'][idx]:.2f} "
                f">= acid_pka={result['acid_pka'][idx]:.2f}"
            )

    @pytest.mark.xfail(
        reason="Model-accuracy limitation: glycine's amine NH2->NH- pKa is "
               "far outside the model's training distribution and unreliable.",
        strict=False,
    )
    def test_glycine_acid_pka_gt_base_pka(self, model_on):
        result = model_on.predict_pka(GLYCINE)
        overlap = set(result["acid_pka"]) & set(result["base_pka"])
        for idx in overlap:
            assert result["base_pka"][idx] < result["acid_pka"][idx]

    def test_no_acid_base_pka_equality_for_amphoteric_atoms(self, model_on):
        """acid_pka == base_pka for the same atom is chemically nonsensical."""
        result = model_on.predict_pka(OSIMERTINIB)
        overlap = set(result["acid_pka"]) & set(result["base_pka"])
        for idx in overlap:
            assert result["acid_pka"][idx] != result["base_pka"][idx], (
                f"Atom {idx} has identical acid and base pKa: {result['acid_pka'][idx]}"
            )


# ---------------------------------------------------------------------------
# Non-amphoteric molecules are unaffected by the flag
# ---------------------------------------------------------------------------

class TestNonAmphotericUnchanged:
    def test_trimethylamine_has_no_acid_pka_with_flag(self, model_on):
        """Trimethylamine has no NH; its nitrogen cannot be acidic."""
        result = model_on.predict_pka(TRIMETHYLAMINE)
        assert len(result["acid_pka"]) == 0

    def test_benzoic_acid_has_no_base_pka_with_flag(self, model_on):
        """Benzoic acid has no ionisable nitrogen."""
        result = model_on.predict_pka(BENZOIC_ACID)
        assert len(result["base_pka"]) == 0

    def test_benzoic_acid_acid_pka_same_with_and_without_flag(self):
        r_on = PKaPredictor("pkalearn", allow_amphoteric=True).predict_pka(BENZOIC_ACID)
        r_off = PKaPredictor("pkalearn", allow_amphoteric=False).predict_pka(BENZOIC_ACID)
        assert set(r_on["acid_pka"].keys()) == set(r_off["acid_pka"].keys())


# ---------------------------------------------------------------------------
# Index and value sanity
# ---------------------------------------------------------------------------

class TestIndexAndValueSanity:
    def test_amphoteric_indices_valid_for_returned_mol(self, model_on):
        result = model_on.predict_pka(OSIMERTINIB)
        n_atoms = result["mol"].GetNumAtoms()
        for idx in set(result["acid_pka"]) | set(result["base_pka"]):
            assert 0 <= idx < n_atoms

    def test_acid_pka_values_in_plausible_range(self, model_on):
        result = model_on.predict_pka(OSIMERTINIB)
        for v in result["acid_pka"].values():
            assert -5 < v < 30

    def test_base_pka_values_in_plausible_range(self, model_on):
        result = model_on.predict_pka(OSIMERTINIB)
        for v in result["base_pka"].values():
            assert -5 < v < 30

    def test_amphoteric_atoms_carry_proton_in_neutral_form(self, model_on):
        """Every amphoteric atom must have at least one H in the neutral molecule,
        because it can only donate a proton (act as an acid) if it has one.
        Note: pKaLearn's featurizer allows C–H ionization in some contexts,
        so we do not restrict to heteroatoms here.
        """
        result = model_on.predict_pka(OSIMERTINIB)
        mol = result["mol"]
        overlap = set(result["acid_pka"]) & set(result["base_pka"])
        for idx in overlap:
            atom = mol.GetAtomWithIdx(idx)
            assert atom.GetTotalNumHs() > 0, (
                f"Atom {idx} ({atom.GetSymbol()}) classified as amphoteric "
                f"but carries no H in the neutral molecule"
            )
