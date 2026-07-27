"""Tests comparing and contrasting the two backends."""
import pytest
from rdkit import Chem

from pick_a_pka import PKaPredictor
from constants import ACETIC_ACID, ANILINE, GLYCINE, BUTANE


@pytest.fixture(scope="module")
def molgpka():
    return PKaPredictor("molgpka")


@pytest.fixture(scope="module")
def pkalearn():
    return PKaPredictor("pkalearn")


# ---------------------------------------------------------------------------
# Both backends return the same output schema
# ---------------------------------------------------------------------------

class TestSharedContract:
    @pytest.mark.parametrize("smiles", [ACETIC_ACID, ANILINE, GLYCINE])
    @pytest.mark.parametrize("backend", ["molgpka", "pkalearn"])
    def test_output_keys_present(self, smiles, backend):
        result = PKaPredictor(backend).predict_pka(smiles)
        assert {"acid_pka", "base_pka", "mol"} <= result.keys()

    @pytest.mark.parametrize("backend", ["molgpka", "pkalearn"])
    def test_non_ionisable_returns_empty_dicts(self, backend):
        result = PKaPredictor(backend).predict_pka(BUTANE)
        assert result["acid_pka"] == {}
        assert result["base_pka"] == {}

    @pytest.mark.parametrize("backend", ["molgpka", "pkalearn"])
    def test_acetic_acid_has_acid_site(self, backend):
        result = PKaPredictor(backend).predict_pka(ACETIC_ACID)
        assert len(result["acid_pka"]) >= 1

    @pytest.mark.parametrize("backend", ["molgpka", "pkalearn"])
    def test_aniline_has_basic_site(self, backend):
        result = PKaPredictor(backend).predict_pka(ANILINE)
        assert len(result["base_pka"]) >= 1


# ---------------------------------------------------------------------------
# Chemical agreement between backends
# ---------------------------------------------------------------------------

class TestChemicalAgreement:
    """
    Both backends should agree on the *type* (acid vs base) of ionisable sites
    for clear-cut molecules, and pKa values should be in the same ballpark.
    """

    def test_acetic_acid_classified_as_acid_by_both(self, molgpka, pkalearn):
        r_m = molgpka.predict_pka(ACETIC_ACID)
        r_p = pkalearn.predict_pka(ACETIC_ACID)
        assert len(r_m["acid_pka"]) >= 1
        assert len(r_p["acid_pka"]) >= 1

    def test_acetic_acid_pka_within_3_units(self, molgpka, pkalearn):
        r_m = molgpka.predict_pka(ACETIC_ACID)
        r_p = pkalearn.predict_pka(ACETIC_ACID)
        pka_m = min(r_m["acid_pka"].values())
        pka_p = min(r_p["acid_pka"].values())
        assert abs(pka_m - pka_p) < 3.0

    def test_aniline_classified_as_base_by_both(self, molgpka, pkalearn):
        r_m = molgpka.predict_pka(ANILINE)
        r_p = pkalearn.predict_pka(ANILINE)
        assert len(r_m["base_pka"]) >= 1
        assert len(r_p["base_pka"]) >= 1

    def test_glycine_has_acid_and_base_both_backends(self, molgpka, pkalearn):
        for result in [molgpka.predict_pka(GLYCINE), pkalearn.predict_pka(GLYCINE)]:
            assert len(result["acid_pka"]) >= 1
            assert len(result["base_pka"]) >= 1


# ---------------------------------------------------------------------------
# Backend-specific behaviour
# ---------------------------------------------------------------------------

class TestMolGpKaSpecific:
    def test_molgpka_no_amphoteric_without_flag(self, molgpka):
        """MolGpKa does not have an amphoteric mode; its two dicts must be disjoint."""
        result = molgpka.predict_pka(GLYCINE)
        overlap = set(result["acid_pka"]) & set(result["base_pka"])
        assert len(overlap) == 0

    def test_molgpka_independent_acid_base_models(self, molgpka):
        """MolGpKa runs two independent GNNs; both can fire on the same molecule."""
        result = molgpka.predict_pka(GLYCINE)
        assert len(result["acid_pka"]) >= 1
        assert len(result["base_pka"]) >= 1


class TestPkaLearnSpecific:
    def test_pkalearn_ladder_is_monotone_without_amphoteric(self, pkalearn):
        """Without allow_amphoteric, the ladder should produce strictly disjoint sets."""
        result = pkalearn.predict_pka(GLYCINE)
        overlap = set(result["acid_pka"]) & set(result["base_pka"])
        assert len(overlap) == 0

    def test_pkalearn_predict_returns_tuple(self):
        """The internal predict() method returns (ladder, starting_mol)."""
        model = PKaPredictor("pkalearn").model
        ladder, starting_mol = model.predict("CC(=O)O")
        assert isinstance(ladder, list)
        assert isinstance(starting_mol, Chem.Mol)

    def test_pkalearn_ladder_steps_have_required_keys(self):
        model = PKaPredictor("pkalearn").model
        ladder, _ = model.predict("CC(=O)O")
        for step in ladder:
            assert "smiles" in step
            assert "center" in step
            assert "pka" in step

    def test_pkalearn_ladder_pka_values_decrease_monotonically(self):
        """Ladder walks from the most basic site (highest pKa) down to the most acidic."""
        model = PKaPredictor("pkalearn").model
        ladder, _ = model.predict(GLYCINE)
        pkas = [step["pka"] for step in ladder]
        assert pkas == sorted(pkas, reverse=True)
