"""Tests for the QupKake backend, using the same 14 molecules verified
against the original QupKake CLI (github.com/Shualdon/QupKake).

QupKake wraps an external package + xTB, so predictions are non-deterministic
(conformer embedding, MMFF/xTB optimization) and slow (~5-60s/molecule). Tests
therefore check chemically-plausible ranges and site presence rather than
exact values, and each molecule is only run once per test session.
"""
import os

import pytest
from rdkit import Chem

pytest.importorskip("qupkake")

from pick_a_pka.backends.qupkake.model import QupKakeModel
from constants import (
    ACETIC_ACID, ANILINE, GLYCINE, HISTIDINE, PYRIDINE, PHENOL, IMIDAZOLE,
    BENZENESULFONAMIDE, METHANESULFONIC_ACID, ETHANOLAMINE, PARA_AMINOPHENOL,
    GUANIDINE, TYROSINE, OSIMERTINIB, METHANE,
)

MOLECULES = {
    "acetic_acid": ACETIC_ACID,
    "aniline": ANILINE,
    "glycine": GLYCINE,
    "histidine": HISTIDINE,
    "pyridine": PYRIDINE,
    "phenol": PHENOL,
    "imidazole": IMIDAZOLE,
    "benzenesulfonamide": BENZENESULFONAMIDE,
    "methanesulfonic_acid": METHANESULFONIC_ACID,
    "ethanolamine": ETHANOLAMINE,
    "4_aminophenol": PARA_AMINOPHENOL,
    "guanidine": GUANIDINE,
    "tyrosine": TYROSINE,
    "osimertinib": OSIMERTINIB,
}

# Cross-checked against the original QupKake CLI (see conversation history).
EXPECTED_ACID_SITES = {
    "acetic_acid", "glycine", "histidine", "phenol", "imidazole",
    "benzenesulfonamide", "methanesulfonic_acid", "4_aminophenol",
    "tyrosine", "osimertinib",
}
EXPECTED_BASE_SITES = {
    "aniline", "glycine", "histidine", "pyridine", "guanidine",
    "ethanolamine", "tyrosine", "osimertinib",
}
AMPHOTERIC = {"glycine", "histidine", "tyrosine", "osimertinib"}


@pytest.fixture(scope="module")
def qupkake():
    return QupKakeModel()


@pytest.fixture(scope="module")
def results(qupkake):
    return {
        name: qupkake.predict_pka(Chem.MolFromSmiles(smi))
        for name, smi in MOLECULES.items()
    }


class TestSchema:
    @pytest.mark.parametrize("name", MOLECULES)
    def test_output_keys_present(self, results, name):
        assert {"acid_pka", "base_pka", "mol"} <= results[name].keys()

    @pytest.mark.parametrize("name", MOLECULES)
    def test_atom_indices_in_range(self, results, name):
        result = results[name]
        n_atoms = result["mol"].GetNumAtoms()
        for idx in list(result["acid_pka"]) + list(result["base_pka"]):
            assert 0 <= idx < n_atoms

    @pytest.mark.parametrize("name", MOLECULES)
    def test_pka_values_are_plausible(self, results, name):
        result = results[name]
        for pka in list(result["acid_pka"].values()) + list(result["base_pka"].values()):
            assert isinstance(pka, float)
            assert -5.0 <= pka <= 20.0


class TestKnownChemistry:
    @pytest.mark.parametrize("name", sorted(EXPECTED_ACID_SITES))
    def test_expected_acid_site_found(self, results, name):
        assert len(results[name]["acid_pka"]) >= 1

    @pytest.mark.parametrize("name", sorted(EXPECTED_BASE_SITES))
    def test_expected_base_site_found(self, results, name):
        assert len(results[name]["base_pka"]) >= 1

    @pytest.mark.parametrize("name", sorted(AMPHOTERIC))
    def test_amphoteric_molecule_has_both_site_types(self, results, name):
        result = results[name]
        assert len(result["acid_pka"]) >= 1
        assert len(result["base_pka"]) >= 1

    def test_methane_runs_without_crashing(self, qupkake):
        # Not asserting empty acid/base dicts: QupKake's own site-detection
        # model can flag a spurious acidic site even on saturated methane.
        result = qupkake.predict_pka(Chem.MolFromSmiles(METHANE))
        assert {"acid_pka", "base_pka", "mol"} <= result.keys()

    # Loose (+-2 pKa unit) tolerance around values observed from the original
    # QupKake CLI, to allow for xTB/conformer nondeterminism without masking
    # real regressions (e.g. a parsing bug returning a wildly different value).
    def test_acetic_acid_pka_near_reference(self, results):
        assert abs(min(results["acetic_acid"]["acid_pka"].values()) - 4.87) < 2.0

    def test_pyridine_pka_near_reference(self, results):
        assert abs(min(results["pyridine"]["base_pka"].values()) - 4.86) < 2.0

    def test_phenol_pka_near_reference(self, results):
        assert abs(min(results["phenol"]["acid_pka"].values()) - 9.41) < 2.0

    def test_aniline_base_pka_near_reference(self, results):
        assert abs(min(results["aniline"]["base_pka"].values()) - 4.31) < 2.0


class TestMicrostates:
    def test_predict_microstates_smoke(self, qupkake):
        result = qupkake.predict_microstates(Chem.MolFromSmiles(ACETIC_ACID), ph=7.4)
        assert "major_state" in result
        assert 0.0 <= result["major_abundance"] <= 100.0


class TestXtbOverride:
    def test_default_construction_leaves_xtbpath_untouched(self, monkeypatch):
        monkeypatch.delenv("XTBPATH", raising=False)
        QupKakeModel()
        assert "XTBPATH" not in os.environ

    def test_explicit_xtb_path_accepts_bundled_binary(self, monkeypatch):
        monkeypatch.delenv("XTBPATH", raising=False)
        import qupkake
        QupKakeModel(xtb_path=qupkake.XTB_LOCATION)
        assert os.environ["XTBPATH"] == qupkake.XTB_LOCATION

    def test_wrong_xtb_version_raises(self, tmp_path):
        from pick_a_pka.core.exceptions import XTBVersionError

        fake_xtb = tmp_path / "xtb"
        fake_xtb.write_text(
            "#!/bin/sh\necho ' * xtb version 6.7.1 (edcfbbe) compiled by x'\n"
        )
        fake_xtb.chmod(0o755)

        with pytest.raises(XTBVersionError):
            QupKakeModel(xtb_path=str(fake_xtb))
