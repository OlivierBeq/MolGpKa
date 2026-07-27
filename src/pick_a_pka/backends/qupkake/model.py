import io
import os
import re
import tempfile
from contextlib import redirect_stdout

from rdkit import Chem
from rdkit.Chem import PandasTools
from rdkit.Chem.MolStandardize import rdMolStandardize

from .utils import verify_xtb_working
from ...core.base import BasePKaModel

_TENSOR_REPR = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


class QupKakeModel(BasePKaModel):
    def __init__(self, device="cpu", tautomerize=False, multiprocessing=False, xtb_path=None):
        # Note: QupKake forces CPU inference under the hood via pl.Trainer(accelerator="cpu")
        super().__init__(device=device)
        self.tautomerize = tautomerize
        self.multiprocessing = multiprocessing

        # Leave XTBPATH untouched unless asked to override: QupKake's bundled
        # 6.4.1 works out of the box, but a newer xTB on PATH can break its parser.
        if xtb_path is not None:
            os.environ["XTBPATH"] = verify_xtb_working(xtb_path=xtb_path)

        try:
            import qupkake
        except ImportError:
            raise ImportError(
                "QupKake is not installed. Please install it from: "
                "https://github.com/Shualdon/QupKake"
            )

    def predict_pka(self, mol: Chem.Mol, uncharged: bool = True) -> dict:
        from qupkake.predict import run_prediction_pipeline

        mol_copy = Chem.Mol(mol)
        if uncharged:
            un = rdMolStandardize.Uncharger()
            mol_copy = un.uncharge(mol_copy)
        if not mol_copy.HasProp("_Name") or not mol_copy.GetProp("_Name"):
            mol_copy.SetProp("_Name", "mol_0")

        acid_pka = {}
        base_pka = {}
        mol_with_hs = None

        # Create an ephemeral workspace for QupKake's file-based datasets
        with tempfile.TemporaryDirectory() as tmpdir:
            for d in ["raw", "processed", "logs", "output"]:
                os.makedirs(os.path.join(tmpdir, d), exist_ok=True)

            input_sdf = os.path.join(tmpdir, "raw", "input.sdf")
            writer = Chem.SDWriter(input_sdf)
            writer.write(mol_copy)
            writer.close()

            # QupKake uses extensive print() statements. Suppress them cleanly.
            f = io.StringIO()
            with redirect_stdout(f):
                try:
                    run_prediction_pipeline(
                        root=tmpdir,
                        filename="input.sdf",
                        tautomerize=self.tautomerize,
                        # Must be "ID": run_prediction_pipeline reloads its own
                        # intermediate SDF without idName, defaulting to "ID";
                        # anything else crashes its final write with a KeyError.
                        name_col="ID",
                        mol_col="ROMol",
                        mp=self.multiprocessing,
                        output="results.sdf",
                    )
                except Exception:
                    # QupKake throws/fails gracefully if no protonation sites are found
                    pass

            results_sdf = os.path.join(tmpdir, "output", "results.sdf")
            if os.path.exists(results_sdf):
                df = PandasTools.LoadSDF(
                    results_sdf, idName="ID", embedProps=True, removeHs=False
                )

                if df is not None and not df.empty:
                    mol_with_hs = df.iloc[0]["ROMol"]
                    for _, row in df.iterrows():
                        idx = int(row["idx"])
                        # A single detected site collapses upstream's tensor to
                        # 0-d, writing "tensor(4.8578)" instead of a plain float.
                        pka_match = _TENSOR_REPR.search(str(row["pka"]))
                        if pka_match is None:
                            continue
                        pka_val = float(pka_match.group())
                        pka_type = row["pka_type"]

                        if pka_type == "acidic":
                            acid_pka[idx] = pka_val
                        elif pka_type == "basic":
                            base_pka[idx] = pka_val

        # If QupKake failed or found no sites, return the original mol
        if mol_with_hs is None:
            return {
                "base_pka": base_pka,
                "acid_pka": acid_pka,
                "mol": Chem.RemoveHs(mol_copy),
            }

        # Map indices back to the Hydrogen-depleted molecule
        mol_no_hs, mapped_base, mapped_acid = self._remap_pka_without_hs(
            mol_with_hs, base_pka, acid_pka
        )

        return {"base_pka": mapped_base, "acid_pka": mapped_acid, "mol": mol_no_hs}

    def _remap_pka_without_hs(self, mol_with_hs, base_pka_dict, acid_pka_dict):
        """
        Remap pKa atom indices in a molecule with explicit hydrogens
        to the molecule without hydrogens.
        """
        for atom in mol_with_hs.GetAtoms():
            atom.SetIntProp("OrigIdx", atom.GetIdx())

        h_to_heavy = {}
        for atom in mol_with_hs.GetAtoms():
            if atom.GetAtomicNum() == 1:
                neighbors = atom.GetNeighbors()
                if neighbors:
                    h_to_heavy[atom.GetIdx()] = neighbors[0].GetIdx()

        mol_no_hs = Chem.RemoveHs(mol_with_hs)

        orig_to_new_idx = {}
        for atom in mol_no_hs.GetAtoms():
            if atom.HasProp("OrigIdx"):
                orig_to_new_idx[atom.GetIntProp("OrigIdx")] = atom.GetIdx()

        new_acid_pka_dict = {}
        new_base_pka_dict = {}

        for old_idx, pka_val in acid_pka_dict.items():
            if old_idx in orig_to_new_idx:
                new_acid_pka_dict[orig_to_new_idx[old_idx]] = pka_val
            elif old_idx in h_to_heavy and h_to_heavy[old_idx] in orig_to_new_idx:
                new_acid_pka_dict[orig_to_new_idx[h_to_heavy[old_idx]]] = pka_val

        for old_idx, pka_val in base_pka_dict.items():
            if old_idx in orig_to_new_idx:
                new_base_pka_dict[orig_to_new_idx[old_idx]] = pka_val
            elif old_idx in h_to_heavy and h_to_heavy[old_idx] in orig_to_new_idx:
                new_base_pka_dict[orig_to_new_idx[h_to_heavy[old_idx]]] = pka_val

        return mol_no_hs, new_base_pka_dict, new_acid_pka_dict

    def predict_microstates(self, mol, ph=7.4, ph_range=None, ph_step=None):
        # QupKake doesn't construct ladders natively, so we recycle the robust
        # general fractional thermodynamic logic implemented inside MolGpKa.
        from pick_a_pka.backends.molgpka.protonation import compute_microstates
        return compute_microstates(self, mol, ph=ph, ph_range=ph_range, ph_step=ph_step)
