import copy
from importlib import resources

import torch
from rdkit import Chem
from rdkit.Chem import rdFMCS

from .network import PkaLearnGNN
from ...core.base import BasePKaModel
from ...core.exceptions import ResourceNotFoundError


def _map_ladder_center_to_neutral(step_mol, center_idx, neutral_mol):
    """Map center_idx (an atom index into step_mol) to the matching atom index
    in neutral_mol, via substructure match, ignoring charges and Hs."""
    # Build a charge-agnostic, H-stripped copy of step_mol for matching
    rw = Chem.RWMol(Chem.RemoveHs(step_mol))
    for atom in rw.GetAtoms():
        atom.SetFormalCharge(0)
        atom.SetNumExplicitHs(0)
        atom.SetNoImplicit(True)
    try:
        Chem.SanitizeMol(rw)
    except Exception:
        pass
    query = rw.GetMol()

    # Try a direct substructure match first (fast path)
    match = neutral_mol.GetSubstructMatch(query)
    if match and center_idx < len(match):
        return match[center_idx]

    # Fall back to MCS for molecules that differ more significantly
    try:
        mcs = rdFMCS.FindMCS(
            [query, neutral_mol],
            atomCompare=rdFMCS.AtomCompare.CompareElements,
            bondCompare=rdFMCS.BondCompare.CompareAny,
            completeRingsOnly=False,
            matchValences=False,
            matchChiralTag=False,
            timeout=3,
        )
        if mcs.numAtoms == 0:
            return None
        patt = Chem.MolFromSmarts(mcs.smartsString)
        if patt is None:
            return None
        q_match = query.GetSubstructMatch(patt)
        n_match = neutral_mol.GetSubstructMatch(patt)
        if q_match and n_match and center_idx in q_match:
            pos = q_match.index(center_idx)
            return n_match[pos]
    except Exception:
        pass

    return None


class PkaLearnModel(BasePKaModel):
    DEFAULT_CONFIG = {
        'atom_feature_element': False,
        'atom_feature_electronegativity': True,
        'atom_feature_hardness': True,
        'atom_feature_atom_size': True,
        'atom_feature_hybridization': True,
        'atom_feature_aromaticity': True,
        'atom_feature_number_of_rings': False,
        'atom_feature_ring_size': True,
        'atom_feature_number_of_Hs': True,
        'atom_feature_formal_charge': True,
        'bond_feature_bond_order': True,
        'bond_feature_conjugation': True,
        'bond_feature_polarization': True,
        'bond_feature_charge_conjugation': True,
        'bond_feature_focused': False,
        'acid_or_base': 'base',
        'mask_size': 4,
        'model_embedding_size': 128,
        'model_gnn_layers': 4,
        'model_fc_layers': 2,
        'model_dropout_rate': 0.0,
        'model_dense_neurons': 448,
        'model_attention_heads': 4
    }

    def __init__(self, device="cpu", config=None, allow_amphoteric: bool = False):
        super().__init__(device=device)
        self.config = config or self.DEFAULT_CONFIG
        self.allow_amphoteric = allow_amphoteric
        self.model = PkaLearnGNN(feature_size=19, edge_dim=7, model_params=self.config)
        self._load_weights()
        self.model.to(self.device)
        self.model.eval()

    def _load_weights(self):
        pkg = "pick_a_pka.backends.pkalearn.resources"
        try:
            with resources.as_file(resources.files(pkg).joinpath("train_AAc-1_best.pth")) as path:
                ckpt = torch.load(path, map_location=self.device, weights_only=True)
            state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
            self.model.load_state_dict(state_dict)
        except Exception as e:
            raise ResourceNotFoundError(f"Cound not load pKaLearn model weights: {e}")

    @torch.no_grad()
    def predict(self, mol_or_smiles):
        """Run the iterative deprotonation ladder.

        Returns (ladder, starting_mol): ladder is a list of LadderStep dicts
        ({'smiles', 'center', 'pka'}); starting_mol is pre-protonated when
        allow_amphoteric=True.
        """
        from .microstates import predict_ladder

        if isinstance(mol_or_smiles, str):
            mol = Chem.MolFromSmiles(mol_or_smiles)
        else:
            mol = Chem.Mol(mol_or_smiles)

        mol_clean = Chem.RemoveHs(mol)

        if self.allow_amphoteric:
            # Pre-protonate neutral, non-aromatic N so they enter the ladder at
            # the top. Aromatic N excluded: protonating it can sanitize into a
            # non-aromatic dication (e.g. imidazole) instead of a single proton.
            rw_mol = Chem.RWMol(mol_clean)
            patt = Chem.MolFromSmarts('[#7+0]')
            if patt:
                for m in rw_mol.GetSubstructMatches(patt):
                    atom = rw_mol.GetAtomWithIdx(m[0])
                    if atom.GetDegree() <= 3 and not atom.GetIsAromatic():
                        atom.SetFormalCharge(1)
                        atom.SetNumExplicitHs(atom.GetNumExplicitHs() + 1)
            try:
                Chem.SanitizeMol(rw_mol)
                mol_clean = rw_mol.GetMol()
            except Exception:
                mol_clean = Chem.RemoveHs(mol)

        # Non-canonical SMILES preserves the native node order perfectly.
        smiles_str = Chem.MolToSmiles(mol_clean, canonical=False)
        # Amphoteric second pass happens in predict_pka, not here, to avoid duplicates.
        ladder = predict_ladder(self, smiles_str, self.config, allow_amphoteric=False)
        return ladder, mol_clean

    def predict_pka(self, mol):
        from .featurizer import from_acid_to_base
        from .featurizer import mol_to_graph
        from .inference import predict_single

        neutral_mol = Chem.RemoveHs(mol) if isinstance(mol, Chem.Mol) else Chem.MolFromSmiles(mol)
        ladder, _ = self.predict(Chem.Mol(neutral_mol))

        base_pka = {}
        acid_pka = {}

        for step in ladder:
            pka = step['pka']
            ladder_idx = step['center']  # index in the ladder's mutated SMILES mol

            # Parse the ladder step's molecule (the protonated acid-form SMILES).
            step_mol = Chem.MolFromSmiles(step['smiles'])
            if step_mol is None:
                step_mol = Chem.MolFromSmiles(step['smiles'], sanitize=False)
            if step_mol is None or ladder_idx >= step_mol.GetNumAtoms():
                continue

            # ladder_idx indexes step_mol, not neutral_mol; map it across.
            neutral_idx = _map_ladder_center_to_neutral(step_mol, ladder_idx, neutral_mol)
            if neutral_idx is None:
                continue

            # fc < 0 after deprotonation -> acidic pKa; fc >= 0 -> basic pKa.
            deprotonated = copy.deepcopy(step_mol)
            b_found, deprotonated, _ = from_acid_to_base(deprotonated, ladder_idx)
            if not b_found:
                continue

            post_fc = deprotonated.GetAtomWithIdx(ladder_idx).GetFormalCharge()
            if post_fc < 0:
                # Skip if already seen as basic; the amphoteric pass below handles it.
                if neutral_idx not in base_pka:
                    acid_pka[neutral_idx] = pka
            else:
                base_pka[neutral_idx] = pka

        # Amphoteric extension: a basic atom's neutral form may still carry a
        # proton it can also lose (e.g. NH2 -> NH-), which the ladder never reaches.
        if self.allow_amphoteric and base_pka:
            for neutral_idx in list(base_pka.keys()):
                if neutral_idx in acid_pka:
                    continue

                atom = neutral_mol.GetAtomWithIdx(neutral_idx)

                # Must carry a proton in its neutral form to act as an acid.
                if atom.GetTotalNumHs() == 0:
                    continue
                # Only meaningful for classic amphoteric heteroatoms.
                if atom.GetSymbol() not in ('N', 'O', 'S', 'P'):
                    continue

                # Confirm deprotonation of the neutral atom yields an anion.
                mol_check = copy.deepcopy(neutral_mol)
                b_found, mol_check, _ = from_acid_to_base(mol_check, neutral_idx)
                if not b_found:
                    continue
                if mol_check.GetAtomWithIdx(neutral_idx).GetFormalCharge() >= 0:
                    continue  # no anion formed -> not a genuine acidic site

                # neutral_mol is already the acid-form input for this transition;
                # adding another proton would re-evaluate base_pka's transition instead.
                data = mol_to_graph(copy.deepcopy(neutral_mol), neutral_idx, self.config)
                if data is None:
                    continue

                acid_pka[neutral_idx] = predict_single(self.model, data, self.device)

        return {
            "base_pka": base_pka,
            "acid_pka": acid_pka,
            "mol": neutral_mol
        }

    def predict_microstates(self, mol, ph=7.4, ph_range=None, ph_step=None):
        from .microstates import compute_microstates
        return compute_microstates(self, mol, ph=ph, ph_range=ph_range, ph_step=ph_step)
