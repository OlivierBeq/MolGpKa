import copy
import math

import numpy as np
from rdkit import Chem

from .change_ionization import parse_smiles, find_centers, addHs, ionizeN
from .featurizer import mol_to_graph
from .inference import predict_single
from ...core.types import LadderStep, MicrostateResult, StateDistribution


class DummyArgs:
    def __init__(self):
        self.carbons_included = False
        self.verbose = 0
        self.mode = "infer"


def _isDigit(char):
    return char in '0123456789'


def _clean_smiles(smiles):
    j = 0
    while j < len(smiles):
        if smiles[j] == ':':
            pos = len(smiles)
            for k in range(j + 1, len(smiles)):
                if not _isDigit(smiles[k]):
                    pos = k
                    break
            smiles = smiles[:j] + smiles[pos:]
            smiles = smiles.replace('[N]', 'N').replace('[n]', 'n').replace('[O]', 'O')
            break
        j += 1
    return smiles


def _infer_round(model_wrapper, smiles, initial, ionization_states_in, config, allow_amphoteric=False):
    dummy_args = DummyArgs()

    if initial:
        smiles = _clean_smiles(smiles)
        smiles = smiles.replace('([H])', '').replace('[H]', '').replace('[C-]', 'C').replace('-c', 'c').replace('[n]',
                                                                                                                'n'
                                                                                                                )

    mol_original = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol_original:
        Chem.rdmolops.RemoveHs(mol_original, sanitize=False)
        Chem.SanitizeMol(mol_original, catchErrors=True)

    negative_nitrogens = []
    pyridinium = []

    if initial:
        ionizable_nitrogens, positive_nitrogens, acidic_nitrogens, negative_oxygens, \
            acidic_oxygens, acidic_carbons, nitro_nitrogens = find_centers(mol_original, 0, smiles, "mol", initial,
                                                                           dummy_args
                                                                           )

        smiles = ionizeN(smiles, mol_original, mol_original.GetNumAtoms(), acidic_nitrogens, acidic_oxygens,
                         acidic_carbons, ionizable_nitrogens, negative_nitrogens, negative_oxygens, nitro_nitrogens,
                         pyridinium, dummy_args
                         )

        smiles = _clean_smiles(smiles)
        j = 0
        atom_idx = 0
        while j < len(smiles):
            _, smiles, j, atom_idx = parse_smiles(smiles, j, atom_idx, initial, ionizable_nitrogens,
                                                  positive_nitrogens, acidic_nitrogens, negative_nitrogens,
                                                  negative_oxygens, acidic_oxygens, acidic_carbons,
                                                  pyridinium, nitro_nitrogens, False, True
                                                  )

        mol_original = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol_original:
            Chem.rdmolops.RemoveHs(mol_original, sanitize=False)
            Chem.SanitizeMol(mol_original, catchErrors=True)
            smiles = addHs(smiles, mol_original, mol_original.GetNumAtoms(), negative_nitrogens)
            smiles = ionizeN(smiles, mol_original, mol_original.GetNumAtoms(), acidic_nitrogens, acidic_oxygens,
                             acidic_carbons, ionizable_nitrogens, negative_nitrogens, negative_oxygens, nitro_nitrogens,
                             pyridinium, dummy_args
                             )

        ionization_states0 = [
            ionizable_nitrogens, positive_nitrogens, acidic_nitrogens,
            negative_nitrogens, negative_oxygens, acidic_oxygens,
            acidic_carbons, nitro_nitrogens
        ]
    else:
        ionization_states0 = ionization_states_in

    predicts, inf_smiles_list, centers, ion_states_list = [], [], [], []
    j, atom_idx = -1, 0
    smiles_A = smiles

    evaluated_in_round = set()

    while j < len(smiles_A):
        st = [copy.deepcopy(x) for x in ionization_states0]
        if j < 0: j = 0

        is_smiles, smiles_A, j, atom_idx = parse_smiles(
            smiles, j, atom_idx, initial, st[0], st[1], st[2], st[3], st[4], st[5], st[6], pyridinium, st[7], True,
            False
        )

        if is_smiles:
            center = atom_idx - 1
            if center in evaluated_in_round:
                continue

            mol_obj_A = Chem.MolFromSmiles(smiles_A, sanitize=False)
            if not mol_obj_A: continue
            Chem.rdmolops.RemoveHs(mol_obj_A, sanitize=False)
            Chem.SanitizeMol(mol_obj_A, catchErrors=True)

            data = mol_to_graph(mol_obj_A, center, config)
            if data is None: continue

            predicts.append(predict_single(model_wrapper.model, data, model_wrapper.device))
            inf_smiles_list.append(smiles_A)
            centers.append(center)
            ion_states_list.append(st)
            evaluated_in_round.add(center)

    return predicts, inf_smiles_list, centers, ion_states_list


def predict_ladder(model_wrapper, original_smiles, config, allow_amphoteric=False) -> list[LadderStep]:
    """Iterative macroscopic deprotonation sequence.

    Mirrors the original pKaLearn CLI behaviour: at each round the site with
    the *highest* predicted pKa is deprotonated first (most basic site first),
    producing a strictly descending pKa ladder.
    """
    all_results = []
    initial = True
    curr_smiles = original_smiles
    curr_ion_states = []

    while True:
        predicts, smis, centers, states = _infer_round(
            model_wrapper, curr_smiles, initial, curr_ion_states, config, allow_amphoteric
        )
        if not predicts: break

        # Select the site with the HIGHEST pKa — the most basic site is
        # deprotonated first as pH rises, matching the original CLI.
        best_idx = predicts.index(max(predicts))
        best_center = centers[best_idx]
        best_pka = predicts[best_idx]
        best_smiles = smis[best_idx]

        # Record before the stall check: a single-site molecule's candidate
        # SMILES can equal curr_smiles even though it's a real deprotonation.
        all_results.append(LadderStep(
            smiles=best_smiles,
            center=best_center,
            pka=best_pka
        )
        )

        if best_smiles == curr_smiles:
            break

        curr_smiles = best_smiles
        curr_ion_states = states[best_idx]
        initial = False

        # Guard against runaway ladders into very strong acid territory.
        if best_pka < -25:
            break

    return all_results


def compute_microstates(model_wrapper, mol, ph=7.4, ph_range=None, ph_step=None) -> dict[
    float, MicrostateResult]:
    """Build microstate distribution from the deprotonation ladder.

    Uses protonation_ladder (via the predictor wrapper) as the single source of
    truth for the ordered state sequence, so that the SMILES in the distribution
    are guaranteed to match those returned by protonation_ladder.  This removes
    any reliance on step["center"] index mapping inside compute_microstates.
    """
    # model_wrapper is the backend model (PkaLearnModel); reach up to the
    # PKaPredictor via the _predictor back-reference set in predict_microstates,
    # or fall back to rebuilding states from predict_pka directly.
    pred = model_wrapper.predict_pka(mol)
    base_pka_dict = pred["base_pka"]
    acid_pka_dict = pred["acid_pka"]
    mol_no_hs = pred["mol"]

    # Build the ordered state list using the same _generate_ordered_states
    # helper that protonation_ladder uses, so SMILES are identical.
    from ...predictor import _generate_ordered_states
    states = _generate_ordered_states(mol_no_hs, base_pka_dict, acid_pka_dict)

    all_pkas = sorted(list(base_pka_dict.values()) + list(acid_pka_dict.values()))

    def _major(dist):
        return max(dist, key=lambda x: x["abundance"])

    def get_dist_at_ph(current_ph):
        if not all_pkas:
            state = states[0]
            return [StateDistribution(smiles=Chem.MolToSmiles(state), mol=state, abundance=100.0)]

        log_ratios = [0.0]
        current_sum_pka = 0.0
        for k in range(1, len(all_pkas) + 1):
            current_sum_pka += all_pkas[k - 1]
            log_ratios.append(k * current_ph - current_sum_pka)

        max_log = max(log_ratios)
        ratios = [10 ** (lr - max_log) for lr in log_ratios]
        total_ratio = sum(ratios)
        fractions = [(r / total_ratio) * 100.0 for r in ratios]

        # Keep states in protonation-ladder order — do NOT sort by abundance.
        # Sorting here would scramble the stable state->colour mapping that
        # plot_microspecies_distribution relies on.
        dist = []
        for frac, state_mol in zip(fractions, states):
            dist.append(StateDistribution(smiles=Chem.MolToSmiles(state_mol), mol=state_mol, abundance=frac))
        return dist

    if ph_range is not None:
        if ph_step is None:
            raise ValueError("`ph_step` must be specified when using `ph_range`.")
        results = {}
        for current_ph in np.arange(ph_range[0], ph_range[1] + (ph_step / 2), ph_step):
            rounded_ph = round(current_ph, max(0, int(math.ceil(-math.log10(ph_step)))))
            dist = get_dist_at_ph(rounded_ph)
            major = _major(dist)
            results[rounded_ph] = MicrostateResult(
                major_state=major["mol"],
                major_abundance=major["abundance"],
                distribution=dist
            )
        return results

    dist = get_dist_at_ph(ph)
    major = _major(dist)
    return MicrostateResult(
        major_state=major["mol"],
        major_abundance=major["abundance"],
        distribution=dist
    )
