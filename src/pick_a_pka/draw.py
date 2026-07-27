import io
import logging
import math
import re
from io import BytesIO

import cairosvg
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from rdkit import Chem
from rdkit.Chem import rdDepictor, rdFMCS
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Geometry import Point2D

from .predictor import PKaPredictor
from .core.canonicalize import optimize_sensical_folding

logger = logging.getLogger(__name__)

# Sentinel epsilon for label-label zero-distance tie-breaking in the
# force-directed loop. A fixed value (rather than random) keeps layout
# deterministic across runs.
_LABEL_EPSILON = 1e-5


def _neutral_query(mol):
    """Return a charge-stripped, Hs-removed copy for substructure matching.

    Charge-agnostic matching handles protonation-state differences (e.g. NH2
    vs NH3+, COOH vs COO-) that would otherwise block substructure search.
    """
    rw = Chem.RWMol(Chem.RemoveHs(mol))
    for atom in rw.GetAtoms():
        atom.SetFormalCharge(0)
        atom.SetNumExplicitHs(0)
        atom.SetNoImplicit(True)
    try:
        Chem.SanitizeMol(rw)
    except Exception:
        pass
    return rw.GetMol()


def _transfer_coords(ref_mol, target_mol):
    """Copy 2D atom positions from ref_mol onto target_mol in-place.

    Charge-agnostic matching handles protonation-state differences (e.g. NH2
    vs NH3+, COOH vs COO-) that would otherwise block substructure search.
    ref_mol must already carry a 2D conformer (e.g. from
    optimize_sensical_folding). After the transfer, every atom of target_mol
    that maps to ref_mol gets ref_mol's coordinates; any unmatched atoms
    (rare) keep whatever layout Compute2DCoords gave them.
    """
    if ref_mol.GetNumConformers() == 0:
        return

    ref_q = _neutral_query(ref_mol)
    tgt_q = _neutral_query(target_mol)

    # Fast path: direct substructure match covers most cases.
    match = target_mol.GetSubstructMatch(ref_q)
    if match and len(match) == ref_q.GetNumAtoms():
        ref_conf = ref_mol.GetConformer()
        tgt_conf = target_mol.GetConformer()
        for ref_idx, tgt_idx in enumerate(match):
            p = ref_conf.GetAtomPosition(ref_idx)
            tgt_conf.SetAtomPosition(tgt_idx, p)
        return

    # Fallback: MCS for larger charge/valence differences.
    try:
        mcs = rdFMCS.FindMCS(
            [ref_q, tgt_q],
            atomCompare=rdFMCS.AtomCompare.CompareElements,
            bondCompare=rdFMCS.BondCompare.CompareAny,
            completeRingsOnly=False,
            matchValences=False,
            matchChiralTag=False,
            timeout=3,
        )
        if mcs.numAtoms == 0:
            return
        patt = Chem.MolFromSmarts(mcs.smartsString)
        if patt is None:
            return
        ref_match = ref_q.GetSubstructMatch(patt)
        tgt_match = tgt_q.GetSubstructMatch(patt)
        if not ref_match or not tgt_match:
            return
        ref_conf = ref_mol.GetConformer()
        tgt_conf = target_mol.GetConformer()
        for ref_idx, tgt_idx in zip(ref_match, tgt_match):
            p = ref_conf.GetAtomPosition(ref_idx)
            tgt_conf.SetAtomPosition(tgt_idx, p)
    except Exception:
        logger.debug("_transfer_coords MCS fallback failed", exc_info=True)


def _place_labels(labels, atom_positions, n_iterations=500):
    """Adjust label positions with a force-directed repulsion pass.

    Pushes labels away from atoms and from each other while keeping each label
    tethered to its origin atom at safe_radius. Modifies labels[i]["pos"] in
    place.

    Args:
        labels: list of label dicts produced by draw_pka (keys: "origin",
            "safe_radius", "pos", "n_lines").
        atom_positions: list of (x, y) tuples for every atom in the molecule.
        n_iterations: number of relaxation steps (default 500).
    """
    for _ in range(n_iterations):
        displacements = [[0.0, 0.0] for _ in labels]

        for i, lbl in enumerate(labels):
            fx, fy = 0.0, 0.0
            lx, ly = lbl["pos"]

            # Repulsion from every atom that is not this label's own origin.
            for ax, ay in atom_positions:
                if (ax, ay) == lbl["origin"]:
                    continue
                dx, dy = lx - ax, ly - ay
                if dx == 0 and dy == 0:
                    dx, dy = _LABEL_EPSILON, _LABEL_EPSILON
                d = math.hypot(dx / 1.5, dy)
                if d < 0.8:
                    force = (0.8 - d) * 0.5
                    fx += (dx / d) * force
                    fy += (dy / d) * force

            # Repulsion between labels, scaled by line count.
            for j, o_lbl in enumerate(labels):
                if i == j:
                    continue
                ox, oy = o_lbl["pos"]
                dx, dy = lx - ox, ly - oy
                if dx == 0 and dy == 0:
                    dx, dy = _LABEL_EPSILON, _LABEL_EPSILON

                avg_lines = (lbl["n_lines"] + o_lbl["n_lines"]) / 2.0
                dy_adj = dy / (0.8 + 0.2 * avg_lines)
                d = math.hypot(dx / 1.5, dy_adj)
                if d < 1.4:
                    force = (1.4 - d) * 1.5
                    fx += (dx / d) * force
                    fy += (dy_adj / d) * force

            # Spring pulling label back toward its tether distance.
            orig_x, orig_y = lbl["origin"]
            dx, dy = orig_x - lx, orig_y - ly
            d = math.hypot(dx, dy)
            if d > 1e-4:
                force = (d - lbl["safe_radius"]) * 0.6
                fx += (dx / d) * force
                fy += (dy / d) * force

            displacements[i] = [fx, fy]

        for i, lbl in enumerate(labels):
            lbl["pos"][0] += displacements[i][0] * 0.3
            lbl["pos"][1] += displacements[i][1] * 0.3


def draw_pka(
    mol: Chem.Mol | str,
    model: PKaPredictor = None,
    image_size: tuple[int, int] = (800, 800),
    padding: float = 0.1,
    vector: bool = True,
) -> str | Image.Image:
    """Draw a molecule annotated with its predicted pKa values.

    Args:
        mol: RDKit Mol or SMILES string. If the mol already carries 2D
            coordinates (e.g. from optimize_sensical_folding), they are
            preserved in the output.
        model: PKaPredictor to use. Defaults to PKaPredictor() (MolGpKa
            backend, CPU).
        image_size: (width, height) in pixels.
        padding: fractional padding passed to the RDKit drawer.
        vector: if True return an SVG string; if False return a PIL Image.

    Returns:
        SVG string (vector=True) or PIL Image (vector=False).
    """
    if model is None:
        model = PKaPredictor()

    input_mol = model._to_mol(mol)[0]
    mol_copy = Chem.Mol(input_mol)

    pred = model.predict_pka(mol_copy)
    base_pka, acid_pka, mol_drawn = pred["base_pka"], pred["acid_pka"], pred["mol"]

    try:
        # Kekulize for a publication-ready single/double bond drawing.
        Chem.Kekulize(mol_drawn, clearAromaticFlags=True)
    except Exception:
        pass

    rdDepictor.SetPreferCoordGen(False)
    if input_mol.GetNumConformers() > 0:
        # mol_drawn is a Hs-stripped copy with no conformer yet; give it one
        # before transferring the caller's coords.
        rdDepictor.Compute2DCoords(mol_drawn)
        _transfer_coords(input_mol, mol_drawn)
    else:
        rdDepictor.Compute2DCoords(mol_drawn)

    # kekulize=False because we already pre-kekulized above.
    mol_prepared = rdMolDraw2D.PrepareMolForDrawing(mol_drawn, kekulize=False)
    conf = mol_prepared.GetConformer()
    n_atoms = mol_prepared.GetNumAtoms()

    if n_atoms == 0:
        return Image.new("RGB", image_size, (255, 255, 255))

    drawer = (rdMolDraw2D.MolDraw2DSVG if vector else rdMolDraw2D.MolDraw2DCairo)(
        image_size[0], image_size[1]
    )
    draw_opts = drawer.drawOptions()
    draw_opts.clearBackground = False
    draw_opts.addAtomIndices = False
    draw_opts.padding = padding

    drawer.DrawMolecule(mol_prepared)

    atom_positions = [
        (conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y)
        for i in range(n_atoms)
    ]
    cog_x = sum(p[0] for p in atom_positions) / n_atoms
    cog_y = sum(p[1] for p in atom_positions) / n_atoms

    # Merge acid (dark red) and base (blue) pKa dicts per atom.
    combined_pka: dict[int, list] = {}
    for idx, pka in acid_pka.items():
        combined_pka.setdefault(idx, []).append((pka, (0.6, 0.0, 0.3, 1.0)))
    for idx, pka in base_pka.items():
        combined_pka.setdefault(idx, []).append((pka, (0.0, 0.2, 0.8, 1.0)))

    labels = []
    for atom_idx, pka_list in combined_pka.items():
        if atom_idx >= n_atoms:
            continue

        pos = atom_positions[atom_idx]
        atom = mol_prepared.GetAtomWithIdx(atom_idx)

        # Local direction: away from the centroid of bonded neighbours.
        v_local_x, v_local_y = 0.0, 0.0
        neighbors = atom.GetNeighbors()
        if neighbors:
            for n in neighbors:
                nx, ny = atom_positions[n.GetIdx()]
                dx, dy = nx - pos[0], ny - pos[1]
                dist = math.hypot(dx, dy)
                if dist > 0:
                    v_local_x += dx / dist
                    v_local_y += dy / dist
            v_local_x, v_local_y = -v_local_x, -v_local_y
        else:
            v_local_x, v_local_y = 0.0, -1.0

        # Global direction: away from the molecule centre of geometry.
        v_glob_x, v_glob_y = pos[0] - cog_x, pos[1] - cog_y
        dist_glob = math.hypot(v_glob_x, v_glob_y)
        if dist_glob > 1e-4:
            v_glob_x /= dist_glob
            v_glob_y /= dist_glob

        vx = v_local_x + 0.8 * v_glob_x
        vy = v_local_y + 0.8 * v_glob_y
        v_len = math.hypot(vx, vy)
        if v_len > 1e-4:
            vx /= v_len
            vy /= v_len
        else:
            vx, vy = 1.0, 0.0

        num_hs = atom.GetTotalNumHs()
        degree = atom.GetDegree()

        safe_radius = 0.5 + (0.3 * num_hs)
        if degree >= 3:
            safe_radius += 0.5
        if len(pka_list) > 1:
            safe_radius += 0.35 * (len(pka_list) - 1)
        safe_radius += 0.15 * abs(vx)

        origin_x, origin_y = pos[0], pos[1]
        labels.append({
            "origin": (origin_x, origin_y),
            "safe_radius": safe_radius,
            "pos": [origin_x + vx * safe_radius, origin_y + vy * safe_radius],
            "pka_list": pka_list,
            "n_lines": len(pka_list),
        })

    _place_labels(labels, atom_positions)

    # Render leader lines and pKa text for every label.
    for lbl in labels:
        lx, ly = lbl["pos"]
        orig_x, orig_y = lbl["origin"]
        safe_rad = lbl["safe_radius"]

        dx, dy = lx - orig_x, ly - orig_y
        d = math.hypot(dx, dy)

        if d > safe_rad + 0.15:
            start_pt = Point2D(
                orig_x + (dx / d) * (safe_rad - 0.1),
                orig_y + (dy / d) * (safe_rad - 0.1),
            )
            end_pt = Point2D(lx - (dx / d) * 0.35, ly - (dy / d) * 0.35)
            drawer.SetColour((0.6, 0.6, 0.6, 1.0))
            drawer.DrawLine(start_pt, end_pt)

        pka_list = lbl["pka_list"]
        line_height = 0.65
        start_y = ly - (len(pka_list) - 1) * line_height / 2.0

        for i, (pka_val, color) in enumerate(pka_list):
            text_pt = Point2D(lx, start_y + i * line_height)
            drawer.SetColour(color)
            drawer.DrawString(f"pKa: {pka_val:.1f}", text_pt)

    drawer.FinishDrawing()
    content = drawer.GetDrawingText()
    if vector:
        return content
    return Image.open(io.BytesIO(content))


def plot_microspecies_distribution(
    mol: Chem.Mol | str,
    model: PKaPredictor = None,
    vector: bool = True,
) -> str | Image.Image:
    """Plot microspecies abundance versus pH for a molecule.

    Generates a combined figure with:
    - A distribution curve for each protonation state (left: annotated
      molecule with pKa callouts; right: the plot itself).
    - Thumbnail structures for every state below the plot.

    Args:
        mol: RDKit Mol or SMILES string.
        model: PKaPredictor to use. Defaults to PKaPredictor() (MolGpKa
            backend, CPU).
        vector: if True return an SVG string; if False return a PIL Image
            (requires cairosvg).

    Returns:
        SVG string (vector=True) or PIL Image (vector=False).
    """
    if model is None:
        model = PKaPredictor()

    mol = model._to_mol(mol)[0]
    mol = optimize_sensical_folding(mol)

    micro_data = model.predict_microstates(mol, ph_range=(0, 14), ph_step=0.05)
    pred = model.predict_pka(mol)
    base_pka_dict, acid_pka_dict = pred["base_pka"], pred["acid_pka"]

    X_pH = sorted(micro_data.keys())

    # state_smiles comes from the same helper as compute_microstates, so the
    # SMILES align; it drives both curve ordering and thumbnail ordering.
    state_smiles = model.protonation_ladder(mol, acid_first=True)

    # Build a SMILES → mol lookup for thumbnail rendering.
    # Index every mol object in micro_data by its canonical SMILES.
    smi_to_mol: dict[str, Chem.Mol] = {}
    for ph in X_pH:
        for d in micro_data[ph]["distribution"]:
            smi = d["smiles"]
            if smi not in smi_to_mol:
                smi_to_mol[smi] = d["mol"]
    # Some states may have negligible abundance and never appear in micro_data
    # — parse them from SMILES as a fallback.
    for smi in state_smiles:
        if smi not in smi_to_mol:
            m = Chem.MolFromSmiles(smi)
            if m is not None:
                smi_to_mol[smi] = m

    num_states = len(state_smiles)

    # Pre-index each pH's distribution by SMILES for O(1) lookup.
    dist_by_ph: dict[float, dict[str, float]] = {
        ph: {d["smiles"]: d["abundance"] for d in micro_data[ph]["distribution"]}
        for ph in X_pH
    }

    Y_abundances = []
    for smi in state_smiles:
        y_curve = [dist_by_ph[ph].get(smi, 0.0) for ph in X_pH]
        Y_abundances.append(y_curve)

    colors = [
        "#e74c3c", "#3498db", "#f39c12", "#2ecc71", "#9b59b6", "#34495e",
        "#8c564b", "#e377c2", "#7e7e7e", "#bcbd22", "#17becf", "#b8105a",
        "#620f77", "#ff9896", "#aec7e8", "#ffbb78", "#98df8a", "#c5b0d5",
        "#56799c", "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae4",
        "#ff4f9b", "#cf99ff",
    ]
    linestyles = ["-", "--", "-.", ":"]

    max_cols = 6
    num_rows = math.ceil(num_states / max_cols)

    fig_width_in = 14
    plot_height_in = 5.0
    row_height_in = 2.4
    text_space_in = 1.0
    top_margin_in = 0.5

    bottom_margin_in = (num_rows * row_height_in) + text_space_in
    fig_height_in = plot_height_in + bottom_margin_in + top_margin_in

    plot_left_margin_frac = 0.4
    plot_width_frac = 0.58

    ax_height_frac = plot_height_in / fig_height_in
    bottom_margin_frac = bottom_margin_in / fig_height_in

    fig = plt.figure(figsize=(fig_width_in, fig_height_in))
    ax = fig.add_axes(
        (plot_left_margin_frac, bottom_margin_frac, plot_width_frac, ax_height_frac)
    )

    for i in range(num_states):
        c_idx = i % len(colors)
        ls_idx = (i // len(colors)) % len(linestyles)
        ax.plot(X_pH, Y_abundances[i], lw=2.5, color=colors[c_idx], linestyle=linestyles[ls_idx])

    ax.set_xlim(0, 14.01)
    ax.set_ylim(0, 101)
    ax.set_xlabel("pH", fontsize=10, color="#333333")
    ax.set_ylabel("Microspecies distribution (%)", fontsize=10, color="#333333")
    ax.set_xticks(np.arange(0, 15, 2))
    ax.set_yticks(np.arange(0, 101, 10))
    ax.tick_params(axis="both", colors="#555555")
    ax.grid(True, linestyle="-", alpha=0.4, color="#cccccc")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#dddddd")
    ax.spines["bottom"].set_color("#dddddd")

    if acid_pka_dict:
        min_acid = min(acid_pka_dict.values())
        fig.text(
            0.085, 0.6 / fig_height_in,
            f"Strongest acidic pKa: {min_acid:.2f}",
            fontsize=10, ha="left",
        )
    if base_pka_dict:
        max_base = max(base_pka_dict.values())
        fig.text(
            0.085, 0.2 / fig_height_in,
            f"Strongest basic pKa: {max_base:.2f}",
            fontsize=10, ha="left",
        )

    grid_x = np.linspace(1, 13, max_cols)

    buf = io.StringIO()
    fig.savefig(buf, format="svg", transparent=True)
    plt.close(fig)
    mpl_svg = buf.getvalue()

    # All thumbnails share the same scaffold orientation as mol (which carries
    # the optimize_sensical_folding conformer). _transfer_coords maps those
    # coords onto each microstate mol charge-agnostically.
    rdDepictor.SetPreferCoordGen(False)

    def _apply_ref_layout(state_mol_raw: Chem.Mol) -> Chem.Mol:
        """Return a mol whose atom positions match the reference layout of mol."""
        state_mol_kek = Chem.Mol(state_mol_raw)
        try:
            Chem.Kekulize(state_mol_kek, clearAromaticFlags=True)
        except Exception:
            pass
        state_no_hs = Chem.RemoveHs(state_mol_kek)
        # Compute a starting conformer first; _transfer_coords overwrites
        # matched-atom positions with the reference coordinates.
        rdDepictor.Compute2DCoords(state_no_hs)
        _transfer_coords(mol, state_no_hs)
        return state_no_hs

    state_molecules = [_apply_ref_layout(smi_to_mol[smi]) for smi in state_smiles]
    injections = []

    mol_size_pt = 160
    internal_res = 1000

    micro_left_margin_frac = 0.08
    micro_width_frac = 0.88

    for i, state_mol in enumerate(state_molecules):
        c_idx = i % len(colors)
        ls_idx = (i // len(colors)) % len(linestyles)

        row = i // max_cols
        col = i % max_cols
        x_val = grid_x[col]

        state_mol_drawn = Chem.Mol(state_mol)
        try:
            mol_prep = rdMolDraw2D.PrepareMolForDrawing(state_mol_drawn, kekulize=False)
        except Exception:
            mol_prep = rdMolDraw2D.PrepareMolForDrawing(state_mol_drawn)

        thumb_drawer = rdMolDraw2D.MolDraw2DSVG(internal_res, internal_res)
        opts = thumb_drawer.drawOptions()
        opts.padding = 0.1
        opts.clearBackground = False
        opts.bondLineWidth = 3
        if hasattr(opts, "maxFontSize"):
            opts.maxFontSize = 24

        thumb_drawer.DrawMolecule(mol_prep)
        thumb_drawer.FinishDrawing()

        svg_mol = thumb_drawer.GetDrawingText()
        start_idx = svg_mol.find("<svg")
        svg_mol = svg_mol[start_idx:]
        svg_mol = re.sub(r"width='.*?px'", "width='100%'", svg_mol, count=1)
        svg_mol = re.sub(r"height='.*?px'", "height='100%'", svg_mol, count=1)

        frac_x = micro_left_margin_frac + (x_val / 14) * micro_width_frac
        x_pt = frac_x * fig_width_in * 72.0
        y_center_in = top_margin_in + plot_height_in + (row + 0.4) * row_height_in
        y_pt = y_center_in * 72.0

        anchor_x = x_pt - (mol_size_pt / 2)
        anchor_y = y_pt - (mol_size_pt / 2)

        injections.append(
            f'<svg x="{anchor_x}" y="{anchor_y}" width="{mol_size_pt}" height="{mol_size_pt}">\n'
            f"{svg_mol}\n</svg>"
        )

        line_y_pt = (top_margin_in + plot_height_in + (row + 0.88) * row_height_in) * 72.0
        line_w_pt = 60
        dash_str = ""
        if linestyles[ls_idx] == "--":
            dash_str = 'stroke-dasharray="8,4"'
        elif linestyles[ls_idx] == "-.":
            dash_str = 'stroke-dasharray="8,4,2,4"'
        elif linestyles[ls_idx] == ":":
            dash_str = 'stroke-dasharray="3,3"'

        injections.append(
            f'<line x1="{x_pt - line_w_pt}" y1="{line_y_pt}" '
            f'x2="{x_pt + line_w_pt}" y2="{line_y_pt}" '
            f'stroke="{colors[c_idx]}" stroke-width="4" {dash_str} />'
        )

    annotated_svg = draw_pka(
        mol, model=model, vector=True,
        image_size=(internal_res, internal_res), padding=0.15,
    )
    start_idx = annotated_svg.find("<svg")
    if start_idx != -1:
        annotated_svg = annotated_svg[start_idx:]
    annotated_svg = re.sub(r"width='.*?px'", "width='100%'", annotated_svg, count=1)
    annotated_svg = re.sub(r"height='.*?px'", "height='100%'", annotated_svg, count=1)

    margin_gap_pt = 35.0
    left_padding_pt = 5.0

    annotated_area_width_in = fig_width_in * plot_left_margin_frac
    annotated_size_pt = annotated_area_width_in * 72.0 * 0.98 - margin_gap_pt - left_padding_pt
    max_allowed_height_pt = plot_height_in * 72.0 * 1.3
    if annotated_size_pt > max_allowed_height_pt:
        annotated_size_pt = max_allowed_height_pt

    annotated_x_pt = left_padding_pt
    plot_center_y_pt = (top_margin_in + plot_height_in / 2.0) * 72.0
    annotated_y_pt = plot_center_y_pt - (annotated_size_pt / 2.0)

    injections.append(
        f'<svg x="{annotated_x_pt}" y="{annotated_y_pt}" '
        f'width="{annotated_size_pt}" height="{annotated_size_pt}">\n'
        f"{annotated_svg}\n</svg>"
    )

    end_tag_idx = mpl_svg.rfind("</svg>")
    final_svg = mpl_svg[:end_tag_idx] + "\n".join(injections) + "\n</svg>"

    if vector:
        return final_svg
    img = BytesIO()
    cairosvg.svg2png(bytestring=final_svg, write_to=img, dpi=300)
    return Image.open(img)
