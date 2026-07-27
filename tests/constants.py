"""Constants used in the tests."""

ACETIC_ACID = "CC(=O)O"  # one acidic site, pKa ≈ 4.76
GLYCINE = "NCC(=O)O"  # amphoteric
L_ALANINE = "C[C@@H](N)C(=O)O"  # chiral centre
D_ALANINE = "C[C@H](N)C(=O)O"
L_ALANINE_PROTONATED = "C[C@@H]([NH3+])C(=O)O"
D_ALANINE_PROTONATED = "C[C@H]([NH3+])C(=O)O"

ANILINE = "c1ccccc1N"

# no ionisable sites
METHANE = "C"
ETHANE = "CC"
PROPANE = "CCC"
BUTANE = "CCCC"
PENTANE = "CCCCC"
HEXANE = "CCCCCC"
OCTANE = "CCCCCCCC"
DECANE = "CCCCCCCCCC"
BENZENE = "c1ccccc1"

MORPHOLINE = "C1COCCN1"  # secondary amine + ether: one basic site, one basic site, pKa ≈ 8.7
NON_CANONICAL_ACETIC_ACID = "OC(C)=O"
CHLOROACETIC_ACID = "ClCC(=O)O"
CYCLOHEXANOL = "OC1CCCCC1"
PHENOL = "c1ccccc1O"
PHENYLALANINE = "N[C@@H](Cc1ccccc1)C(=O)O"  # amphoteric
TRIMETHYLAMINE = "CN(C)C"  # simple amine with no acidic proton in the neutral form
BENZOIC_ACID = "OC(=O)c1ccccc1"  # carboxylic acid with no basic nitrogen

# contains NH groups that are both basic and acidic
OSIMERTINIB = "CN1C=C(C2=CC=CC=C21)C3=NC(=NC=C3)NC4=C(C=C(C(=C4)NC(=O)C=C)N(C)CCN(C)C)OC"

CHEMBL5646830 = "CN1C=C(C2=CC=CC=C21)C3=NC(=NC=C3)NC4=CC=C(N(C)CCN(C)C)C=C4"

ACETAMINOPHEN = "CC(=O)Nc1ccc(O)cc1"

HISTIDINE = "NC(Cc1c[nH]cn1)C(=O)O"  # amphoteric, amine + imidazole + carboxyl
PYRIDINE = "c1ccncc1"  # one basic ring N
IMIDAZOLE = "c1cnc[nH]1"  # ring N-H (acidic) + ring N (basic)
BENZENESULFONAMIDE = "NS(=O)(=O)c1ccccc1"
METHANESULFONIC_ACID = "CS(=O)(=O)O"
ETHANOLAMINE = "NCCO"
PARA_AMINOPHENOL = "Nc1ccc(O)cc1"
GUANIDINE = "NC(=N)N"
TYROSINE = "NC(Cc1ccc(O)cc1)C(=O)O"  # amphoteric, amine + phenol + carboxyl
