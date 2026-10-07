"""Physical constants.

Values match the Fortran ``constants.inc`` so results stay comparable with
legacy SIMC runs.  Units: MeV, cm, fm unless noted.
"""

import math

# Masses (MeV)
ME = 0.51099906
MP = 938.27231
MN = 939.56563
MPI = 139.57018
MPI0 = 134.9766
MK = 493.677
MRHO = 769.3
MD = 1875.613
MLAMBDA = 1115.68
MSIGMA0 = 1192.64
MSIGMA_MINUS = 1197.45
MDELTA = 1232.0

AMU = 931.49432            # MeV
HBARC = 197.327053         # MeV fm
ALPHA = 1.0 / 137.0359895
DEGRAD = 180.0 / math.pi   # degrees per radian
MU_P = 2.792847            # proton magnetic moment (nuclear magnetons)

# Conversions / luminosity
FM2_TO_UB = 1.0e4          # 1 fm^2 = 1e4 microbarn
E_CHARGE_C = 1.602176634e-19
N_AVOGADRO = 6.02214076e23
