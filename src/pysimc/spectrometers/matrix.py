"""Building block for COSY-style transport: vectorised polynomial matrices.

The Fortran spectrometer codes (``hms/mc_hms.f`` ...) evaluate sums
    out_i = sum_t  c_t * prod_j  x_j ^ e_tj
event by event, reading coefficients from ``forward_cosy.dat`` /
``recon_cosy.dat``.  ``PolynomialMatrix`` does the same for a whole batch.

TODO:
  * a parser for the COSY ``.dat`` files -> ``PolynomialMatrix`` per output
  * ``MatrixSpectrometer(Spectrometer)``: forward transport, aperture checks
    (``apertures_*.inc``), multiple scattering, detector-resolution smearing,
    reconstruction.  Register it under "hms", "shms", ... replacing the box.
"""

from __future__ import annotations

import numpy as np


class PolynomialMatrix:
    def __init__(self, exponents: np.ndarray, coefficients: np.ndarray, chunk: int = 2048):
        self.exponents = np.asarray(exponents, dtype=np.intp)       # (T, k)
        self.coefficients = np.asarray(coefficients, dtype=float)   # (T,)
        if self.exponents.shape[0] != self.coefficients.shape[0]:
            raise ValueError("exponents and coefficients disagree on number of terms")
        self.chunk = chunk
        self._emax = int(self.exponents.max()) if self.exponents.size else 0

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """x: (n, k) input variables -> (n,) output."""
        x = np.asarray(x, dtype=float)
        n, k = x.shape
        if k != self.exponents.shape[1]:
            raise ValueError(f"expected {self.exponents.shape[1]} variables, got {k}")
        out = np.empty(n)
        powers = np.arange(self._emax + 1)
        for lo in range(0, n, self.chunk):
            xs = x[lo:lo + self.chunk]
            terms = np.ones((xs.shape[0], self.exponents.shape[0]))
            for j in range(k):
                table = xs[:, j, None] ** powers            # (m, emax+1)
                terms *= table[:, self.exponents[:, j]]
            out[lo:lo + self.chunk] = terms @ self.coefficients
        return out
