"""Functional model of the GainP associative processor (Levi et al., J. Syst. Archit. 172, 2026, 103632).

Modeled operations, as described in Sections 3.2 to 3.5 and Fig. 8 of the GainP paper:
  compare(cube)   one cycle. Searches the unmasked rows of every column for a pattern (rows not in the cube
                  are masked). Matching columns set their tag register, which accumulates (OR) over successive
                  compares until the next write.
  write_match(r)  one cycle. Writes the accumulated tag of every column into row r and resets the tags.
  write_row(r,v)  one cycle. Writes an arbitrary bit per column into row r (ordinary memory write).
Data are stored column-major: one entry per column, one bit per row. Columns operate in lockstep.

Energy model (our modeling choices, stated in the paper):
  * Cycle time 2 ns (GainP paper, Section 5.1).
  * Compare: 10.4 fJ per column (match line) per compare (GainP paper, Section 5.1). The paper splits compare
    power into predischarge 34%, evaluation 45%, sensing 9% and tagging 12% (Fig. 14b). A match line charges
    only on a mismatch, so we charge the full 10.4 fJ to a mismatching column and only the sensing and tagging
    share (21%, 2.184 fJ) to a matching column.
  * Write: 10.4 fJ per written bit, independent of the data (value given by the GainP authors: write energy per
    bit is about equal to compare energy per bit).
"""
import numpy as np

T_CYCLE = 2e-9
E_CMP_MIS = 10.4e-15
E_CMP_MATCH = 0.21 * 10.4e-15
E_WRITE = 10.4e-15


class GainP:
    def __init__(self, nrows, ncols, cols_per_group=1, log=True):
        """cols_per_group: number of adjacent columns whose activity is summed into one trace (one group)."""
        assert ncols % cols_per_group == 0
        self.S = np.zeros((nrows, ncols), dtype=bool)
        self.tag = np.zeros(ncols, dtype=bool)
        self.g = cols_per_group; self.ngroups = ncols // cols_per_group
        self.log = log
        self.kind = []          # 'C' compare, 'W' write, per cycle
        self.mis = []           # per compare cycle: mismatching columns per group (int array)
        self.n_cmp = 0; self.n_wr = 0
        self.E = np.zeros(ncols, np.float64)   # total energy per column (J)

    def compare(self, cube):
        m = np.ones(self.S.shape[1], dtype=bool)
        for r, b in cube:
            m &= self.S[r] if b else ~self.S[r]
        self.tag |= m
        self.n_cmp += 1
        self.E += np.where(m, E_CMP_MATCH, E_CMP_MIS)
        if self.log:
            self.kind.append('C')
            self.mis.append(self.g - m.reshape(self.ngroups, self.g).sum(1).astype(np.int32))

    def write_match(self, r):
        self.S[r] = self.tag; self.tag = np.zeros_like(self.tag); self._w()

    def write_row(self, r, v):
        self.S[r] = v; self._w()

    def _w(self):
        self.n_wr += 1
        self.E += E_WRITE
        if self.log:
            self.kind.append('W'); self.mis.append(None)

    def func(self, out_row, cubes):
        """Evaluate a Boolean function given as a cover (list of cubes, each a list of (row, bit)) and write it."""
        for c in cubes:
            self.compare(c)
        self.write_match(out_row)

    def energy_trace(self):
        """Energy per cycle (J) for every group: array (ngroups, ncycles)."""
        out = np.empty((self.ngroups, len(self.kind)), np.float64)
        for c, (k, m) in enumerate(zip(self.kind, self.mis)):
            if k == 'W':
                out[:, c] = self.g * E_WRITE
            else:
                out[:, c] = m * E_CMP_MIS + (self.g - m) * E_CMP_MATCH
        return out

    def mismatch_matrix(self):
        """Mismatch counts per group and cycle (writes: -1). int16 array (ngroups, ncycles)."""
        out = np.full((self.ngroups, len(self.kind)), -1, np.int16)
        for c, m in enumerate(self.mis):
            if m is not None:
                out[:, c] = m
        return out
