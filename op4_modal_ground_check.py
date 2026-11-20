"""
Craig-Bampton OP4 model checks - directly from the delivered matrices (KAA, MAA).

  1. Rigid-body mass properties   R^T M R    (ECSS-E-ST-32-03C 5.4.2)
  2. Rigid-body strain energy     R^T K R    (5.4.3, = Nastran GROUNDCHECK A-set)
     + grounding forces per boundary grid
  3. Free-free modes              6 rigid + elastic, chi ratio  (5.7)
  4. Clamped modes                interface grids fixed (rigid panel)
     + modal participation factors, effective masses, fractions (= MEFFMASS)

Run from VS Code: edit USER SETTINGS, press Run. Results go to the console,
to <OUT_PREFIX>_report.txt and to CSV tables next to this script.

Requires: pip install numpy scipy
Memory:   KAA + MAA are held dense: 2 x n^2 x 8 bytes (n = 7548 -> ~0.9 GB),
          plus working copies for the eigen-solutions -> plan ~3 GB of free RAM.

Assumptions:
  * a-set order = boundary GRIDs sorted by ID (6 DOF each), then SPOINTs sorted
    (validated earlier: GROUNDCHECK energies matched MSC to 6 digits).
  * GRIDs in the .asm are in the basic system (CP = CD = 0).
  * Clamped case = all 6 DOF of the clamp grids fixed. This is equivalent to an
    RBE2 tying them to a fixed grid (the "rigid panel fixed" subcase).
"""
import csv
import math
import os
import re
import sys
import time

import numpy as np
from scipy.linalg import eigh

# ============================================================================
# USER SETTINGS
# ============================================================================
OP4_FILE = "KINETIK_RISE_RFEM_OP4.op4"
ASM_FILE = "KINETIK_RISE_8J_RA_RFEM.asm"

K_NAME = "KAA"
M_NAME = "MAA"

GROUND_REF = [-0.52, -0.181, -0.04179]   # GROUNDCHECK reference point (grid 9999999)
MASS_REF = [0.0, 0.0, 0.0]               # MEFFMASS / weight reference point (basic origin,
                                         # as OptiStruct GRDPNT=0 / MSC PARAM,GRDPNT,0)
TOP_N = 25                               # grids listed per motion in groundcheck

F_MAX_HZ = 2000.0                        # compute all modes up to this frequency
RUN_FREEFREE = True
RUN_CLAMPED = True

# Clamp grids: EITHER give them explicitly ...
CLAMP_GRIDS = []                         # e.g. list(range(10000001, 10000129))
# ... OR read the dependent grids of an RBE2 from the MSC deck
#     (works also if the RBE2 lines are commented out with '$')
CLAMP_FROM_DAT = "KINETIK_RISE_8J_RFEM.dat"
CLAMP_RBE2_EID = 9999999

OUT_PREFIX = "op4_modal_check"           # files written next to this script
# ============================================================================

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = ['T1', 'T2', 'T3', 'R1', 'R2', 'R3']


class _Tee:
    def __init__(self, *s):
        self.s = s

    def write(self, d):
        for x in self.s:
            x.write(d)

    def flush(self):
        for x in self.s:
            x.flush()


# ----------------------------------------------------------------- .asm / .dat
def _num(s):
    s = s.strip()
    if not s:
        return 0.0
    if 'E' not in s.upper() and 'D' not in s.upper():
        for k in range(len(s) - 1, 0, -1):
            if s[k] in '+-' and s[k - 1] not in 'eE':
                s = s[:k] + 'E' + s[k:]
                break
    return float(s.replace('D', 'E').replace('d', 'E'))


def _small(line):
    return [line[i:i + 8].strip() for i in range(0, 80, 8)]


def read_asm(path):
    grids, spoints = {}, set()
    lines = open(path, errors='replace').read().splitlines()
    i = 0
    while i < len(lines):
        ln = lines[i]
        u = ln.upper()
        if u.startswith('$'):
            i += 1
            continue
        if u.startswith('GRID*'):
            l2 = lines[i + 1] if i + 1 < len(lines) else ''
            f = [ln[0:8].strip()] + [ln[k:k + 16].strip() for k in range(8, 72, 16)]
            f += [l2[k:k + 16].strip() for k in range(8, 72, 16)]
            grids[int(f[1])] = (int(f[2] or 0), [_num(f[3]), _num(f[4]), _num(f[5])],
                                int(f[6] or 0) if len(f) > 6 and f[6] else 0)
            i += 2
            continue
        if u.startswith('GRID'):
            f = [x.strip() for x in ln.split(',')] if ',' in ln else _small(ln)
            f += [''] * 10
            grids[int(f[1])] = (int(f[2] or 0), [_num(f[3]), _num(f[4]), _num(f[5])],
                                int(f[6] or 0))
        elif u.startswith('SPOINT'):
            f = [x.strip() for x in ln.split(',')] if ',' in ln else _small(ln)
            v = [x for x in f[1:] if x]
            if 'THRU' in [x.upper() for x in v]:
                spoints.update(range(int(v[0]), int(v[2]) + 1))
            else:
                spoints.update(int(x) for x in v)
        i += 1
    return grids, sorted(spoints)


def read_rbe2_dependents(path, eid):
    """Dependent grids of RBE2 <eid> (small or free field). Also works if the card is
    commented out by putting '$' in front of each line."""
    out, active, off = [], False, 0
    for raw in open(path, errors='replace').read().splitlines():
        up = raw.upper()
        k = up.find('RBE2')
        if 0 <= k <= 3 and set(raw[:k]) <= {'$', ' '}:
            off = k
            ln = raw[off:]
            f = [x.strip() for x in ln.split(',')] if ',' in ln else _small(ln)
            active = (f[1] == str(eid))
            if active:
                out += [int(x) for x in f[4:9] if x.isdigit()]
            continue
        if not active:
            continue
        ln = raw[off:] if set(raw[:off]) <= {'$', ' '} else raw
        f = [x.strip() for x in ln.split(',')] if ',' in ln else _small(ln)
        if f[0] and f[0][0] not in '+*':
            active = False          # next card
            continue
        ints = [x for x in f[1:9] if x.isdigit()]
        if not ints:
            active = False
            continue
        out += [int(x) for x in ints]
    return out


# ----------------------------------------------------------------- OP4
def read_op4(path, names):
    """Read the requested matrices (dense) from an ASCII OP4. Others are skipped."""
    want = {n.upper() for n in names}
    mats = {}
    with open(path, 'r') as fh:
        while want - set(mats):
            hdr = fh.readline()
            if not hdr:
                break
            if not hdr.strip():
                continue
            ncol, nrow, form, typ = (int(hdr[i:i + 8]) for i in range(0, 32, 8))
            name = hdr[32:40].strip().upper()
            m = re.search(r'(\d+)E(\d+)\.(\d+)', hdr[40:])
            per, w = (int(m.group(1)), int(m.group(2))) if m else (3, 23)
            keep = name in want
            if keep:
                if typ not in (1, 2):
                    sys.exit(f'{name}: complex OP4 type {typ} not supported')
                print(f'Reading {name}: {nrow} x {ncol}, form {form}, type {typ}, '
                      f'format {per}E{w}')
                A = np.zeros((nrow, ncol))
                lower_only = True
            t0 = time.time()
            while True:
                rec = fh.readline().split()
                c, r0, nw = int(rec[0]), int(rec[1]), int(rec[2])
                nlines = -(-nw // per)
                if keep:
                    buf = ''.join(fh.readline()[:per * w] for _ in range(nlines))[:nw * w]
                else:
                    for _ in range(nlines):
                        fh.readline()
                if c > ncol:
                    break
                if keep:
                    if r0 <= 0:
                        sys.exit(f'{name}: BIGMAT/sparse OP4 strings not supported')
                    A[r0 - 1:r0 - 1 + nw, c - 1] = np.frombuffer(
                        buf.encode(), dtype=f'S{w}').astype(np.float64)
                    if c > 1 and r0 < c:
                        lower_only = False
                    if c % 1000 == 0:
                        sys.__stdout__.write(f'  {name} column {c}/{ncol}\r')
                        sys.__stdout__.flush()
            if keep:
                if form == 6 and lower_only:
                    A = np.tril(A) + np.tril(A, -1).T
                print(f'  {name} done in {time.time() - t0:.0f} s '
                      f'({"lower triangle, symmetrised" if form == 6 and lower_only else "full columns"})'
                      f'          ')
                mats[name] = A
    missing = want - set(mats)
    if missing:
        sys.exit(f'Matrices not found in OP4: {sorted(missing)}')
    return mats


# ----------------------------------------------------------------- helpers
def rigid_body_matrix(xyz, n_spoints, ref):
    ng = len(xyz)
    R = np.zeros((6 * ng + n_spoints, 6))
    for k in range(ng):
        d = np.asarray(xyz[k]) - ref
        b = 6 * k
        for t in range(3):
            R[b + t, t] = 1.0
            R[b + 3 + t, 3 + t] = 1.0
        R[b:b + 3, 3] = [0.0, -d[2], d[1]]
        R[b:b + 3, 4] = [d[2], 0.0, -d[0]]
        R[b:b + 3, 5] = [-d[1], d[0], 0.0]
    return R


def mass_properties(MR, ref):
    """Mass, CoG (basic) and inertia about CoG from R^T M R about point ref."""
    m = MR[0, 0]
    S = MR[0:3, 3:6] / m           # = [[0, z, -y], [-z, 0, x], [y, -x, 0]], CoG rel. to ref
    c = np.array([S[1, 2], -S[0, 2], S[0, 1]])
    I_ref = MR[3:6, 3:6]
    I_cg = I_ref - m * (np.dot(c, c) * np.eye(3) - np.outer(c, c))
    return m, np.asarray(ref) + c, I_cg


def solve_modes(K, M, fmax, allow_negative, overwrite=False):
    """Generalised eigenproblem K phi = lam M phi; phi is mass-normalised."""
    lam_max = (2 * math.pi * fmax) ** 2
    lo = -1e6 if allow_negative else -1e-6
    t0 = time.time()
    lam, phi = eigh(K, M, subset_by_value=(lo, lam_max), driver='gvx',
                    check_finite=False, overwrite_a=overwrite, overwrite_b=overwrite)
    print(f'  {len(lam)} modes up to {fmax:g} Hz in {time.time() - t0:.0f} s')
    f = np.sign(lam) * np.sqrt(np.abs(lam)) / (2 * math.pi)
    return lam, f, phi


# ----------------------------------------------------------------- main
def main():
    out_txt = os.path.join(HERE, OUT_PREFIX + '_report.txt')
    fh_out = open(out_txt, 'w')
    sys.stdout = _Tee(sys.__stdout__, fh_out)
    print(f'OP4: {OP4_FILE}\nASM: {ASM_FILE}\n')

    grids, spoints = read_asm(ASM_FILE)
    gids = sorted(grids)
    xyz = np.array([grids[g][1] for g in gids])
    bad = [g for g in gids if grids[g][0] != 0 or grids[g][2] != 0]
    print(f'.asm: {len(gids)} GRIDs, {len(spoints)} SPOINTs')
    if bad:
        print(f'WARNING: {len(bad)} grids with CP/CD != 0 (first {bad[:5]}) - '
              'rotations/participations for those are not in basic.')
    nb, nq = 6 * len(gids), len(spoints)
    n = nb + nq

    mats = read_op4(OP4_FILE, [K_NAME, M_NAME])
    K, M = mats[K_NAME.upper()], mats[M_NAME.upper()]
    if K.shape != (n, n) or M.shape != (n, n):
        sys.exit(f'Size mismatch: matrices {K.shape}, a-set from .asm {n}')
    print(f'Symmetry check: max|K-K^T| = {np.abs(K - K.T).max():.3e}, '
          f'max|M-M^T| = {np.abs(M - M.T).max():.3e}')
    print(f'K_bq max |.| = {np.abs(K[:nb, nb:]).max():.3e} (CB: should be 0)\n')

    # ---------------- 1. weight check
    Rm = rigid_body_matrix(xyz, nq, np.array(MASS_REF))
    MR = Rm.T @ M @ Rm
    m, cg, I_cg = mass_properties(MR, MASS_REF)
    print('=' * 78)
    print(f'1. RIGID-BODY MASS PROPERTIES  (reference point {MASS_REF})')
    print('=' * 78)
    with np.printoptions(precision=6, suppress=False, linewidth=140):
        print('R^T M R:\n', MR)
    print(f'Mass = {m:.6E}   CoG = ({cg[0]:.6E}, {cg[1]:.6E}, {cg[2]:.6E})')
    print('Translational masses (should be equal):', np.diag(MR)[:3])
    with np.printoptions(precision=6, linewidth=140):
        print('Inertia about CoG:\n', I_cg)

    # ---------------- 2. groundcheck
    Rg = rigid_body_matrix(xyz, nq, np.array(GROUND_REF))
    F = K @ Rg
    KR = Rg.T @ F
    E = 0.5 * np.diag(KR)
    print('\n' + '=' * 78)
    print(f'2. RIGID-BODY STRAIN ENERGY 0.5*R^T K R  (reference {GROUND_REF})')
    print('=' * 78)
    for j in range(6):
        flag = 'OK' if abs(E[j]) <= 1e-3 else 'EXCEEDS 1e-3 J'
        print(f'  {LAB[j]}: {E[j]: .6E}   {flag}')
    with np.printoptions(precision=3, linewidth=140):
        print('K_R = R^T K R:\n', KR)
        print('eigenvalues of K_R:', np.linalg.eigvalsh(0.5 * (KR + KR.T)))
    print('Max |grounding force| on modal DOFs:', np.abs(F[nb:]).max() if nq else 'n/a')
    for j in range(6):
        Fg = F[:nb, j].reshape(-1, 6)
        fm = np.linalg.norm(Fg[:, :3], axis=1)
        mm = np.linalg.norm(Fg[:, 3:], axis=1)
        o = np.argsort(-fm)[:TOP_N]
        print(f'\n  {LAB[j]}: top {TOP_N} grids by |F| (share {fm[o].sum() / fm.sum():.1%})')
        print(f'  {"GRID":>10} {"x":>9} {"y":>9} {"z":>9} {"|F| [N]":>12} {"|M| [Nm]":>12}')
        for k in o:
            print(f'  {gids[k]:>10} {xyz[k, 0]:9.4f} {xyz[k, 1]:9.4f} {xyz[k, 2]:9.4f} '
                  f'{fm[k]:12.4E} {mm[k]:12.4E}')
    del F

    # ---------------- 3. free-free modes
    if RUN_FREEFREE:
        print('\n' + '=' * 78)
        print('3. FREE-FREE MODES')
        print('=' * 78)
        lam, f, _ = solve_modes(K, M, F_MAX_HZ, allow_negative=True)
        n_show = min(len(f), 30)
        print(f'  {"mode":>4} {"eigenvalue":>14} {"freq [Hz]":>14}')
        for i in range(n_show):
            print(f'  {i + 1:>4} {lam[i]:14.6E} {f[i]:14.6E}')
        if len(f) >= 7:
            fr = np.abs(f[:6]).max()
            chi = fr / f[6]
            print(f'\n  [dense solver]  max |rigid-mode freq| = {fr:.4E} Hz, chi = {chi:.4E}')
            print('  (the dense solver resolves near-zero eigenvalues only to ~eps*||K||/min(M);'
                  ' identical rigid values = solver floor, not the model)')
            # Rayleigh-Ritz on the exact rigid-body subspace: accurate rigid eigenvalues
            MRg = Rg.T @ M @ Rg
            lam_rr = eigh(0.5 * (KR + KR.T), MRg, eigvals_only=True)
            f_rr = np.sign(lam_rr) * np.sqrt(np.abs(lam_rr)) / (2 * math.pi)
            fr2 = np.abs(f_rr).max()
            chi2 = fr2 / f[6]
            print('  [Rayleigh-Ritz on R]  rigid eigenvalues:', np.array2string(lam_rr, precision=3))
            print(f'  Max |rigid-mode freq| = {fr2:.4E} Hz   '
                  f'({"OK" if fr2 <= 5e-3 else "EXCEEDS"} 5e-3 Hz, 5.7a)')
            print(f'  First elastic mode    = {f[6]:.4E} Hz')
            print(f'  chi = {chi2:.4E}   ({"OK" if chi2 <= 1e-4 else "EXCEEDS"} 1e-4, 5.7b)')
            if (lam_rr < 0).any():
                print(f'  NOTE: {int((lam_rr < 0).sum())} negative rigid eigenvalue(s) '
                      '(round-off level if |lam| is tiny)')
        with open(os.path.join(HERE, OUT_PREFIX + '_freefree.csv'), 'w', newline='') as fc:
            w = csv.writer(fc)
            w.writerow(['mode', 'eigenvalue', 'freq_Hz'])
            for i in range(len(f)):
                w.writerow([i + 1, f'{lam[i]:.8E}', f'{f[i]:.8E}'])

    # ---------------- 4. clamped modes + MEFFMASS
    if RUN_CLAMPED:
        print('\n' + '=' * 78)
        print('4. CLAMPED MODES + MODAL EFFECTIVE MASS')
        print('=' * 78)
        cg_ids = list(CLAMP_GRIDS)
        if not cg_ids and CLAMP_FROM_DAT:
            cg_ids = read_rbe2_dependents(CLAMP_FROM_DAT, CLAMP_RBE2_EID)
            print(f'  Clamp grids from RBE2 {CLAMP_RBE2_EID} in {os.path.basename(CLAMP_FROM_DAT)}')
        idx = {g: k for k, g in enumerate(gids)}
        missing = [g for g in cg_ids if g not in idx]
        cg_ids = [g for g in cg_ids if g in idx]
        print(f'  {len(cg_ids)} clamp grids ({len(missing)} not boundary grids, ignored)'
              + (f': {missing[:5]}' if missing else ''))
        if not cg_ids:
            print('  No clamp grids -> clamped case skipped.')
        else:
            c_dofs = np.concatenate([np.arange(6 * idx[g], 6 * idx[g] + 6) for g in cg_ids])
            mask = np.ones(n, bool)
            mask[c_dofs] = False
            f_dofs = np.where(mask)[0]
            Kff = K[np.ix_(f_dofs, f_dofs)]
            Mff = M[np.ix_(f_dofs, f_dofs)]
            lam, f, phi = solve_modes(Kff, Mff, F_MAX_HZ, allow_negative=False,
                                      overwrite=True)
            del Kff, Mff
            # Base excitation: the clamp moves rigidly (about MASS_REF) and the free DOFs
            # follow rigidly -> load vector (M_ff R_f + M_fc R_c) = (M R)[free rows].
            # phi is mass-normalised -> Gamma = phi^T (M R)_f ,  Meff = Gamma^2
            MRf = (M @ Rm)[f_dofs]
            G = phi.T @ MRf
            Meff = G ** 2
            tot = np.diag(MR)
            frac = Meff / tot
            cum = np.cumsum(frac, axis=0)
            print(f'  First clamped mode: {f[0]:.4E} Hz')
            print(f'\n  Total rigid-body mass/inertia (reference {MASS_REF}):', tot)
            print(f'\n  {"mode":>4} {"freq [Hz]":>11} ' + ' '.join(f'{x:>9}' for x in LAB)
                  + '   <- effective mass fraction')
            for i in range(len(f)):
                print(f'  {i + 1:>4} {f[i]:11.4E} ' + ' '.join(f'{x:9.3E}' for x in frac[i]))
            print(f'  {"SUM":>4} {"":>11} ' + ' '.join(f'{x:9.4f}' for x in cum[-1]))
            for t in (0.8, 0.9):
                reach = [int(np.argmax(cum[:, j] >= t)) + 1 if cum[-1, j] >= t else None
                         for j in range(6)]
                print(f'  mode reaching {t:.0%}: ' + '  '.join(
                    f'{LAB[j]}={reach[j] if reach[j] else "-"}' for j in range(6)))
            with open(os.path.join(HERE, OUT_PREFIX + '_clamped_meff.csv'), 'w',
                      newline='') as fc:
                w = csv.writer(fc)
                w.writerow(['mode', 'freq_Hz']
                           + [f'Gamma_{x}' for x in LAB]
                           + [f'Meff_{x}' for x in LAB]
                           + [f'frac_{x}' for x in LAB]
                           + [f'cum_{x}' for x in LAB])
                for i in range(len(f)):
                    w.writerow([i + 1, f'{f[i]:.6E}']
                               + [f'{x:.6E}' for x in G[i]]
                               + [f'{x:.6E}' for x in Meff[i]]
                               + [f'{x:.6E}' for x in frac[i]]
                               + [f'{x:.6E}' for x in cum[i]])

    sys.stdout = sys.__stdout__
    fh_out.close()
    print(f'\nDone. Report: {out_txt}')


if __name__ == '__main__':
    main()
