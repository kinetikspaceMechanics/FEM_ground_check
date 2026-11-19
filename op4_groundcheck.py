"""
Rigid-body grounding check directly on a Craig-Bampton OP4 (KAA), streaming.

Computes F = KAA * R for the six unit rigid-body motions about a reference
point, the strain energies E = 0.5 * R^T K R (should match Nastran GROUNDCHECK
A-set), and lists the boundary grids carrying the largest grounding forces.

Usage:
    VS Code: set OP4_FILE / ASM_FILE in USER SETTINGS below, then Run.
    Command line (optional, overrides the settings):
    python op4_groundcheck.py KINETIK_RISE_RFEM_OP4.op4 KINETIK_RISE_8J_RA_RFEM.asm
           [--ref -0.52 -0.181 -0.04179] [--top 25]

Assumptions (checked/printed where possible):
  * First matrix in the OP4 is KAA, ASCII, real double (header 1P,3E23.16).
  * A-set order = boundary GRIDs sorted by ID (6 DOF each), then SPOINTs
    (modal DOF, 1 each) sorted by ID - as in the f06 boundary sequence table.
  * GRID coordinates in the .asm are in the basic system (CP=0) and the
    displacement system is basic (CD=0). Grids with CP/CD != 0 are reported.
"""
import argparse
import os
import sys
import numpy as np

# ============================================================================
# USER SETTINGS - edit these and press Run in VS Code (F5 / Ctrl+F5)
# ============================================================================
OP4_FILE = r"D:\SCRATCH\019_FEM_PDR\011-ROBOTIC_ARM\SE_RENUM\KINETIK_RISE_RFEM_OP4.op4"
ASM_FILE = r"D:\SCRATCH\019_FEM_PDR\011-ROBOTIC_ARM\SE_RENUM\KINETIK_RISE_8J_RA_RFEM.asm"
REF_POINT = [-0.52, -0.181, -0.04179]   # location of grid 9999999 (basic, m)
TOP_N = 25                              # grids listed per motion
OUT_FILE = "groundcheck_op4.txt"        # written next to this script
# ============================================================================


class _Tee:
    """Print to the VS Code terminal and to the output file at the same time."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()


# ----------------------------------------------------------------- .asm parsing
def _fields_small(line):
    return [line[i:i + 8].strip() for i in range(0, 80, 8)]


def _fields_large(l1, l2):
    f = [l1[0:8].strip()] + [l1[i:i + 16].strip() for i in range(8, 72, 16)]
    f += [l2[i:i + 16].strip() for i in range(8, 72, 16)]
    return f


def _num(s):
    s = s.strip()
    if not s:
        return 0.0
    # Nastran short exponent forms: 1.5-3, -2.+4
    if 'E' not in s.upper() and 'D' not in s.upper():
        for k in range(len(s) - 1, 0, -1):
            if s[k] in '+-' and s[k - 1] not in 'eE':
                s = s[:k] + 'E' + s[k:]
                break
    return float(s.replace('D', 'E').replace('d', 'E'))


def read_asm(path):
    grids, spoints = {}, set()
    with open(path, errors='replace') as fh:
        lines = fh.read().splitlines()
    i = 0
    while i < len(lines):
        ln = lines[i]
        u = ln.upper()
        if u.startswith('$'):
            i += 1
            continue
        if u.startswith('GRID*'):
            l2 = lines[i + 1] if i + 1 < len(lines) else ''
            f = _fields_large(ln, l2)
            gid, cp = int(f[1]), int(f[2] or 0)
            xyz = [_num(f[3]), _num(f[4]), _num(f[5])]
            cd = int(f[6] or 0) if len(f) > 6 and f[6] else 0
            grids[gid] = (cp, xyz, cd)
            i += 2
            continue
        if u.startswith('GRID'):
            f = [x.strip() for x in ln.split(',')] if ',' in ln else _fields_small(ln)
            f += [''] * 10
            gid, cp = int(f[1]), int(f[2] or 0)
            xyz = [_num(f[3]), _num(f[4]), _num(f[5])]
            cd = int(f[6] or 0)
            grids[gid] = (cp, xyz, cd)
        elif u.startswith('SPOINT'):
            f = [x.strip() for x in ln.split(',')] if ',' in ln else _fields_small(ln)
            vals = [x for x in f[1:] if x]
            if 'THRU' in [v.upper() for v in vals]:
                a, b = int(vals[0]), int(vals[2])
                spoints.update(range(a, b + 1))
            else:
                spoints.update(int(v) for v in vals)
        i += 1
    return grids, sorted(spoints)


# ----------------------------------------------------------------- OP4 streaming
def stream_kaa(path, ndof, R):
    """Accumulate F = K R column by column. Handles full or lower-triangular
    symmetric storage and multiple string records per column."""
    Fcol = np.zeros((ndof, 6))      # sum_c K[:,c] R[c,:]  (stored part)
    Ftr = np.zeros((ndof, 6))       # transpose contributions (off-diagonal)
    lower_only = True
    nstored = nzero = 0
    with open(path, 'r') as fh:
        hdr = fh.readline()
        # fixed format: 4 x I8, then name A8 (e.g. "       2KAA     1P,3E23.16")
        ncol, nrow, form, typ = (int(hdr[i:i + 8]) for i in range(0, 32, 8))
        name = hdr[32:40]
        print(f'OP4 matrix: {name.strip()}  {nrow} x {ncol}  form={form} type={typ}')
        print(f'Header: {hdr.rstrip()}')
        if ncol != ndof or nrow != ndof:
            sys.exit(f'Size mismatch: OP4 {nrow}x{ncol} vs a-set from .asm {ndof}')
        while True:
            rec = fh.readline()
            if not rec:
                break
            p = rec.split()
            if len(p) < 3:
                continue
            c, r0, nw = int(p[0]), int(p[1]), int(p[2])
            # read nw values, fixed width 23
            nlines = -(-nw // 3)                     # 3 values per line (E23.16)
            buf = ''.join(fh.readline()[:69] for _ in range(nlines))[:nw * 23]
            if c > ncol:            # end-of-matrix record
                break
            v = np.frombuffer(buf.encode(), dtype='S23').astype(np.float64)
            nstored += nw
            nzero += int(np.count_nonzero(v == 0.0))
            ci = c - 1
            rows = np.arange(r0 - 1, r0 - 1 + nw)
            if c > 1 and r0 < c:
                lower_only = False
            Fcol[rows] += np.outer(v, R[ci])
            off = rows != ci
            Ftr[ci] += v[off] @ R[rows[off]]
            if c % 500 == 0:
                sys.__stdout__.write(f'  column {c}/{ncol}\r')
                sys.__stdout__.flush()
    print()
    print(f'Stored terms: {nstored:,}  of which exact zeros: {nzero:,}  '
          f'(dense full matrix would be {ncol*nrow:,})')
    F = Fcol + Ftr if lower_only else Fcol
    print('Storage detected:', 'lower triangle (symmetrised)' if lower_only else 'full columns')
    return F


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('op4', nargs='?', default=OP4_FILE)
    ap.add_argument('asm', nargs='?', default=ASM_FILE)
    ap.add_argument('--ref', nargs=3, type=float, default=REF_POINT)
    ap.add_argument('--top', type=int, default=TOP_N)
    a = ap.parse_args()

    for f in (a.op4, a.asm):
        if not os.path.isfile(f):
            sys.exit(f'File not found: {f}\nEdit OP4_FILE / ASM_FILE at the top of the script.')

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), OUT_FILE)
    out_fh = open(out_path, 'w')
    sys.stdout = _Tee(sys.__stdout__, out_fh)
    print(f'OP4: {a.op4}\nASM: {a.asm}\nReference point: {a.ref}\n')

    grids, spoints = read_asm(a.asm)
    gids = sorted(grids)
    bad = [g for g in gids if grids[g][0] != 0 or grids[g][2] != 0]
    print(f'.asm: {len(gids)} GRIDs, {len(spoints)} SPOINTs')
    if bad:
        print(f'WARNING: {len(bad)} grids with CP/CD != 0 (first: {bad[:5]}). '
              'Rotations will be wrong for those; translations still valid only if CD=0.')

    ndof = 6 * len(gids) + len(spoints)
    ref = np.array(a.ref)
    R = np.zeros((ndof, 6))
    xyz = np.array([grids[g][1] for g in gids])
    for k, g in enumerate(gids):
        d = xyz[k] - ref
        b = 6 * k
        for t in range(3):
            R[b + t, t] = 1.0                     # translations
            R[b + 3 + t, 3 + t] = 1.0             # rotation dofs
        # u = theta x d  ->  columns for theta_x, theta_y, theta_z
        R[b:b + 3, 3] = [0.0, -d[2], d[1]]
        R[b:b + 3, 4] = [d[2], 0.0, -d[0]]
        R[b:b + 3, 5] = [-d[1], d[0], 0.0]
    # modal SPOINT rows stay zero

    F = stream_kaa(a.op4, ndof, R)
    KR = R.T @ F
    E = 0.5 * np.diag(KR)
    lab = ['T1', 'T2', 'T3', 'R1', 'R2', 'R3']
    print('\nRigid-body strain energy 0.5*R^T K R  (compare with GROUNDCHECK A-set):')
    for j in range(6):
        print(f'  {lab[j]}: {E[j]: .6E}')
    print('\nK_R = R^T K R (6x6):')
    with np.printoptions(precision=3, suppress=False, linewidth=140):
        print(KR)
    print('eigenvalues of K_R:', np.linalg.eigvalsh(0.5 * (KR + KR.T)))

    print('\nModal DOF grounding forces (should be ~0 for CB): max |F| =',
          np.abs(F[6 * len(gids):]).max() if spoints else 'n/a')

    for j in range(6):
        Fg = F[:6 * len(gids), j].reshape(-1, 6)
        fmag = np.linalg.norm(Fg[:, :3], axis=1)
        mmag = np.linalg.norm(Fg[:, 3:], axis=1)
        order = np.argsort(-fmag)[:a.top]
        tot = fmag.sum()
        print(f'\n=== Motion {lab[j]}: top {a.top} grids by |force| '
              f'(share of sum |F| = {fmag[order].sum() / tot:.1%}) ===')
        print(f'{"GRID":>10} {"x":>9} {"y":>9} {"z":>9} {"|F| [N]":>12} {"|M| [Nm]":>12}')
        for k in order:
            x, y, z = xyz[k]
            print(f'{gids[k]:>10} {x:9.4f} {y:9.4f} {z:9.4f} {fmag[k]:12.4E} {mmag[k]:12.4E}')

    sys.stdout = sys.__stdout__
    out_fh.close()
    print(f'\nDone. Results saved to: {out_path}')


if __name__ == '__main__':
    main()
