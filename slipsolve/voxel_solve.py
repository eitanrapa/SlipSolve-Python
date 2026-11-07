"""The voxels' strain equation without forming `Gamma`: FFT convolution and Krylov.

`slipsolve.heterogeneity.HeterogeneousEngine` solves `(I + Gamma D) eps = E` for the
strain at the voxels. Its dense path assembles `Gamma` (`(6 n)^2`) and factors it, which
stops at ~10k voxels on 128 GB. On a regular grid in a layered reference `Gamma` is
translation invariant laterally: the block between two depth rows depends only on the
lateral offset, so applying it is a 2-D convolution per pair of rows, an FFT product
(`ConvolutionGamma`), and the equation is solved by GMRES (`gmres`) with each
voxel's own `6 x 6` block as the preconditioner. Memory is `O(n)` in the voxels apart from
the kernels, `O(nx ny nz^2)`.

The kernels are the dense path's numbers: one `slipsolve.heterogeneity.VoxelSource`
per source row, at every lateral offset of the active voxels' bounding box (the dense path
evaluates only the offsets that occur between active voxels). So the two paths differ only
by the FFT's rounding and the solve's tolerance (`tests/test_slip_voxel_solve.py`).
"""
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import scipy.fft as sfft

from .heterogeneity import VoxelSource, _steps

#: Relative residual `gmres` stops at, by default: far below the voxels' own
#: discretisation error (about 1e-2 of the heterogeneity's effect at 1 km cells).
GMRES_TOLERANCE = 1e-6

#: Krylov vectors kept before a restart.
GMRES_RESTART = 40

#: Most matrix-vector products a column may take.
GMRES_MAX_ITER = 400

#: Right-hand sides solved together (one FFT product serves the block).
SOLVE_BLOCK = 32

#: Threads the kernels' batched product is split over, in chunks of frequencies (None: every CPU). numpy runs a
#: batched `matmul` one small product at a time, effectively on one thread whatever BLAS's own count, so splitting it
#: over a thread pool scales with the threads. BLAS must then run at one thread inside the pool (concurrent
#: multithreaded OpenBLAS calls are slower than serial ones), so the pool holds it there while it runs
#: (`_blas_control`); without that control the product stays serial.
PRODUCT_THREADS = None


def _blas_control():
    """OpenBLAS's `(get, set)` thread-count functions, found through ctypes among the loaded libraries, or None."""
    import ctypes

    try:
        lib = ctypes.CDLL(None)
        get, put = lib.openblas_get_num_threads, lib.openblas_set_num_threads
    except (OSError, AttributeError):
        return None
    get.restype, get.argtypes = ctypes.c_int, []
    put.restype, put.argtypes = None, [ctypes.c_int]
    return get, put


def batched_matmul(a, b, threads=PRODUCT_THREADS):
    """`a @ b` over their leading axes (both C-contiguous), split in chunks of the leading axes over a thread pool.

    Each chunk is numpy's own `matmul` on one BLAS thread (the count is restored after), so the result is the
    serial product's, chunk by chunk. Serial when `threads` is 1, the leading axes are short, or BLAS's thread count
    cannot be controlled (`_blas_control`).
    """
    threads = (os.cpu_count() or 1) if threads is None else int(threads)
    lead = a.shape[:-2]
    n = int(np.prod(lead)) if lead else 1
    control = _blas_control() if threads > 1 and n >= 2 * threads else None
    if control is None:
        return np.matmul(a, b)
    af = a.reshape(n, *a.shape[-2:])
    bf = b.reshape(n, *b.shape[-2:])
    out = np.empty(lead + (a.shape[-2], b.shape[-1]), dtype=np.result_type(a, b))
    of = out.reshape(n, *out.shape[-2:])
    edges = np.linspace(0, n, threads + 1).astype(int)

    def work(k):
        np.matmul(af[edges[k]:edges[k + 1]], bf[edges[k]:edges[k + 1]], out=of[edges[k]:edges[k + 1]])

    get, put = control
    before = get()
    put(1)
    try:
        with ThreadPoolExecutor(threads) as pool:
            list(pool.map(work, range(threads)))
    finally:
        put(before)
    return out


class ConvolutionGamma:
    """`Gamma` of a `slipsolve.heterogeneity.VoxelModel` as one 2-D convolution per pair of depth rows.

    `matvec(m)` is `Gamma @ m` for `m` of shape `(6 n,)` or `(6 n, k)`, `n` the
    active voxels in the model's order: the moment densities scattered onto the bounding box
    of the active voxels, `rfft2` laterally, one batched product with the kernels of every
    pair of rows, inverse transform, gathered back. The kernels take
    `px * (py // 2 + 1) * (6 nz)^2` complex numbers (`nbytes`), `px >= 2 nx - 1` and
    `py >= 2 ny - 1` the padded lateral sizes.

    `workers` > 1 computes the kernels' pairs of rows in that many processes
    (`slipsolve.layered_tde.engine_pool`); `fft_workers` threads the transforms and `product_threads` the
    kernels' batched product (`batched_matmul`; None: every CPU).
    """

    def __init__(self, voxels, engine, log=None, workers=None, fft_workers=-1, product_threads=PRODUCT_THREADS):
        from .heterogeneity import _voxel_task
        from .layered_tde import engine_pool

        self.voxels = voxels
        self.fft_workers = fft_workers
        self.product_threads = product_threads
        n = voxels.n_active
        if n == 0:
            raise ValueError("no active voxels")
        dx, dy = _steps(voxels)
        index = voxels.index
        lo = index.min(axis=0)
        self.rows = np.unique(index[:, 2])
        self.shape = (int(index[:, 0].max() - lo[0] + 1), int(index[:, 1].max() - lo[1] + 1), int(self.rows.size))
        nx, ny, nz = self.shape
        self.ix = index[:, 0] - lo[0]
        self.iy = index[:, 1] - lo[1]
        self.iz = np.searchsorted(self.rows, index[:, 2])
        self.channel = 6 * self.iz[:, None] + np.arange(6)[None, :]               # (n, 6)
        self.padded = (sfft.next_fast_len(2 * nx - 1, real=True), sfft.next_fast_len(2 * ny - 1, real=True))
        px, py = self.padded

        di, dj = np.meshgrid(np.arange(-(nx - 1), nx), np.arange(-(ny - 1), ny), indexing="ij")
        di, dj = di.ravel(), dj.ravel()
        x_edges, y_edges, z_edges = voxels.x_edges, voxels.y_edges, voxels.z_edges
        tasks = []
        for s, iz_s in enumerate(self.rows):
            bounds = (x_edges[:2], y_edges[:2], z_edges[iz_s:iz_s + 2])
            centre = 0.5 * np.array([x_edges[0] + x_edges[1], y_edges[0] + y_edges[1]])
            for r, iz_r in enumerate(self.rows):
                z = np.full(di.size, -0.5 * (z_edges[iz_r] + z_edges[iz_r + 1]))
                tasks.append((s, r, bounds, centre[0] + di * dx, centre[1] + dj * dy, z))
        #: `(px, py // 2 + 1, 6 nz, 6 nz)`: `[kx, ky, 6 r + a, 6 s + b]`, strain `a` in row `r`
        #: per moment density `b` in row `s`.
        self.kernels = np.zeros((px, py // 2 + 1, 6 * nz, 6 * nz), dtype=complex)
        zero = int(np.flatnonzero((di == 0) & (dj == 0))[0])
        row_self = np.empty((nz, 6, 6))
        t0 = time.time()

        def place(task, field):                                   # field (n_off, 6 moment b, 6 strain a)
            s, r = task[:2]
            if s == r:
                row_self[r] = field[zero].T
            k = np.zeros((px, py, 6, 6))
            k[di % px, dj % py] = np.swapaxes(field, 1, 2)          # [.., a, b]
            self.kernels[:, :, 6 * r:6 * r + 6, 6 * s:6 * s + 6] = sfft.rfftn(k, axes=(0, 1),
                                                                             workers=self.fft_workers)

        pool = engine_pool(engine, workers)
        if pool is None:
            sources = {}
            for task in tasks:
                s, bounds = task[0], task[2]
                if s not in sources:
                    sources[s] = VoxelSource(engine, *bounds)
                place(task, sources[s].strain(*task[3:]))
        else:
            from .layered_tde import pool_map

            with pool:
                for k, field in pool_map(pool, _voxel_task, ((k, (task[2], *task[3:], True)) for k, task in enumerate(tasks))):
                    place(tasks[k], field)
        if log:
            log(f"Gamma kernels: {nz} rows x {di.size} offsets a pair, {nx} x {ny} -> {px} x {py} padded, "
                f"{self.nbytes / 1e9:.2f} GB [{time.time() - t0:.0f}s]")
        #: Each voxel's own `6 x 6` block of `Gamma` (`[a, b]`), its row's zero offset.
        self.self_blocks = row_self[self.iz]

    @property
    def nbytes(self):
        return self.kernels.nbytes

    @property
    def size(self):
        """Unknowns, `6 n`."""
        return 6 * self.ix.size

    def matvec(self, m):
        """`Gamma @ m`, `m` `(6 n,)` or `(6 n, k)`."""
        m = np.asarray(m, dtype=float)
        single = m.ndim == 1
        m = m.reshape(self.ix.size, 6, -1)
        k = m.shape[-1]
        nx, ny, nz = self.shape
        px, py = self.padded
        grid = np.zeros((nx, ny, 6 * nz, k))
        grid[self.ix[:, None], self.iy[:, None], self.channel] = m
        spectrum = sfft.rfftn(grid, s=(px, py), axes=(0, 1), workers=self.fft_workers)
        del grid
        product = batched_matmul(self.kernels, spectrum, getattr(self, "product_threads", PRODUCT_THREADS))
        del spectrum
        field = sfft.irfftn(product, s=(px, py), axes=(0, 1), workers=self.fft_workers)
        out = field[self.ix[:, None], self.iy[:, None], self.channel]              # (n, 6, k)
        out = out.reshape(-1, k)
        return out[:, 0] if single else out

    def dense(self):
        """`Gamma` as a dense `(6 n, 6 n)` array, column by column (small grids: tests)."""
        out = np.empty((self.size, self.size))
        eye = np.eye(self.size)
        for start in range(0, self.size, 256):
            out[:, start:start + 256] = self.matvec(eye[:, start:start + 256])
        return out


class StrainSystem:
    """`(I + Gamma D) eps = E` with `Gamma` a `ConvolutionGamma`, solved by `gmres`.

    `D` is the voxels' stiffness departure (`slipsolve.heterogeneity.stiffness_block`),
    the moment density being `-D eps`. Right-preconditioned by each voxel's own block,
    `P_v = I + Gamma_vv D_v`: the Eshelby self-interaction, leaving GMRES the voxels'
    interactions with each other.
    """

    def __init__(self, gamma, log=None):
        from .heterogeneity import stiffness_block

        self.gamma = gamma
        v = gamma.voxels
        self.d = stiffness_block(v.dlam[v.active], v.dmu[v.active])                 # (n, 6, 6)
        p = np.eye(6)[None] + np.einsum("nab,nbc->nac", gamma.self_blocks, self.d)
        self.p_inv = np.linalg.inv(p)
        self.log = log
        self.iterations = []

    def apply(self, eps):
        """`(I + Gamma D) eps` for `eps` `(6 n, k)`."""
        n = self.d.shape[0]
        e = eps.reshape(n, 6, -1)
        moment = np.einsum("nab,nbk->nak", self.d, e).reshape(6 * n, -1)
        return eps + self.gamma.matvec(moment)

    def precondition(self, y):
        n = self.d.shape[0]
        return np.einsum("nab,nbk->nak", self.p_inv, y.reshape(n, 6, -1)).reshape(6 * n, -1)

    def solve(self, rhs, tol=GMRES_TOLERANCE, restart=GMRES_RESTART, max_iter=GMRES_MAX_ITER, block=SOLVE_BLOCK):
        """`eps` for each column of `rhs` `(6 n, cols)`, in blocks of `block` columns."""
        rhs = np.asarray(rhs, dtype=float)
        single = rhs.ndim == 1
        rhs = rhs.reshape(rhs.shape[0], -1)
        out = np.empty_like(rhs)
        t0 = time.time()
        for start in range(0, rhs.shape[1], block):
            cols = slice(start, start + block)
            y, its = gmres(lambda v: self.apply(self.precondition(v)), rhs[:, cols], tol=tol, restart=restart,
                           max_iter=max_iter)
            out[:, cols] = self.precondition(y)
            self.iterations.extend(its.tolist())
        if self.log:
            its = np.asarray(self.iterations[-rhs.shape[1]:])
            self.log(f"GMRES: {rhs.shape[1]} columns of {rhs.shape[0]} unknowns, {its.min()}-{its.max()} products "
                     f"(median {np.median(its):.0f}) to {tol:g} [{time.time() - t0:.0f}s]")
        return out[:, 0] if single else out


def gmres(operator, b, tol=GMRES_TOLERANCE, restart=GMRES_RESTART, max_iter=GMRES_MAX_ITER):
    """Restarted GMRES on every column of `b` `(n, k)` at once: `(x, products)`.

    `operator(v)` maps `(n, j)` to `(n, j)` and is applied to the columns not yet
    converged. A column stops when its residual is within `tol` of its right-hand side's
    norm (a zero column gives zero); one still above it after `max_iter` products raises,
    naming how far it got. Arnoldi by modified Gram-Schmidt, Givens rotations per column.

    The live columns are kept contiguous: a column that converges inside a cycle is finished
    there (its correction added to `x`) and dropped from the cycle's arrays, so the
    orthogonalisation runs on views. Indexing the live columns out of the full arrays at every
    step instead can cost more than the FFT products themselves; the iterates are the same.
    """
    b = np.asarray(b, dtype=float)
    n, k = b.shape
    x = np.zeros((n, k))
    norm_b = np.linalg.norm(b, axis=0)
    target = tol * norm_b
    products = np.zeros(k, dtype=int)
    todo = np.flatnonzero(norm_b > 0.0)

    def finish(basis, h, g, steps, cols):
        for c in range(cols.size):
            s = int(steps[c])
            if s:
                y = np.linalg.solve(np.triu(h[:s, :s, c]), g[:s, c])
                x[:, cols[c]] += np.tensordot(y, basis[:s, :, c], axes=(0, 0))

    while todo.size:
        started = np.any(x[:, todo])
        r = b[:, todo] - operator(x[:, todo]) if started else b[:, todo].copy()
        products[todo] += int(started)
        beta = np.linalg.norm(r, axis=0)
        done = beta <= target[todo]
        if done.all():
            break
        todo, r, beta = todo[~done], r[:, ~done], beta[~done]
        over = products[todo] >= max_iter
        if over.any():
            worst = int(todo[np.argmax(beta / norm_b[todo])])
            raise RuntimeError(f"GMRES: column {worst} took {products[worst]} products and is at a relative "
                               f"residual {np.max(beta / norm_b[todo]):.1e}, not {tol:g}")
        m = todo.size
        cols, aim, scale = todo.copy(), target[todo], beta.copy()
        basis = np.empty((restart + 1, n, m))
        np.divide(r, beta, out=basis[0])
        del r
        h = np.zeros((restart + 1, restart, m))
        cs = np.zeros((restart, m))
        sn = np.zeros((restart, m))
        g = np.zeros((restart + 1, m))
        g[0] = beta
        tmp = np.empty((n, m))
        for j in range(restart):
            w = operator(basis[j])
            products[cols] += 1
            for i in range(j + 1):
                hij = np.einsum("nm,nm->m", basis[i], w)
                np.multiply(basis[i], hij, out=tmp)
                w -= tmp
                h[i, j] = hij
            hnext = np.linalg.norm(w, axis=0)
            h[j + 1, j] = hnext
            np.divide(w, np.where(hnext > 0.0, hnext, 1.0), out=basis[j + 1])
            del w
            for i in range(j):                                       # earlier rotations on the new column
                a, c = h[i, j].copy(), h[i + 1, j].copy()
                h[i, j] = cs[i] * a + sn[i] * c
                h[i + 1, j] = -sn[i] * a + cs[i] * c
            a, c = h[j, j], h[j + 1, j]
            rho = np.hypot(a, c)
            rho = np.where(rho > 0.0, rho, 1.0)
            cs[j], sn[j] = a / rho, c / rho
            h[j, j] = rho
            h[j + 1, j] = 0.0
            g[j + 1] = -sn[j] * g[j]
            g[j] = cs[j] * g[j]
            converged = (np.abs(g[j + 1]) <= aim) | (hnext <= 1e-14 * scale)
            if converged.any() and j + 1 < restart:
                finish(basis, h, g, np.where(converged, j + 1, 0), cols)
                keep = ~converged
                if not keep.any():
                    cols = cols[:0]
                    break
                kept = np.empty((restart + 1, n, int(keep.sum())))
                kept[:j + 2] = basis[:j + 2][:, :, keep]
                basis = kept
                h, cs, sn, g = h[:, :, keep], cs[:, keep], sn[:, keep], g[:, keep]
                cols, aim, scale = cols[keep], aim[keep], scale[keep]
                tmp = np.empty((n, cols.size))
        else:
            j = restart - 1
        finish(basis, h, g, np.full(cols.size, j + 1), cols)
    return x, products


__all__ = ["ConvolutionGamma", "StrainSystem", "batched_matmul", "gmres", "GMRES_TOLERANCE", "SOLVE_BLOCK"]
