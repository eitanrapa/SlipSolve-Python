"""Convergence order of this package's triangle quadrature against the reference's.

This package cuts each triangle into ``k**2`` equal-weight point sources at the
centroids of **all** ``k**2`` congruent sub-triangles -- ``k(k+1)/2`` upright plus
``k(k-1)/2`` inverted. That is second-order accurate.

SlipSolve-Curve uses ``interpolate_triangle2(v1,v2,v3,15)``, read from source at
`external/legacy_matlab/Inversion/layered_green/`. Its own header says
*"points strictly INSIDE a 3D triangle ... Edge/vertex points are excluded"*, with
``#points = (n-1)(n-2)/2`` -- which is where **91** comes from at ``n = 15``.
`comb_green_nodes_xyz.m` weights them **equally** (``pmoment = area/npatch``) and
applies a tent via `taperTriangle`, so the rule reproduced here is faithful.

⚠️ **Excluding the boundary is the defect, and it is not a small one.** The points
``(i,j,k)/n`` with ``i,j,k >= 1`` are a *complete* lattice on the triangle shrunk
by ``(n-3)/n`` about its centroid. At ``n = 15`` that is 0.8 linear -- the rule
integrates over the inner **64% of the area** and never samples the rim. The
shrinkage dies only as ``3/n``, which caps the rule at first order however many
points are added.

⚠️ **Two earlier descriptions in this repo were wrong** and this script exists to
stop them recurring. `_triangle_quadrature`'s docstring said the reference takes
"only the upright" sub-triangles -- it does not, it takes interior lattice nodes.
And it said the error "stops falling however many points are added", "plateaus
around 5e-3" -- it does not plateau, it converges at first order. The apparent
plateau came from measuring on the real layered integrand, where EDGRN
table-interpolation error takes over and hides the continued slow convergence.

Tested against **exact** integrals, so nothing here is circular -- no high-order
rule stands in for the truth:

    int L1^a L2^b L3^c dA  =  2A * a! b! c! / (a+b+c+2)!

over the barycentric coordinates of a triangle of area ``A``.

⚠️ This measures the **rule**, not the adaptive order, and the two wins are
independent. Adapting the order is what makes most pairs cheap (56.7% of pairs on
the real problem need ``k=1``); the orientation is what makes each order accurate.
`bench_greens.py` measures the other half -- and measures cost, not accuracy.

    python scripts/bench_quadrature_rule.py
"""

import argparse
from math import factorial

import numpy as np

#: Reference triangle, area 1/2. Any triangle gives the same relative errors --
#: the rule is affine-invariant, since barycentric coordinates are.
_P = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

#: Monomials in the barycentric coordinates, as (a, b, c) exponents. Degree 2 and
#: up: a rule reproducing constants and linears exactly says nothing, and both
#: rules do reproduce those (their points are symmetric about the centroid).
_TESTS = [(2, 0, 0), (1, 1, 0), (3, 0, 0), (2, 1, 0), (1, 1, 1), (4, 0, 0), (2, 2, 0)]


def exact(a, b, c):
    """``int L1^a L2^b L3^c dA`` over the unit triangle (area 1/2)."""
    return factorial(a) * factorial(b) * factorial(c) / factorial(a + b + c + 2)


def sub_triangle_centroids(n):
    """This package: centroids of all ``n**2`` sub-triangles, equal weights."""
    points = []
    for a in range(n):
        for b in range(n - a):
            c = n - 1 - a - b
            points.append(((a + 1 / 3) * _P[0] + (b + 1 / 3) * _P[1]
                           + (c + 1 / 3) * _P[2]) / n)
            if c >= 1:
                points.append(((a + 2 / 3) * _P[0] + (b + 2 / 3) * _P[1]
                               + (c - 1 / 3) * _P[2]) / n)
    points = np.asarray(points)
    return points, np.full(len(points), 1.0 / len(points))


def reference_interior(n):
    """`interpolate_triangle2`: ``i+j+k=n``, ``i,j,k>=1``, at ``(i,j,k)/n``.

    Equal weights, as `comb_green_nodes_xyz.m` assigns them (``area/npatch``).
    ``(n-1)(n-2)/2`` points, so ``n=15`` gives the reference's 91.
    """
    points = [(i * _P[0] + j * _P[1] + (n - i - j) * _P[2]) / n
              for i in range(1, n - 1) for j in range(1, n - i)]
    points = np.asarray(points)
    return points, np.full(len(points), 1.0 / len(points))


def worst_error(n, rule=sub_triangle_centroids):
    """Largest relative error over :data:`_TESTS` for ``rule`` at subdivision ``n``."""
    points, weights = rule(n)
    l2, l3 = points[:, 0], points[:, 1]
    bary = np.stack([1.0 - l2 - l3, l2, l3], axis=1)
    worst = 0.0
    for a, b, c in _TESTS:
        got = 0.5 * np.sum(weights * bary[:, 0] ** a * bary[:, 1] ** b
                           * bary[:, 2] ** c)
        worst = max(worst, abs(got - exact(a, b, c)) / exact(a, b, c))
    return worst


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--max-n", type=int, default=136)
    args = ap.parse_args()

    # Subdivisions chosen so the reference's point count (n-1)(n-2)/2 roughly
    # doubles each row, starting at the 91 it actually ships.
    ns = np.array([n for n in (6, 10, 15, 22, 32, 46, 66, 95, 136) if n <= args.max_n])
    ref = np.array([worst_error(int(n), reference_interior) for n in ns])
    npts = np.array([(n - 1) * (n - 2) // 2 for n in ns])
    # This package's rule at the SAME point budget (its count is n**2).
    nb = np.maximum(2, np.round(np.sqrt(npts)).astype(int))
    both = np.array([worst_error(int(n)) for n in nb])

    print("%5s %9s %14s   %5s %9s %14s %9s"
          % ("ref n", "ref pts", "REFERENCE", "n", "pts", "this package", "ratio"))
    for n, p_, e, n2, e2 in zip(ns, npts, ref, nb, both):
        print("%5d %9d %14.3e   %5d %9d %14.3e %8.1fx"
              % (n, p_, e, n2, n2 * n2, e2, e / e2))

    print("\nfitted order vs POINT COUNT:  reference N**%.2f   this package N**%.2f"
          % (np.polyfit(np.log(npts), np.log(ref), 1)[0],
             np.polyfit(np.log(nb ** 2), np.log(both), 1)[0]))
    print("             vs subdivision n:  reference n**%.2f"
          % np.polyfit(np.log(ns), np.log(ref), 1)[0])
    print("A PLATEAU would fit N**0. The reference converges -- at first order in n,")
    print("so only as sqrt of the point count. -2 in k is what this package's error")
    print("model assumes, and only the both-orientation rule delivers it.")

    # Why: the interior lattice is a complete lattice on a SHRUNKEN triangle.
    print("\nthe shrinkage that causes it:")
    print("%6s %12s %14s %14s" % ("n", "scale (n-3)/n", "area sampled", "ref error"))
    for n, e in zip(ns, ref):
        s = (n - 3) / n
        print("%6d %12.3f %13.0f%% %14.3e" % (n, s, 100 * s * s, e))

    e91 = worst_error(15, reference_interior)
    for n in range(2, 40):
        if worst_error(n) <= e91:
            print("\nthe reference's 91-point accuracy (%.2e) is matched by this" % e91)
            print("package's rule at n=%d, i.e. %d points -- %.1fx fewer."
                  % (n, n * n, 91 / (n * n)))
            break


if __name__ == "__main__":
    main()
