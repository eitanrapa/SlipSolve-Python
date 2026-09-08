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

⚠️ **The defect is the EQUAL WEIGHTS, not the excluded boundary** -- measured, and
the distinction matters because the obvious fix does not work. A lattice's nodes
own unequal areas: an interior node is shared by six sub-triangles, an edge node
by three, a corner by one. Weighting them equally misrepresents that, and the
mismatch lives in the ``O(1/n)`` boundary layer.

Putting the boundary back changes nothing --- ``all nodes, equal`` converges at
``n**-1.03`` against interior-only's ``n**-1.04``, errors within 1% of each other.
Weighting correctly does: the P1 vertex rule (every sub-triangle giving ``area/3``
to each of its three vertices, so the weights come out ``6:3:1``) reaches
``n**-1.99``. So the reference is avoiding a real problem with a fix that does not
address it.

⚠️ **Three earlier descriptions in this repo were wrong** and this script exists
to stop them recurring. `_triangle_quadrature`'s docstring said the reference takes
"only the upright" sub-triangles -- it does not, it takes interior lattice nodes.
It said the error "stops falling however many points are added", "plateaus around
5e-3" -- it does not plateau, it converges at first order; the apparent plateau
came from measuring on the real layered integrand, where EDGRN table-interpolation
error takes over and hides the continued slow convergence. And it then blamed the
first order on the ``(n-3)/n`` shrinkage of the interior lattice, which the
``all nodes, equal`` row here refutes: remove the shrinkage entirely and the rule
is still first order.

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


def all_nodes_equal(n):
    """Every lattice node, boundary included, equal weights -- the obvious "fix".

    Included to show it is not one. If the shrinkage of the interior lattice were
    the cause of first-order convergence, restoring the boundary would restore
    second order. It does not.
    """
    points = [(i * _P[0] + j * _P[1] + (n - i - j) * _P[2]) / n
              for i in range(0, n + 1) for j in range(0, n + 1 - i)]
    points = np.asarray(points)
    return points, np.full(len(points), 1.0 / len(points))


def all_nodes_weighted(n):
    """Every lattice node with the area it actually owns -- the P1 vertex rule.

    Each sub-triangle contributes ``area/3`` to each of its three vertices, so an
    interior node accumulates six such shares, an edge node three and a corner
    one: weights ``6:3:1``. This is the composite rule the equal-weight lattice is
    a broken form of, and it recovers second order -- which is what identifies the
    weights, rather than the boundary, as the defect.
    """
    from collections import defaultdict

    weight = defaultdict(float)
    for a in range(n):
        for b in range(n - a):
            c = n - 1 - a - b
            for key in ((a + 1, b, c), (a, b + 1, c), (a, b, c + 1)):
                weight[key] += 1.0 / 3.0
            if c >= 1:
                for key in ((a + 1, b + 1, c - 1), (a, b + 1, c), (a + 1, b, c)):
                    weight[key] += 1.0 / 3.0
    keys = list(weight)
    points = np.asarray([(i * _P[0] + j * _P[1] + k * _P[2]) / n for i, j, k in keys])
    w = np.asarray([weight[k] for k in keys])
    return points, w / w.sum()


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
    ap.add_argument("--max-n", type=int, default=66)
    args = ap.parse_args()

    ns = np.array([n for n in (6, 10, 15, 22, 32, 46, 66, 95) if n <= args.max_n])
    rules = [("REF interior, equal", reference_interior),
             ("all nodes, equal", all_nodes_equal),
             ("all nodes, 6:3:1", all_nodes_weighted),
             ("OURS centroids", sub_triangle_centroids)]
    err = {name: np.array([worst_error(int(n), f) for n in ns]) for name, f in rules}

    print("%5s" % "n" + "".join("%22s" % name for name, _ in rules))
    for i, n in enumerate(ns):
        print("%5d" % n + "".join("%22.3e" % err[name][i] for name, _ in rules))

    print()
    for name, _ in rules:
        order = np.polyfit(np.log(ns), np.log(err[name]), 1)[0]
        print("  order in n:  %-22s n**%.2f" % (name, order))

    print("""
Read the middle two rows together. Putting the boundary back changes nothing --
the shrinkage of the interior lattice is NOT what makes it first order. Giving the
nodes the area they actually own does: 6:3:1 recovers second order. The defect is
the equal weights.

Centroids avoid the question rather than answering it: a sub-triangle centroid is
interior to exactly one cell and owns exactly area/k**2, so there is no sharing to
weight for. That is why -2 -- the exponent the package's error model assumes -- is
what this rule delivers.""")

    # ⚠️ 91 = (15-1)(15-2)/2, the interior-node count at n=15 -- a lattice number,
    # not an optimised rule count (Dunavant's go 1,3,4,6,7,12,13,16,19,25,27,33,
    # 37,42,48,52,61 -- no 91). It falls out of choosing n=15, and is not a target.
    target = worst_error(15, reference_interior)
    for n in range(2, 40):
        if worst_error(n) <= target:
            print("\nthe reference's 91-point accuracy %.2e is matched by this rule"
                  % target)
            print("at n=%d, i.e. %d points -- %.1fx fewer." % (n, n * n, 91 / (n * n)))
            break


if __name__ == "__main__":
    main()
