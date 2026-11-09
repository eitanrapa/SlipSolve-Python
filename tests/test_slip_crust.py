"""The layered crust (`slipsolve.crust.VelocityModel`): moduli, sampling, files, interfaces and exact triangle means."""

import numpy as np
import pytest

from slipsolve.crust import VelocityModel


# -- the velocity model ------------------------------------------------------

def test_velocity_model_moduli():
    model = VelocityModel.uniform(vp=6000.0, vs=3464.0, rho=2670.0)
    np.testing.assert_allclose(model.mu, 2670.0 * 3464.0 ** 2)
    # vp/vs = sqrt(3) is a Poisson solid, nu = 0.25.
    assert abs(model.poisson() - 0.25) < 1e-3


def test_velocity_model_samples_and_clamps():
    model = VelocityModel([0.0, 10e3, 30e3], [4.0e3, 6.0e3, 6.5e3],
                          [2.0e3, 3.5e3, 3.7e3], [2.4e3, 2.7e3, 2.9e3])
    shallow, deep = model.at(0.0, "mu"), model.at(30e3, "mu")
    assert shallow < deep, "the shallow crust must be the softer end"
    # Below the deepest row the bottom layer continues as a half-space.
    assert model.at(100e3, "mu") == pytest.approx(deep)
    assert model.at(-5e3, "mu") == model.at(5e3, "mu"), "depth sign must not matter"


def test_velocity_model_from_file(tmp_path):
    path = tmp_path / "crust.txt"
    path.write_text("# depth vp vs rho\n0 5000 2900 2600\n15000 6500 3700 2900\n")
    model = VelocityModel.from_file(path)
    assert len(model) == 2
    np.testing.assert_allclose(model.depth, [0.0, 15e3])

    # A five-column form leads with a row index (an EDGRN or PSGRN input file's layer block).
    indexed = tmp_path / "edgrn_style.txt"
    indexed.write_text("  1  0.0 5000 2900 2600\n  2  15000 6500 3700 2900\n")
    np.testing.assert_allclose(VelocityModel.from_file(indexed).vp, model.vp)


def test_velocity_model_rejects_ragged_input():
    with pytest.raises(ValueError, match="same length"):
        VelocityModel([0, 1], [1, 2], [1], [1, 2])


def _layer_cake():
    """Uniform layers with steps at 2 and 10.58 km: a mean over them has a closed form."""
    return VelocityModel(depth=[0, 2e3, 2e3, 10.58e3, 10.58e3, 40e3],
                         vp=[3.75e3] * 2 + [6.10e3] * 2 + [6.50e3] * 2,
                         vs=[1.95e3] * 2 + [3.50e3] * 2 + [3.65e3] * 2,
                         rho=[2.37e3] * 2 + [2.75e3] * 2 + [2.87e3] * 2)


def _area_above(depth, a, b, c):
    """Fraction of a planar triangle with sorted vertex depths a <= b <= c above ``depth``.

    Depth over a triangle's area is the triangular distribution with mode ``b``.
    """
    if depth <= a:
        return 0.0
    if depth >= c:
        return 1.0
    if depth <= b:
        return (depth - a) ** 2 / ((c - a) * (b - a))
    return 1.0 - (c - depth) ** 2 / ((c - a) * (c - b))


@pytest.mark.parametrize("depths", [(0.0, 0.0, 5e3), (1e3, 11e3, 25e3), (3e3, 8e3, 9e3),
                                    (0.0, 12e3, 12e3), (500.0, 2e3, 30e3)])
def test_triangle_mean_is_exact_across_interfaces(depths):
    """A triangle cut by steps gets each layer's rigidity weighted by its area share.

    Checked against the triangular-distribution CDF, which is independent of the
    divided-difference route the method takes. Sampling ``at`` inside the triangle
    is only first-order accurate here, because a sample decides its whole cell.
    """
    model = _layer_cake()
    a, b, c = depths
    edges = [0.0, 2e3, 10.58e3, np.inf]
    layer = [model.at(1e3, "mu"), model.at(5e3, "mu"), model.at(20e3, "mu")]
    expected = sum(mu * (_area_above(bot, a, b, c) - _area_above(top, a, b, c))
                   for mu, top, bot in zip(layer, edges[:-1], edges[1:]))
    assert model.triangle_mean(depths) == pytest.approx(expected, rel=1e-12)
    # Vertex order and depth sign are irrelevant, as they are to `at`.
    assert model.triangle_mean([-c, -a, -b]) == pytest.approx(expected, rel=1e-12)


def test_triangle_mean_of_a_gradient_is_its_centroid_value():
    """Linear in depth inside a layer, so the mean is the value at the mean depth."""
    model = VelocityModel([0.0, 10e3, 30e3], [4.0e3, 6.0e3, 6.5e3],
                          [2.0e3, 3.5e3, 3.7e3], [2.4e3, 2.7e3, 2.9e3])
    for depths in ([11e3, 14e3, 29e3], [12e3, 12e3, 20e3], [12e3, 20e3, 20e3],
                   [15e3, 15e3 + 1e-4, 18e3], [15e3, 15e3, 15e3]):
        assert model.triangle_mean(depths) == pytest.approx(
            model.at(np.mean(depths), "mu"), rel=1e-9), depths


def test_triangle_mean_broadcasts_and_matches_fine_sampling():
    """Many triangles at once, and the slow first-order route converging onto it."""
    from slipsolve._quadrature import _triangle_quadrature

    model = _layer_cake()
    rng = np.random.default_rng(3)
    depths = rng.uniform(0.0, 25e3, size=(4, 5, 3))
    means = model.triangle_mean(depths)
    assert means.shape == (4, 5)
    lattice, _ = _triangle_quadrature(*np.eye(3), 300)
    sampled = model.at(depths @ lattice.T, "mu").mean(axis=-1)
    np.testing.assert_allclose(means, sampled, rtol=2e-3)
    with pytest.raises(ValueError, match="3 vertex depths"):
        model.triangle_mean(np.zeros((4, 2)))


def test_interfaces_are_steps_not_gradients():
    """A step is two rows at one depth that differ; a gradient is not a step."""
    model = VelocityModel(depth=[0.0, 2e3, 2e3, 10e3, 15e3, 15e3],
                          vp=[3750.0, 3750.0, 6000.0, 6500.0, 6500.0, 8000.0],
                          vs=[1950.0, 1950.0, 3450.0, 3700.0, 3700.0, 4600.0],
                          rho=[2370.0, 2370.0, 2700.0, 2800.0, 2800.0, 3300.0])
    assert model.interfaces == [2e3, 15e3]


def test_round_trips_through_a_dict():
    model = VelocityModel([0.0, 2e3, 2e3, 40e3], [3750.0, 3750.0, 6100.0, 6100.0],
                          [1950.0, 1950.0, 3500.0, 3500.0], [2370.0, 2370.0, 2750.0, 2750.0], name="lid")
    back = VelocityModel.from_dict(model.to_dict())
    for field in ("depth", "vp", "vs", "rho"):
        np.testing.assert_array_equal(getattr(back, field), getattr(model, field))
    assert back.name == "lid" and back.interfaces == [2e3]
