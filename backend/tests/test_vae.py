import numpy as np
import pytest

from app.models.vae import VariationalAutoencoder


def toy_data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.random(n)
    return np.column_stack([a, 1 - a + rng.normal(0, 0.02, n), rng.random(n)]).clip(0, 1).astype(np.float32)


def test_training_is_reproducible_with_a_seed():
    X = toy_data()
    runs = []
    for _ in range(2):
        v = VariationalAutoencoder(3, 2, [16, 8], seed=5)
        for _ in range(5):
            v.train_epoch(X, 32)
        runs.append(v)
    for k in runs[0].params:
        np.testing.assert_array_equal(runs[0].params[k], runs[1].params[k])


def test_training_reduces_loss_and_does_not_touch_global_random_state():
    X = toy_data()
    np.random.seed(123)
    before = np.random.get_state()[1].copy()
    v = VariationalAutoencoder(3, 2, [16, 8], seed=1)
    first = v.train_epoch(X, 32)[0]
    for _ in range(40):
        last = v.train_epoch(X, 32)[0]
    v.generate(10, seed=1)
    assert last < first
    np.testing.assert_array_equal(np.random.get_state()[1], before)


def test_gradient_matches_finite_differences():
    X = toy_data(16)
    v = VariationalAutoencoder(3, 2, [6, 4], seed=2)

    def loss_with_fixed_eps(eps):
        mu, logvar, enc = v.encode(X)
        z = mu + np.exp(0.5 * logvar) * eps
        X_hat, dec = v.decode(z)
        return v.compute_loss(X, X_hat, mu, logvar)[0], {**enc, **dec, "eps": eps}

    eps = np.random.default_rng(0).standard_normal((16, 2))
    _, cache = loss_with_fixed_eps(eps)
    grads = v.backward(cache)
    for key in ("W_out", "W_mu", "W_e1", "b_d1"):
        idx = tuple(0 for _ in v.params[key].shape)
        orig = v.params[key][idx]
        v.params[key][idx] = orig + 1e-5
        up = loss_with_fixed_eps(eps)[0]
        v.params[key][idx] = orig - 1e-5
        down = loss_with_fixed_eps(eps)[0]
        v.params[key][idx] = orig
        assert grads[key][idx] == pytest.approx((up - down) / 2e-5, rel=1e-3, abs=1e-6)


def test_generate_is_reproducible_and_in_unit_range():
    v = VariationalAutoencoder(3, 2, [8, 4], seed=0)
    a, b = v.generate(50, 1.0, seed=9), v.generate(50, 1.0, seed=9)
    np.testing.assert_array_equal(a, b)
    assert a.shape == (50, 3) and a.min() >= 0 and a.max() <= 1
    assert not np.array_equal(a, v.generate(50, 1.0, seed=10))


def test_save_load_roundtrip(tmp_path):
    v = VariationalAutoencoder(3, 2, [8, 4], seed=0)
    v.train_epoch(toy_data(), 32)
    v.save(str(tmp_path / "w.npz"))
    w = VariationalAutoencoder(3, 2, [8, 4], seed=99)
    w.load(str(tmp_path / "w.npz"))
    np.testing.assert_array_equal(v.generate(20, seed=1), w.generate(20, seed=1))


def test_load_rejects_incompatible_or_corrupt_files(tmp_path):
    v = VariationalAutoencoder(3, 2, [8, 4], seed=0)
    v.save(str(tmp_path / "w.npz"))
    with pytest.raises(ValueError, match="incompatible"):
        VariationalAutoencoder(3, 2, [16, 4]).load(str(tmp_path / "w.npz"))
    with pytest.raises(ValueError, match="incompatible"):
        VariationalAutoencoder(5, 2, [8, 4]).load(str(tmp_path / "w.npz"))
    (tmp_path / "junk.npz").write_bytes(b"not an npz")
    with pytest.raises(ValueError, match="could not be read"):
        v.load(str(tmp_path / "junk.npz"))
    np.savez(str(tmp_path / "partial.npz"), W_e1=v.params["W_e1"])
    with pytest.raises(ValueError, match="missing parameter"):
        v.load(str(tmp_path / "partial.npz"))
    with pytest.raises(FileNotFoundError):
        v.load(str(tmp_path / "nope.npz"))


@pytest.mark.parametrize("hidden", [[8], [], [8, 4, 2], [8, 0]])
def test_architecture_must_have_two_positive_hidden_layers(hidden):
    with pytest.raises(ValueError):
        VariationalAutoencoder(3, 2, hidden)
