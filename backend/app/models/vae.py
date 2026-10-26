import os
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

# Weight on the reconstruction term = 1 / (2 * sigma^2) of a Gaussian decoder. 20 corresponds to
# sigma^2 = 0.025, a sensible scale for features min-max scaled to [0, 1]. Chosen empirically on
# the sample student dataset (see BACKEND_DEVELOPMENT.md); with weight 1-10 the KL term dominated
# and the posterior collapsed.
DEFAULT_RECON_WEIGHT = 20.0


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -25.0, 25.0)))

def leaky_relu(x: np.ndarray, alpha: float = 0.2) -> np.ndarray:
    return np.where(x > 0, x, alpha * x)

def leaky_relu_deriv(x: np.ndarray, alpha: float = 0.2) -> np.ndarray:
    return np.where(x > 0, 1.0, alpha)

class VariationalAutoencoder:
    """
    Variational Autoencoder (VAE) for tabular synthesis, written in plain NumPy
    (no PyTorch/TensorFlow): a fully connected network with exactly two hidden
    layers in the encoder and decoder, hand-derived backpropagation and Adam.

    Loss = recon_weight * reconstruction + beta * KL, with
      reconstruction = squared error summed over features, averaged over the batch,
      recon_weight   = 1 / (2 * sigma^2) of a fixed-variance Gaussian decoder, and
      KL             = KL(q(z|x) || N(0, I)) summed over latent dims, averaged over the batch.
    Summing (rather than averaging) the reconstruction error over features keeps
    it on the same scale as the KL term; with a per-feature mean the KL term
    dominates and the posterior collapses (the decoder ignores z and emits column
    means), which is what the Step 1 audit observed.

    All randomness comes from a per-instance ``numpy.random.Generator`` so training
    and sampling are reproducible and never touch NumPy's global random state.
    """
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 8,
        hidden_dims: List[int] = [128, 64],
        learning_rate: float = 0.001,
        beta: float = 1.0,
        seed: Optional[int] = None,
        recon_weight: float = DEFAULT_RECON_WEIGHT,
    ):
        if input_dim < 1 or latent_dim < 1:
            raise ValueError("input_dim and latent_dim must be positive")
        if len(hidden_dims) != 2 or any(int(h) < 1 for h in hidden_dims):
            raise ValueError("hidden_dims must contain exactly two positive layer sizes")
        self.rng = np.random.default_rng(seed)
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.lr = learning_rate
        self.beta = beta
        self.recon_weight = float(recon_weight)
        self.h1_dim, self.h2_dim = int(hidden_dims[0]), int(hidden_dims[1])

        # Xavier / Glorot Initialization
        self.params: Dict[str, np.ndarray] = {
            # Encoder
            "W_e1": self.rng.standard_normal((input_dim, self.h1_dim)) * np.sqrt(2.0 / (input_dim + self.h1_dim)),
            "b_e1": np.zeros((1, self.h1_dim)),
            "W_e2": self.rng.standard_normal((self.h1_dim, self.h2_dim)) * np.sqrt(2.0 / (self.h1_dim + self.h2_dim)),
            "b_e2": np.zeros((1, self.h2_dim)),
            "W_mu": self.rng.standard_normal((self.h2_dim, latent_dim)) * np.sqrt(2.0 / (self.h2_dim + latent_dim)),
            "b_mu": np.zeros((1, latent_dim)),
            "W_logvar": self.rng.standard_normal((self.h2_dim, latent_dim)) * np.sqrt(2.0 / (self.h2_dim + latent_dim)),
            "b_logvar": np.zeros((1, latent_dim)),
            # Decoder
            "W_d1": self.rng.standard_normal((latent_dim, self.h2_dim)) * np.sqrt(2.0 / (latent_dim + self.h2_dim)),
            "b_d1": np.zeros((1, self.h2_dim)),
            "W_d2": self.rng.standard_normal((self.h2_dim, self.h1_dim)) * np.sqrt(2.0 / (self.h2_dim + self.h1_dim)),
            "b_d2": np.zeros((1, self.h1_dim)),
            "W_out": self.rng.standard_normal((self.h1_dim, input_dim)) * np.sqrt(2.0 / (self.h1_dim + input_dim)),
            "b_out": np.zeros((1, input_dim)),
        }

        # Adam Optimizer Moments
        self.m = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.t = 0
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8

    def encode(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        # Forward pass through Encoder
        z1 = X @ self.params["W_e1"] + self.params["b_e1"]
        a1 = leaky_relu(z1)

        z2 = a1 @ self.params["W_e2"] + self.params["b_e2"]
        a2 = leaky_relu(z2)

        mu = a2 @ self.params["W_mu"] + self.params["b_mu"]
        logvar = np.clip(a2 @ self.params["W_logvar"] + self.params["b_logvar"], -10.0, 10.0)

        cache = {"X": X, "z1": z1, "a1": a1, "z2": z2, "a2": a2, "mu": mu, "logvar": logvar}
        return mu, logvar, cache

    def reparameterize(self, mu: np.ndarray, logvar: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        std = np.exp(0.5 * logvar)
        eps = self.rng.standard_normal(mu.shape)
        z = mu + std * eps
        return z, eps

    def decode(self, z: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        # Forward pass through Decoder
        z_d1 = z @ self.params["W_d1"] + self.params["b_d1"]
        a_d1 = leaky_relu(z_d1)

        z_d2 = a_d1 @ self.params["W_d2"] + self.params["b_d2"]
        a_d2 = leaky_relu(z_d2)

        z_out = a_d2 @ self.params["W_out"] + self.params["b_out"]
        X_hat = sigmoid(z_out)

        cache = {"z": z, "z_d1": z_d1, "a_d1": a_d1, "z_d2": z_d2, "a_d2": a_d2, "z_out": z_out, "X_hat": X_hat}
        return X_hat, cache

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        mu, logvar, enc_cache = self.encode(X)
        z, eps = self.reparameterize(mu, logvar)
        X_hat, dec_cache = self.decode(z)

        cache = {**enc_cache, **dec_cache, "eps": eps}
        return X_hat, mu, logvar, cache

    def compute_loss(self, X: np.ndarray, X_hat: np.ndarray, mu: np.ndarray, logvar: np.ndarray) -> Tuple[float, float, float]:
        B = X.shape[0]
        # Reconstruction: squared error summed over features, averaged over the batch
        recon_loss = float(np.sum(np.square(X - X_hat)) / B)
        
        # KL Divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        var = np.exp(logvar)
        kl_div = float(-0.5 * np.sum(1.0 + logvar - np.square(mu) - var) / B)
        
        total_loss = float(self.recon_weight * recon_loss + self.beta * kl_div)
        return total_loss, recon_loss, kl_div

    def backward(self, cache: Dict[str, Any]) -> Dict[str, np.ndarray]:
        X = cache["X"]
        X_hat = cache["X_hat"]
        mu = cache["mu"]
        logvar = cache["logvar"]
        eps = cache["eps"]
        z = cache["z"]
        B = X.shape[0]
        D = X.shape[1]

        grads: Dict[str, np.ndarray] = {}

        # 1. Gradient of the (summed-over-features, batch-averaged) squared error w.r.t. the sigmoid output
        dX_hat = (2.0 * self.recon_weight / B) * (X_hat - X)
        dz_out = dX_hat * X_hat * (1.0 - X_hat)

        grads["W_out"] = cache["a_d2"].T @ dz_out
        grads["b_out"] = np.sum(dz_out, axis=0, keepdims=True)

        # 2. Backprop through Decoder Layer 2
        da_d2 = dz_out @ self.params["W_out"].T
        dz_d2 = da_d2 * leaky_relu_deriv(cache["z_d2"])
        grads["W_d2"] = cache["a_d1"].T @ dz_d2
        grads["b_d2"] = np.sum(dz_d2, axis=0, keepdims=True)

        # 3. Backprop through Decoder Layer 1
        da_d1 = dz_d2 @ self.params["W_d2"].T
        dz_d1 = da_d1 * leaky_relu_deriv(cache["z_d1"])
        grads["W_d1"] = z.T @ dz_d1
        grads["b_d1"] = np.sum(dz_d1, axis=0, keepdims=True)

        # 4. Gradient w.r.t latent code z
        dz = dz_d1 @ self.params["W_d1"].T

        # 5. Gradient w.r.t mu and logvar
        # z = mu + exp(0.5 * logvar) * eps
        std = np.exp(0.5 * logvar)
        dmu_recon = dz
        dlogvar_recon = dz * (0.5 * std * eps)

        # KL gradients: d/dmu [ -0.5 * (1 + logvar - mu^2 - exp(logvar)) ] = mu / B
        dmu_kl = (self.beta / B) * mu
        # d/dlogvar = -0.5 * (1 - exp(logvar)) / B = 0.5 * (exp(logvar) - 1) / B
        dlogvar_kl = (self.beta / (2.0 * B)) * (np.exp(logvar) - 1.0)

        dmu = dmu_recon + dmu_kl
        dlogvar = dlogvar_recon + dlogvar_kl

        # 6. Backprop through Encoder heads
        grads["W_mu"] = cache["a2"].T @ dmu
        grads["b_mu"] = np.sum(dmu, axis=0, keepdims=True)
        grads["W_logvar"] = cache["a2"].T @ dlogvar
        grads["b_logvar"] = np.sum(dlogvar, axis=0, keepdims=True)

        da2 = dmu @ self.params["W_mu"].T + dlogvar @ self.params["W_logvar"].T
        dz2 = da2 * leaky_relu_deriv(cache["z2"])
        grads["W_e2"] = cache["a1"].T @ dz2
        grads["b_e2"] = np.sum(dz2, axis=0, keepdims=True)

        da1 = dz2 @ self.params["W_e2"].T
        dz1 = da1 * leaky_relu_deriv(cache["z1"])
        grads["W_e1"] = X.T @ dz1
        grads["b_e1"] = np.sum(dz1, axis=0, keepdims=True)

        return grads

    def step(self, grads: Dict[str, np.ndarray]):
        # Adam Optimizer Parameter Update
        self.t += 1
        for k in self.params:
            g = np.clip(grads[k], -5.0, 5.0) # Gradient clipping
            self.m[k] = self.beta1 * self.m[k] + (1.0 - self.beta1) * g
            self.v[k] = self.beta2 * self.v[k] + (1.0 - self.beta2) * np.square(g)

            m_hat = self.m[k] / (1.0 - (self.beta1 ** self.t))
            v_hat = self.v[k] / (1.0 - (self.beta2 ** self.t))

            self.params[k] -= (self.lr / (np.sqrt(v_hat) + self.eps)) * m_hat

    def train_epoch(self, X: np.ndarray, batch_size: int) -> Tuple[float, float, float]:
        indices = self.rng.permutation(X.shape[0])
        total_l, recon_l, kl_l = 0.0, 0.0, 0.0
        n_batches = int(np.ceil(X.shape[0] / batch_size))

        for b in range(n_batches):
            b_idx = indices[b * batch_size : (b + 1) * batch_size]
            X_batch = X[b_idx]

            X_hat, mu, logvar, cache = self.forward(X_batch)
            loss, recon, kl = self.compute_loss(X_batch, X_hat, mu, logvar)
            grads = self.backward(cache)
            self.step(grads)

            total_l += loss
            recon_l += recon
            kl_l += kl

        return total_l / n_batches, recon_l / n_batches, kl_l / n_batches

    def generate(
        self,
        n_samples: int,
        temperature: float = 1.0,
        seed: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """Decode z ~ N(0, temperature^2 I). Pass ``seed`` (or ``rng``) for reproducible output."""
        if rng is None:
            rng = np.random.default_rng(seed) if seed is not None else self.rng
        z = rng.standard_normal((n_samples, self.latent_dim)) * temperature
        X_hat, _ = self.decode(z)
        return X_hat

    def save(self, filepath: str):
        """Write weights to ``filepath`` (.npz) atomically."""
        tmp = filepath + ".tmp.npz"
        np.savez(tmp, **self.params)
        os.replace(tmp, filepath)

    def load(self, filepath: str):
        """Load weights; raises ValueError if the file is unreadable or does not match this architecture."""
        try:
            with np.load(filepath, allow_pickle=False) as data:  # closes the file handle (Windows cannot replace an open file)
                loaded = {k: np.asarray(data[k]) for k in self.params}
        except FileNotFoundError:
            raise
        except KeyError as e:
            raise ValueError(f"model file is missing parameter {e}") from e
        except Exception as e:  # corrupt / not an npz
            raise ValueError(f"model file could not be read: {e}") from e
        for k, arr in loaded.items():
            if arr.shape != self.params[k].shape:
                raise ValueError(
                    f"model file is incompatible: '{k}' has shape {arr.shape}, expected {self.params[k].shape}"
                )
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"model file contains non-finite values in '{k}'")
        self.params = loaded
