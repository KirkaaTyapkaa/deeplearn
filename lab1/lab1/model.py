import numpy as np

# Мережа: X -> Linear(4, 8) -> ReLU -> Linear(8, 3)
# Ваги мають форму (in, out), тому Z = X @ W + b.


def init_params():
    rng = np.random.default_rng(0)
    W1 = rng.normal(0.0, np.sqrt(2 / 4), size=(4, 8))        # He
    W2 = rng.normal(0.0, np.sqrt(2 / (8 + 3)), size=(8, 3))  # Xavier
    return {
        "W1": W1,
        "b1": np.zeros(8),
        "W2": W2,
        "b2": np.zeros(3),
    }


def log_softmax(Z):
    # зсув на максимум рядка, щоб exp не переповнювався
    Z = Z - Z.max(axis=1, keepdims=True)
    return Z - np.log(np.exp(Z).sum(axis=1, keepdims=True))


def forward(params, X, y):
    """Прямий прохід. Повертає втрату і кеш для backward."""
    Z1 = X @ params["W1"] + params["b1"]
    A1 = np.maximum(Z1, 0)
    Z2 = A1 @ params["W2"] + params["b2"]

    logp = log_softmax(Z2)
    N = X.shape[0]
    loss = float(-logp[np.arange(N), y].mean())

    cache = {"X": X, "Z1": Z1, "A1": A1, "P": np.exp(logp), "y": y}
    return loss, cache


def backward(params, cache, divide_by_n=True):
    """Ручний зворотний прохід.

    divide_by_n=False - навмисна помилка для досліду (без ділення на N).
    """
    X, Z1, A1, P, y = cache["X"], cache["Z1"], cache["A1"], cache["P"], cache["y"]
    N = X.shape[0]

    Y = np.zeros_like(P)
    Y[np.arange(N), y] = 1

    dZ2 = P - Y                  # (N, 3)
    if divide_by_n:
        dZ2 = dZ2 / N

    dW2 = A1.T @ dZ2             # (8, 3)
    db2 = dZ2.sum(axis=0)        # (3,)

    dA1 = dZ2 @ params["W2"].T   # (N, 8)
    dZ1 = dA1 * (Z1 > 0)         # (N, 8), похідна ReLU

    dW1 = X.T @ dZ1              # (4, 8)
    db1 = dZ1.sum(axis=0)        # (8,)

    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}
