import numpy as np

from model import forward

TORCH_TOL = 1e-12
EPS = 1e-6
NUM_TOL = 1e-7


def compare_with_torch(loss_np, grads_np, loss_t, grads_t):
    """Повертає рядки таблиці: (назва, макс. абсолютна різниця, чи пройдено)."""
    rows = []
    d = abs(loss_np - loss_t)
    ok = np.isfinite(loss_np) and np.isfinite(loss_t) and d <= TORCH_TOL
    rows.append(("Втрата", d, ok))
    for k in ("W1", "b1", "W2", "b2"):
        a, b = grads_np[k], grads_t[k]
        assert a.shape == b.shape
        diff = np.abs(a - b)
        ok = np.isfinite(a).all() and np.isfinite(b).all() and (diff <= TORCH_TOL).all()
        rows.append((f"Градієнт {k}", diff.max(), ok))
    return rows


def numeric_check(params, X, y, grads):
    """Центральна різниця для W1[0,0], b1[0], W2[0,0], b2[0]."""
    targets = [("W1[0,0]", "W1", (0, 0)), ("b1[0]", "b1", (0,)),
               ("W2[0,0]", "W2", (0, 0)), ("b2[0]", "b2", (0,))]
    signs = forward(params, X, y)[1]["Z1"] > 0
    rows = []
    relu_ok = True
    for name, key, idx in targets:
        old = params[key][idx]

        params[key][idx] = old + EPS
        L_plus, c_plus = forward(params, X, y)
        params[key][idx] = old - EPS
        L_minus, c_minus = forward(params, X, y)
        params[key][idx] = old

        # перевірка, що ±eps не перетинає злам ReLU
        relu_ok &= ((c_plus["Z1"] > 0) == signs).all() and ((c_minus["Z1"] > 0) == signs).all()

        g_num = (L_plus - L_minus) / (2 * EPS)
        g_man = grads[key][idx]
        diff = abs(g_num - g_man)
        rows.append((name, g_man, g_num, diff, diff <= NUM_TOL))
    return rows, relu_ok
