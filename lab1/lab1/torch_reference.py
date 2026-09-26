import torch
from torch import nn


def build_torch_model(params):
    """Та сама мережа в PyTorch з параметрами NumPy-мережі.

    nn.Linear зберігає вагу як (out, in), тому ваги транспонуємо.
    """
    lin1 = nn.Linear(4, 8, dtype=torch.float64)
    lin2 = nn.Linear(8, 3, dtype=torch.float64)
    with torch.no_grad():
        lin1.weight.copy_(torch.tensor(params["W1"].T))
        lin1.bias.copy_(torch.tensor(params["b1"]))
        lin2.weight.copy_(torch.tensor(params["W2"].T))
        lin2.bias.copy_(torch.tensor(params["b2"]))
    return lin1, lin2


def torch_loss_and_grads(params, X, y):
    lin1, lin2 = build_torch_model(params)
    logits = lin2(torch.relu(lin1(torch.tensor(X))))
    loss = nn.functional.cross_entropy(logits, torch.tensor(y))
    loss.backward()

    grads = {
        "W1": lin1.weight.grad.T.numpy(),
        "b1": lin1.bias.grad.numpy(),
        "W2": lin2.weight.grad.T.numpy(),
        "b2": lin2.bias.grad.numpy(),
    }
    return loss.item(), grads
