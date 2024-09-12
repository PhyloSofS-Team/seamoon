import torch


def get_optimizer(parameters, optimizer_name, learning_rate):
    if optimizer_name == "Adam":
        return torch.optim.Adam(parameters, lr=learning_rate)
    elif optimizer_name == "SGD":
        return torch.optim.SGD(parameters, lr=learning_rate)
    elif optimizer_name == "Adagrad":
        return torch.optim.Adagrad(parameters, lr=learning_rate)
    elif optimizer_name == "Adadelta":
        return torch.optim.Adadelta(parameters, lr=learning_rate)
    elif optimizer_name == "RMSprop":
        return torch.optim.RMSprop(parameters, lr=learning_rate)
    elif optimizer_name == "AdamW":
        return torch.optim.AdamW(parameters, lr=learning_rate)
    else:
        raise ValueError("Invalid optimizer name")
