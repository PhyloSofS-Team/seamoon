from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os


def flatten_dict(d, parent_key="", sep="_"):
    items = []
    for k, v in d.items():
        new_key = parent_key
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            v = str(v)
        else:
            items.append((new_key + sep + k if new_key else k, v))
    return dict(items)


def is_numerical(value):
    return isinstance(value, (int, float))


def extract_numerical_values(data):
    numerical_dict = {}
    if isinstance(data, dict):
        for key, value in data.items():
            if is_numerical(value):
                numerical_dict[key] = value
            elif isinstance(value, dict):
                sub_numerical_dict = extract_numerical_values(value)
                numerical_dict.update(sub_numerical_dict)
    return numerical_dict


class TensorBoardLogger:
    def __init__(self, config_file, params):

        config_file_short = os.path.splitext(os.path.basename(config_file))[0]
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.run_name = f"{config_file_short}_{current_time}"
        self.createdir(params["Data"]["log_dir"])
        log_dir = f'{params["Data"]["log_dir"]}/runs/{self.run_name}'
        self.writer = SummaryWriter(log_dir)

    def createdir(self, log_dir):
        os.makedirs(log_dir, exist_ok=True)

    def log_scalar(self, tag, value, step=None):
        self.writer.add_scalar(tag, value, step)

    def log_text(self, tag, text_string, step=None):
        self.writer.add_text(tag, text_string, step)

    def log_hparam(self, hparam_dict, min_test_loss, min_train_loss):
        metric_dict = {"min_test_loss": min_test_loss, "min_train_loss": min_train_loss}
        hparam_dict = flatten_dict(hparam_dict)
        self.writer.add_hparams(
            hparam_dict, metric_dict, run_name="../../hparam/" + self.run_name
        )

    def log_numerical_params(self, params):
        numerical_dict = extract_numerical_values(params)
        for key, value in numerical_dict.items():
            self.log_scalar(key, value)

    def log_figure(self, tag, figure, global_step=None):
        self.writer.add_figure(tag, figure, global_step, close=True)

    def log_histogram(self, tag, values, step=None):
        self.writer.add_histogram(tag, values, step)

    def close(self):
        self.writer.close()
