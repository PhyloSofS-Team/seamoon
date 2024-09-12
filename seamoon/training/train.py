from seamoon.utils.params import load_params
from seamoon.utils.logger import TensorBoardLogger
from seamoon.utils.seeding import set_seed
from seamoon.data.data_set import CustomDataset
from seamoon.data.data_loader import create_data_loader
from seamoon.model.loss import CustomLoss
from seamoon.model.optimizer import get_optimizer
from seamoon.model.neural_net import HEADS
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from glob import glob
import torch.nn.functional as F
import numpy as np
import torch
import time
import sys
import json
import inspect
import os


def setup_environment(config_file):
    params = load_params(config_file)
    set_seed(seed=params["Misc"]["seed"])

    logger = TensorBoardLogger(config_file, params)
    logger.log_numerical_params(params)
    logger.log_text("Parameters", json.dumps(params, indent=4))

    train_dataset = CustomDataset(
        list_path=params["Data"]["training_split_path"],
        precomputed_path=params["Data"]["precomputed_path"],
        num_modes=params["Model_Configuration"]["num_modes_gt"],
        emb_model=params["Model_Configuration"]["emb_model"],
        noise=params["Data"]["noise"],
    )
    train_loader = create_data_loader(
        train_dataset,
        params["Training_Configuration"]["batch_size"],
        num_workers=4,
        pin_memory=False,
    )

    validation_dataset = CustomDataset(
        list_path=params["Data"]["validation_split_path"],
        precomputed_path=params["Data"]["precomputed_path"],
        num_modes=params["Model_Configuration"]["num_modes_gt"],
        emb_model=params["Model_Configuration"]["emb_model"],
    )
    validation_loader = create_data_loader(
        validation_dataset,
        params["Training_Configuration"]["batch_size"],
        num_workers=1,
        pin_memory=False,
    )

    head = HEADS[params["Head"]["head_selection"]](
        in_features=params["Head"]["in_features"],
        kernel_sizes=params["Head"]["kernel_sizes"],
        num_modes=params["Model_Configuration"]["num_modes_pred"],
        dropout_coeff=params["Training_Configuration"]["dropout"],
        hidden_sizes=params["Head"]["hidden_sizes"],
        qr_reg=params["Head"]["qr_reg"],
        use_bn=params["Head"]["use_bn"],
    ).cuda()

    logger.log_text(
        tag="Head source code",
        text_string=inspect.getsource(HEADS[params["Head"]["head_selection"]]),
    )

    logger.log_scalar(
        tag="Number of trainable parameters",
        value=sum(p.numel() for p in head.parameters() if p.requires_grad),
    )

    if params["Loss"]["loss"] == "L1":
        loss_fn = CustomLoss(
            num_modes=params["Model_Configuration"]["num_modes_pred"],
            allow_permutation=params["Loss"]["allow_permutations"],
            allow_reflections=params["Loss"]["allow_reflections"],
            allow_sign_flip=params["Loss"]["allow_sign_flip"],
            reweight_centrality=params["Loss"]["reweight_centrality"],
        )

    optimizer = get_optimizer(
        parameters=head.parameters(),
        optimizer_name=params["Training_Configuration"]["optimizer"],
        learning_rate=params["Training_Configuration"]["learning_rate"],
    )

    scheduler = StepLR(
        optimizer=optimizer,
        step_size=params["Training_Configuration"]["scheduler_step_size"],
        gamma=params["Training_Configuration"]["scheduler_gamma"],
    )

    return {
        "params": params,
        "logger": logger,
        "train_loader": train_loader,
        "validation_loader": validation_loader,
        "head": head,
        "loss_fn": loss_fn,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "nb_epochs": params["Training_Configuration"]["nb_epochs"],
    }


def process_initial_epoch(dataloader, env_objects):

    loss_fn, logger, params = [
        env_objects[key] for key in ("loss_fn", "logger", "params")
    ]
    coeff_modes_value = params["Loss"]["coeff_modes"]
    coeff_torque_value = params["Loss"]["coeff_torque"]

    zero_losses_modes = []
    zero_losses_torque = []

    losses_dict_sample = {}
    for batch in dataloader:
        eigvects, eigvals, ref, seq_lengths, coverage, names = [
            batch[key]
            for key in ("eigvects", "eigvals", "ref", "seq_lengths", "coverage", "name")
        ]
        eigvects, eigvals, ref, seq_lengths, coverage = (
            eigvects.cuda(),
            eigvals.cuda().float(),
            ref.cuda(),
            seq_lengths.cuda(),
            coverage.cuda(),
        )
        zero_modes = torch.zeros_like(eigvects)
        zero_modes = zero_modes.reshape(
            zero_modes.shape[0], zero_modes.shape[1], int(zero_modes.shape[2] / 3), 3
        )
        zero_modes = zero_modes[
            :, 0 : params["Model_Configuration"]["num_modes_pred"], :, :
        ]
        loss_dict_zero_modes = loss_fn(
            zero_modes,
            eigvects,
            eigvals,
            ref,
            coverage,
            seq_lengths,
            coeff_modes_value,
            coeff_torque_value,
        )
        zero_losses_modes.append(float(loss_dict_zero_modes["loss_modes"]))
        zero_losses_torque.append(float(loss_dict_zero_modes["loss_torque"]))
        for k, mode, torque in zip(
            names,
            loss_dict_zero_modes["individual_loss_modes"],
            loss_dict_zero_modes["individual_loss_torque"],
        ):
            losses_dict_sample[k] = [mode.item(), torque.item()]
    return (np.mean(zero_losses_modes), np.mean(zero_losses_torque), losses_dict_sample)


def process_epoch(epoch, dataloader, is_training, env_objects):

    loss_fn, optimizer, head, logger, params = [
        env_objects[key] for key in ("loss_fn", "optimizer", "head", "logger", "params")
    ]

    coeff_modes_value = params["Loss"]["coeff_modes"]
    coeff_torque_value = params["Loss"]["coeff_torque"]

    losses_modes = []
    losses_torque = []

    losses_dict_sample = {}

    head.train() if is_training else head.eval()
    tag = "Training" if is_training else "Validation"

    # Handling initial epoch separately
    if epoch == -1:  # intial epoch with blank data

        head.eval()
        avg_blank_modes, avg_blank_torque, losses_dict_sample = process_initial_epoch(
            dataloader, env_objects
        )
        se_loss = np.array([value[0] for value in losses_dict_sample.values()])
        torque_loss = np.array([value[1] for value in losses_dict_sample.values()])
        logger.log_histogram(f"{tag} modes", se_loss, epoch)
        logger.log_histogram(f"{tag} torque", torque_loss, epoch)
        return (avg_blank_modes, avg_blank_torque)

    for batch in dataloader:

        emb, eigvects, eigvals, ref, coverage, seq_lengths, names = [
            batch[key]
            for key in (
                "emb",
                "eigvects",
                "eigvals",
                "ref",
                "coverage",
                "seq_lengths",
                "name",
            )
        ]

        emb, eigvects, eigvals, ref, coverage, seq_lengths = (
            emb.cuda().float(),
            eigvects.cuda(),
            eigvals.cuda().float(),
            ref.cuda(),
            coverage.cuda(),
            seq_lengths.cuda(),
        )
        modes_pred = head(emb, seq_lengths)

        loss_dict = loss_fn(
            modes_pred,
            eigvects,
            eigvals,
            ref,
            coverage,
            seq_lengths,
            coeff_modes_value,
            coeff_torque_value,
            return_modes=False,
            is_training=is_training,
        )

        losses_modes.extend(loss_dict["individual_loss_modes"].tolist())
        losses_torque.extend(loss_dict["individual_loss_torque"].tolist())

        if is_training:
            loss_dict["total_loss"].backward()
            optimizer.step()
            optimizer.zero_grad()

    avg_loss_modes = np.mean(losses_modes)
    avg_loss_torque = np.mean(losses_torque)

    return avg_loss_modes, avg_loss_torque


def train_loop(config_file):
    env_objects = setup_environment(config_file)
    logger = env_objects["logger"]
    params = env_objects["params"]

    train_loader = env_objects["train_loader"]
    validation_loader = env_objects["validation_loader"]

    nb_files = len(train_loader.dataset) + len(validation_loader.dataset)

    pbar = tqdm(total=env_objects["nb_epochs"])
    min_test_loss = float("inf")
    min_train_loss = float("inf")
    best_model_epoch = None
    best_model_path = None
    head = env_objects["head"]

    try:
        start_time = time.time()

        for epoch in range(0, env_objects["nb_epochs"]):  # -1 to include initial epoch
            start_time = time.time()
            logger.log_scalar(
                "Learning rate", env_objects["scheduler"].get_last_lr()[0], epoch
            )

            # Training
            avg_loss_modes, avg_loss_torque = process_epoch(
                epoch, train_loader, is_training=True, env_objects=env_objects
            )
            logger.log_scalar("Training loss modes", avg_loss_modes, epoch)
            logger.log_scalar("Training loss torque", avg_loss_torque, epoch)

            min_train_loss = min(min_train_loss, avg_loss_modes)

            # Validation
            with torch.no_grad():
                avg_loss_modes, avg_loss_torque = process_epoch(
                    epoch, validation_loader, is_training=False, env_objects=env_objects
                )
            logger.log_scalar("Validation loss modes", avg_loss_modes, epoch)
            logger.log_scalar("Validation loss torque", avg_loss_torque, epoch)

            if epoch >= 0:
                env_objects["scheduler"].step()
                pbar.update(1)

            epoch_duration = time.time() - start_time
            logger.log_scalar("Epoch duration", epoch_duration, epoch)
            logger.log_scalar(
                "Epoch duration per sample", epoch_duration / nb_files, epoch
            )

            if avg_loss_modes < min_test_loss:
                min_test_loss = avg_loss_modes
                model_dir = f"weights/{logger.run_name}"
                new_best_model_path = f"{model_dir}/best_model_epoch_{epoch}.pt"
                os.makedirs(model_dir, exist_ok=True)
                if best_model_path is not None and os.path.exists(best_model_path):
                    os.remove(best_model_path)
                torch.save(head.state_dict(), new_best_model_path)
                best_model_path = new_best_model_path
                tqdm.write(f"New best model saved at {new_best_model_path}")

        min_test_loss = min(min_test_loss, avg_loss_modes)
        # logger.log_hparam(params, min_test_loss, min_train_loss)
        logger.close()

    except KeyboardInterrupt:
        logger.log_hparam(params, min_test_loss, min_train_loss)
        logger.close()
        pbar.close()


if __name__ == "__main__":
    train_loop(sys.argv[1])
