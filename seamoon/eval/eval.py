import torch
from tqdm import tqdm
from seamoon.data.data_set import CustomDataset
from seamoon.data.data_loader import create_data_loader
from seamoon.model.neural_net import HEADS
from seamoon.utils.params import load_params
import csv
import numpy as np
import os
import re
import sys


def evaluate(
    model_path,
    config_file,
    list_path,
    precomputed_path,
    output_path,
    batch_size=1,
    torque_mode=False,
    infer_only=False,
    device="cuda",
):

    if torque_mode:
        from seamoon.torque.solver import get_unique_rotations
        from seamoon.torque.utilities import apply_rotation
        from wolframclient.evaluation import WolframLanguageSession

        kernel = "/usr/local/Wolfram/WolframEngine/14.0/Executables/WolframKernel"
        session = WolframLanguageSession(kernel)
        session.start()
    try:
        params = load_params(config_file)

        emb_model = params["Model_Configuration"]["emb_model"]
        num_modes = params["Model_Configuration"]["num_modes_pred"]

        model = HEADS[params["Head"]["head_selection"]](
            in_features=params["Head"]["in_features"],
            kernel_sizes=params["Head"]["kernel_sizes"],
            num_modes=params["Model_Configuration"]["num_modes_pred"],
            dropout_coeff=params["Training_Configuration"]["dropout"],
            hidden_sizes=params["Head"]["hidden_sizes"],
            qr_reg=params["Head"]["qr_reg"],
            use_bn=params["Head"]["use_bn"],
        ).to(device)

        model.load_state_dict(
            torch.load(model_path, map_location=device, weights_only=False)
        )
        model.eval()
        dataset = CustomDataset(
            list_path=list_path,
            precomputed_path=precomputed_path,
            emb_model=emb_model,
            num_modes=num_modes,
        )
        model_name = model_path.split("/")[-2]
        data_loader = create_data_loader(
            dataset, batch_size=1, num_workers=1, pin_memory=False, noise_std=0
        )
        if torque_mode:
            model_name += "_torque"

        with torch.no_grad():
            if not infer_only:
                success = 0
                total_samples = 0
            for batch in tqdm(data_loader):

                emb, seq_lengths, names = [
                    batch[key] for key in ("emb", "seq_lengths", "full_name")
                ]
                emb, seq_lengths = emb.to(device).float(), seq_lengths.to(device)
                modes_pred = model(emb, seq_lengths)
                if torque_mode:
                    ref = batch["ref"].to(device)

                batch_size, _, seq_len, _ = modes_pred.shape
                mask = (
                    torch.arange(seq_len).expand(batch_size, seq_len).to(device)
                    < seq_lengths[:, None]
                )
                mask = mask.unsqueeze(1).unsqueeze(-1)
                modes_pred.masked_fill_(~mask, 0)
                total_force = (
                    torch.sum(modes_pred, dim=2, keepdim=True)
                    / seq_lengths[:, None, None, None]
                )
                modes_pred = modes_pred - total_force
                modes_pred.masked_fill_(~mask, 0)

                modes_pred = modes_pred / (
                    torch.norm(modes_pred, dim=(-1, -2), keepdim=True)
                )
                modes_pred = modes_pred * seq_lengths.sqrt()[:, None, None, None]

                if torque_mode:
                    modes_pred = modes_pred.cpu().numpy()
                    seq_lengths = seq_lengths.cpu().numpy()
                    n_structs = modes_pred.shape[0]
                    n_modes_per_struct = modes_pred.shape[1]
                    max_length_batch = modes_pred.shape[2]
                    new_modes_preds = np.zeros(
                        (n_structs, n_modes_per_struct, 4, max_length_batch, 3)
                    )
                    for i in range(n_structs):
                        ref = torch.reshape(ref, (ref.shape[0], ref.shape[1] // 3, 3))
                        struct = ref[i][: seq_lengths[i]]
                        for j in range(n_modes_per_struct):
                            mode = modes_pred[i, j][: seq_lengths[i]]
                            rotations = get_unique_rotations(struct, mode, session)
                            assert (
                                len(rotations) == 4
                            ), "We obtained more than 4 rotations, maybe change 1e-3 in the definition of get_unique_rotations"
                            # rotations : liste de 4 matrices de rotation

                            for k, rot in enumerate(rotations):
                                new_modes_preds[i, j, k, :] = np.pad(
                                    apply_rotation(rot, mode),
                                    ((0, max_length_batch - seq_lengths[i]), (0, 0)),
                                    "constant",
                                    constant_values=(0, 0),
                                )

                    seq_lengths = torch.tensor(seq_lengths).int().to(device)
                    modes_pred = torch.tensor(new_modes_preds).float().to(device)
                    modes_pred = modes_pred.reshape(
                        (n_structs, n_modes_per_struct * 4, max_length_batch, 3)
                    )

                if infer_only and not torque_mode:
                    os.makedirs(f"{output_path}/{model_name}", exist_ok=True)
                    for _, (name, modes, seqlen) in enumerate(
                        zip(names, modes_pred, seq_lengths)
                    ):
                        for j, mode in enumerate(modes):
                            np.savetxt(
                                f"{output_path}/{model_name}/{name}_mode_{j}.txt",
                                mode[:seqlen].cpu().numpy(),
                                fmt="%.6f",
                            )
                    continue
                if infer_only and torque_mode:
                    modes_pred = modes_pred.reshape(
                        (
                            modes_pred.shape[0],
                            modes_pred.shape[1] // 4,
                            4,
                            modes_pred.shape[2],
                            modes_pred.shape[3],
                        )
                    )
                    os.makedirs(f"{output_path}/{model_name}", exist_ok=True)
                    for _, (name, modes, seqlen) in enumerate(
                        zip(names, modes_pred, seq_lengths)
                    ):
                        for j, mode in enumerate(modes):
                            for k, mode_rot in enumerate(mode):
                                np.savetxt(
                                    f"{output_path}/{model_name}/{name}_mode_{j}_rot_{k}.txt",
                                    mode_rot[:seqlen].cpu().numpy(),
                                    fmt="%.6f",
                                )
                    continue
                if not infer_only:
                    eigvects, eigvals, coverage, ref = [
                        batch[key] for key in ("eigvects", "eigvals", "coverage", "ref")
                    ]
                    eigvects, eigvals, coverage, ref = (
                        eigvects.to(device),
                        eigvals.to(device).float(),
                        coverage.to(device),
                        ref.to(device),
                    )
                    coverage = coverage[:, None, :, None].float()
                    modes_truth = torch.reshape(
                        eigvects, (batch_size, num_modes, seq_len, 3)
                    )

                    modes_truth = modes_truth.float()
                    individual_loss_all_combinations = torch.zeros(
                        (modes_pred.shape[0], modes_pred.shape[1], modes_truth.shape[1])
                    )
                    individual_traces = torch.zeros(
                        (modes_pred.shape[0], modes_pred.shape[1], modes_truth.shape[1])
                    )
                    if torque_mode:
                        output_modes = torch.zeros_like(modes_pred)
                    else:
                        output_modes = torch.zeros(
                            (
                                modes_pred.shape[0],
                                modes_pred.shape[1],
                                modes_truth.shape[1],
                                modes_truth.shape[2],
                                3,
                            )
                        )

                    for i in range(modes_pred.shape[1]):
                        for j in range(modes_truth.shape[1]):

                            mode_i = modes_pred[:, i, :, :][:, None, :, :]
                            mode_truth_j = modes_truth[:, j, :, :][:, None, :, :]

                            ### rot
                            if torque_mode is False:
                                prods = torch.einsum(
                                    "bijk,bijl->bikl", coverage * mode_truth_j, mode_i
                                )  # (batch,num_modes,3,3) # la suite est invariante au sign flip d'un mode

                                U, S, V = torch.linalg.svd(prods, full_matrices=False)
                                trace = torch.sum(S, dim=-1)

                                individual_traces[:, i, j] = torch.sum(
                                    S, dim=-1
                                ).squeeze(-1)

                                best_U = U
                                best_V = V

                                # Compute the best rotations
                                best_rotation = torch.matmul(best_U, best_V).transpose(
                                    -2, -1
                                )

                                mode_truth_j = torch.einsum(
                                    "bkij,bklj->bkli", best_rotation, mode_truth_j
                                )
                                mode_i_rotated = torch.einsum(
                                    "bkij,bklj->bkli",
                                    best_rotation.permute(0, 1, 3, 2),
                                    mode_i,
                                )

                            c_bi_numerator = (
                                torch.sum(
                                    coverage * mode_truth_j * mode_i, dim=(-2, -1)
                                )
                                / seq_lengths[:, None]
                            )
                            c_bi_denominator = (
                                torch.sum(coverage * mode_i**2, dim=(-2, -1))
                                / seq_lengths[:, None]
                            )
                            c_bi_optimal = c_bi_numerator / c_bi_denominator
                            c_bi_optimal = c_bi_optimal.float()

                            mode_i_adjusted = mode_i * c_bi_optimal[:, :, None, None]
                            if torque_mode:
                                output_modes[:, i] = mode_i
                            else:
                                output_modes[:, i, j] = (
                                    mode_i_rotated * c_bi_optimal[:, :, None, None]
                                )
                            zero_loss = (
                                torch.sum(
                                    torch.sum(
                                        coverage
                                        * (
                                            (
                                                mode_truth_j
                                                - torch.zeros_like(mode_i_adjusted)
                                            )
                                            ** 2
                                        ),
                                        dim=(2, 3),
                                    ),
                                    dim=1,
                                )
                                / seq_lengths
                            )
                            individual_loss_modes = (
                                torch.sum(
                                    torch.sum(
                                        coverage
                                        * ((mode_truth_j - mode_i_adjusted) ** 2),
                                        dim=(-2, -1),
                                    ),
                                    dim=1,
                                )
                                / seq_lengths
                            ) / zero_loss

                            individual_loss_all_combinations[:, i, j] = (
                                individual_loss_modes
                            )

                    for lossess in individual_loss_all_combinations:
                        if torch.min(lossess) < 0.6:
                            success += 1
                        total_samples += 1

                    # save the losses and the modes
                    os.makedirs(f"{output_path}/{model_name}", exist_ok=True)
                    for b, name in enumerate(names):
                        # save the losses
                        with open(
                            f"{output_path}/{model_name}/{name}_losses.csv", "w"
                        ) as f:
                            losses = individual_loss_all_combinations[b].cpu().numpy()
                            writer = csv.writer(f)
                            num_cols = losses.shape[1]
                            header = ["mode_name"] + [
                                f"gt_{i}" for i in range(num_cols)
                            ]
                            writer.writerow(header)
                            for i in range(losses.shape[0]):
                                if torque_mode:
                                    row_name = f"pred_{i // 4}_rot_{i % 4}"
                                else:
                                    row_name = f"pred_{i // num_modes}_aligned_on_gt_{i % num_modes}"
                                writer.writerow([row_name] + losses[i].tolist())
                        # save the modes
                        if torque_mode:
                            output_modes = output_modes.reshape(
                                (
                                    output_modes.shape[0],
                                    output_modes.shape[1] // 4,
                                    4,
                                    output_modes.shape[2],
                                    output_modes.shape[3],
                                )
                            )
                            for i in range(num_modes):
                                for j in range(4):
                                    mode = output_modes[b, i, j]
                                    np.savetxt(
                                        f"{output_path}/{model_name}/{name}_pred_{i}_rot_{j}.txt",
                                        mode[: seq_lengths[b]].cpu().numpy(),
                                        fmt="%.6f",
                                    )
                        else:
                            for i in range(num_modes):
                                for j in range(num_modes):
                                    mode = output_modes[b, i, j]
                                    np.savetxt(
                                        f"{output_path}/{model_name}/{name}_pred_{i}_aligned_on_gt_{j}.txt",
                                        mode[: seq_lengths[b]].cpu().numpy(),
                                        fmt="%.6f",
                                    )
        if not infer_only:
            print(f"total samples: {total_samples}, success: {success}")
    finally:
        if torque_mode:
            session.terminate()
