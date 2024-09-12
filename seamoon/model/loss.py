import torch
import itertools
import sys


def to_d3(mat):
    ## mat must be(nb_conf, nb_res*3)
    ## output is (nb_conf, nb_res, 3)
    return torch.reshape(mat, (mat.shape[0], mat.shape[1] // 3, 3))


def generate_combinations_tensor(nb_modes):
    combinations = itertools.product([1, -1], repeat=nb_modes - 1)
    combinations_tensor = (
        torch.tensor([[1] + list(comb) for comb in combinations]).float().cuda()
    )
    return combinations_tensor


def apply_best_permutations(query, best_permutations):
    batch_size, nb_modes, seq_len, _ = query.shape
    best_permutations_tensor = torch.tensor(
        best_permutations, dtype=torch.long, device=query.device
    )
    gather_indices = (
        best_permutations_tensor.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, seq_len, 3)
    )
    query_permuted = torch.gather(query, 1, gather_indices)
    return query_permuted


def find_inverse_permutations(best_permutations):
    inverse_permutations = []
    for perm in best_permutations:
        inverse_perm = [0] * len(perm)
        for i, p in enumerate(perm):
            inverse_perm[p] = i
        inverse_permutations.append(inverse_perm)
    return inverse_permutations


def piecewise_function(x, N):
    if 0 <= x <= N / 2:
        return (2 / N) * x
    elif N / 2 < x <= N:
        return 2 - (2 / N) * x
    else:
        return 0


def compute_centrality(modes, seqlen):
    batch_size, nb_modes, seqlen_pad, _ = modes.shape
    norms = torch.sqrt(
        torch.sum(modes**2, dim=-1, keepdim=True)
    )  # [batch, nb_modes, seqlen+pad, 1]
    norms = norms / seqlen[:, None, None, None]
    ramp_vectors = torch.zeros(
        (batch_size, seqlen_pad), dtype=modes.dtype, device=modes.device
    )
    for i in range(batch_size):
        seq_len = seqlen[i].item()
        ramp_vectors[i, :seq_len] = torch.tensor(
            [piecewise_function(x, seq_len) for x in range(seq_len)]
        )

    ramp_vectors = ramp_vectors.unsqueeze(1).unsqueeze(-1)  # [batch, 1, seqlen+pad, 1]
    ramp_vectors = ramp_vectors.expand(batch_size, nb_modes, seqlen_pad, 1)

    weighted_norms = norms * ramp_vectors  # [batch, nb_modes, seqlen+pad, 1]
    centrality_measures = torch.sum(weighted_norms, dim=2).squeeze(
        -1
    )  # [batch, nb_modes]

    return centrality_measures


class CustomLoss:
    def __init__(
        self,
        num_modes,
        allow_permutation=True,
        allow_sign_flip=True,
        allow_reflections=True,
        reweight_centrality=False,
    ):

        if allow_permutation:  # permutation between the modes
            self.allow_permutation = True
        else:
            self.allow_permutation = False

        if allow_sign_flip:  # sign flip of the modes
            self.signs_tensor = generate_combinations_tensor(num_modes)
        else:
            self.signs_tensor = torch.tensor([[1] * num_modes]).float().cuda()

        if allow_reflections:  # reflection in the predicted best rotation
            self.allow_reflections = True
        else:
            self.allow_reflections = False

        if reweight_centrality:
            self.reweight_centrality = True
        else:
            self.reweight_centrality = False

    def __call__(
        self,
        modes,
        modes_truth,
        eigvals,
        ref_conf,
        coverage,
        seq_lengths,
        coeff_modes,
        coeff_torque,
        return_modes: bool = False,
        is_training: bool = True,
    ):

        seq_lengths = torch.count_nonzero(coverage, dim=1)
        mask = coverage != 0
        mask = mask.unsqueeze(1).unsqueeze(-1)
        modes.masked_fill_(~mask, 0)
        total_force = (
            torch.sum(modes, dim=2, keepdim=True) / seq_lengths[:, None, None, None]
        )
        modes = modes - total_force
        modes.masked_fill_(~mask, 0)

        batch_size, nb_modes, tNres = modes_truth.size()
        modes = (
            modes.float()
        )  # modes already have the (batch_size, nb_modes, tNres, 3) shape
        modes_truth = torch.reshape(
            modes_truth, (batch_size, nb_modes, int(tNres / 3), 3)
        )
        modes_truth = modes_truth.float()
        coverage = coverage.clone().detach().cuda()[:, None, :, None].float()

        with torch.no_grad():
            best_rotation, best_signs, best_permutations, modes_truth_permuted = (
                self.find_best_rotation(modes_truth, modes, coverage)
            )
            best_signs = best_signs.cuda()

            modes_truth_rotated = torch.einsum(
                "bij,bklj->bkli", best_rotation, modes_truth_permuted
            )
            modes_truth_rotated *= best_signs.unsqueeze(-1).unsqueeze(-1)

            rotated_ref_conf = torch.einsum(
                "bij,bkj->bki", best_rotation, to_d3(ref_conf)
            )
            rotated_ref_conf = rotated_ref_conf[:, None, :, :]

        if (torch.sum(modes * modes, dim=(2, 3)) == 0).any():
            pass

        else:
            # renormalization of the prediction to the optimal length to minimize the loss
            c_bi_numerator = (
                torch.sum(coverage * modes_truth_rotated * modes, dim=(2, 3))
                / seq_lengths[:, None]
            )
            c_bi_denominator = (
                torch.sum(coverage * modes**2, dim=(2, 3)) / seq_lengths[:, None]
            )
            c_bi_optimal = c_bi_numerator / c_bi_denominator
            c_bi_optimal = c_bi_optimal.float()
            modes = modes * c_bi_optimal[:, :, None, None]

        if is_training and self.reweight_centrality:
            # give more importance to the central mouvments
            centrality_measures = compute_centrality(modes_truth_rotated, seq_lengths)
        else:
            centrality_measures = torch.ones(
                (batch_size, nb_modes), dtype=torch.float, device="cuda"
            )

        individual_loss_modes = (
            torch.sum(
                torch.sum(
                    coverage
                    * (
                        centrality_measures[..., None, None]
                        * (modes_truth_rotated - modes) ** 2
                    ),
                    dim=(2, 3),
                ),
                dim=1,
            )
            / seq_lengths
        )
        loss_modes = individual_loss_modes.mean()
        total_torque = (
            torch.sum(torch.cross(rotated_ref_conf, modes, dim=-1), dim=2)
            / seq_lengths[:, None, None]
        )
        individual_loss_torque = torch.sum(total_torque**2, dim=(1, 2))
        loss_torque = individual_loss_torque.mean()

        total_loss = coeff_modes * loss_modes + coeff_torque * loss_torque

        loss_dict = {
            "total_loss": total_loss,
            "loss_modes": loss_modes,
            "loss_torque": loss_torque,
            "individual_loss_modes": individual_loss_modes,
            "individual_loss_torque": individual_loss_torque,
        }

        if return_modes:
            # return the modes aligned on the gt
            with torch.no_grad():
                modes = apply_best_permutations(modes, best_permutations)
                modes = torch.einsum(
                    "bij,bklj->bkli", best_rotation.permute(0, 2, 1), modes
                )
                modes *= best_signs.unsqueeze(-1).unsqueeze(-1)
                loss_dict["modes"] = modes

        return loss_dict

    def find_best_rotation(self, query, target, coverage):
        # eigvals shape (batch, nb_modes)
        batch_size, nb_modes, _, _ = query.shape
        best_rotations = torch.empty((batch_size, 3, 3), device=query.device)
        best_signs = torch.empty((batch_size, nb_modes), device=query.device)
        best_metric = torch.full((batch_size,), float("-inf"), device=query.device)
        best_permutations = [None] * batch_size
        query_permuted = torch.empty_like(
            query
        )  # to store the best permutation per sample
        # target and query shape (batch, nb_modes, seq_len, 3)

        for permutation in itertools.permutations(range(nb_modes)):

            permuted_query = query[:, permutation, :, :]
            prods = torch.einsum(
                "bijk,bijl->bikl", coverage * permuted_query, target
            )  # (batch,num_modes,3,3) # la suite est invariante au changement de signe d'un mode
            sums = torch.einsum(
                "ij,bjkl->bikl", self.signs_tensor, prods
            )  # (batch,nb_signs(num_modes),3,3)
            U, S, V = torch.linalg.svd(sums, full_matrices=False)
            trace_S = torch.sum(S, dim=-1)

            if self.allow_reflections is False:
                d = torch.det(torch.matmul(U, V).transpose(-2, -1))
                flip = d < 0
                if flip.any():
                    indices = torch.where(flip)
                    for i, j in zip(indices[0], indices[1]):
                        V[i, j, -1] *= -1
                        S[i, j, -1] *= -1
            max_trace_indices = torch.argmax(trace_S, dim=1)
            batch_indices = torch.arange(batch_size).to(max_trace_indices.device)
            best_U = U[batch_indices, max_trace_indices]
            best_V = V[batch_indices, max_trace_indices]
            current_best_rotations = torch.matmul(best_U, best_V).transpose(-2, -1)
            current_best_signs = self.signs_tensor[max_trace_indices]

            for i in range(batch_size):
                if trace_S[i, max_trace_indices[i]] > best_metric[i]:
                    best_metric[i] = trace_S[i, max_trace_indices[i]]
                    best_rotations[i] = current_best_rotations[i]
                    best_signs[i] = current_best_signs[i]
                    best_permutations[i] = permutation
                    query_permuted[i] = permuted_query[i]

            if self.allow_permutation is False:
                break

        return best_rotations, best_signs, best_permutations, query_permuted
