import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import random
import math
import torch.nn as nn
import sys


def custom_collate_fn(batch):
    emb_list = [item["emb"].squeeze(0) for item in batch]
    coverage_list = [item["coverage"] for item in batch]
    name_list = [item["name"] for item in batch]
    full_name_list = [item["full_name"] for item in batch]
    sequences_list = [item["sequence"] for item in batch]
    seq_lengths = torch.tensor([len(seq) for seq in sequences_list])
    if all("eigvects" in item for item in batch):
        eigvects_list = [item["eigvects"] for item in batch]
        padded_eigvects = pad_sequence(eigvects_list, batch_first=True, padding_value=0)
    else:
        padded_eigvects = None

    if all("eigvals" in item for item in batch):
        eigvals_list = [item["eigvals"] for item in batch]
        eigvals = torch.stack(eigvals_list)
    else:
        eigvals = None

    if all("ref" in item for item in batch):
        ref_list = [item["ref"].squeeze(-1) for item in batch]
        padded_ref = pad_sequence(ref_list, batch_first=True, padding_value=0).float()
    else:
        padded_ref = None

    padded_emb = pad_sequence(emb_list, batch_first=True, padding_value=0)
    padded_coverage = pad_sequence(coverage_list, batch_first=True, padding_value=0)

    result = {
        "emb": padded_emb,
        "coverage": padded_coverage,
        "sequences": sequences_list,
        "name": name_list,
        "seq_lengths": seq_lengths,
        "full_name": full_name_list,
    }

    if padded_eigvects is not None:
        result["eigvects"] = padded_eigvects.permute(0, 2, 1)
    if eigvals is not None:
        result["eigvals"] = eigvals
    if padded_ref is not None:
        result["ref"] = padded_ref

    return result


class NoisyLengthSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, batch_size, noise_std=1):
        self.dataset = dataset
        self.batch_size = batch_size
        self.noise_std = noise_std
        self.length_cache = {}
        self.is_first_epoch = True

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        if self.is_first_epoch:
            # First epoch: build the length cache
            for idx in range(len(self.dataset)):
                sample = self.dataset[idx]
                emb_len = len(sample["emb"])
                self.length_cache[idx] = emb_len
            self.is_first_epoch = False

        # Add random noise to lengths and sort by these noisy lengths
        noisy_lengths = [
            (idx, length + random.gauss(0, self.noise_std))
            for idx, length in self.length_cache.items()
        ]
        noisy_lengths.sort(key=lambda x: x[1])

        sorted_indices = [idx for idx, _ in noisy_lengths]
        batches = [
            sorted_indices[i : i + self.batch_size]
            for i in range(0, len(sorted_indices), self.batch_size)
        ]

        if self.noise_std > 0:
            random.shuffle(batches)

        for batch in batches:
            for idx in batch:
                yield idx


def create_data_loader(
    dataset, batch_size, num_workers=0, pin_memory=False, noise_std=1
):
    sampler = NoisyLengthSampler(dataset, batch_size, noise_std=noise_std)

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        collate_fn=custom_collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return data_loader
