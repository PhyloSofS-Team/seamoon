import torch
from torch.utils.data import Dataset


def to_d3(mat):
    ## mat must be(nb_conf, nb_res*3)
    ## output is (nb_conf, nb_res, 3)
    return torch.reshape(mat, (mat.shape[0], mat.shape[1] // 3, 3))


class CustomDataset(Dataset):
    def __init__(self, list_path, precomputed_path, emb_model, num_modes=3, noise=0.0):

        self.model = emb_model
        self.precomputed_path = precomputed_path
        self.num_modes = num_modes
        self.noise = noise
        with open(list_path, "r") as f:
            self.sample_names = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.sample_names)

    def __getitem__(self, idx):

        sample_name = self.sample_names[idx]
        data_path = f"{self.precomputed_path}/{sample_name}_{self.model}_data.pt"
        sample_data = torch.load(
            data_path, map_location=torch.device("cpu"), weights_only=False
        )

        eigvals = sample_data.get("eigvals", None)
        eigvects = sample_data.get("eigvects", None)
        ref = sample_data.get("ref", None)

        emb = sample_data["emb"]
        name = sample_data["data"][0][0]
        sequence = sample_data["data"][0][1]
        coverage = torch.tensor(sample_data["data"][0][2])

        if self.num_modes is not None and eigvals is not None and eigvects is not None:
            eigvals = eigvals[: self.num_modes]
            eigvects = eigvects[:, : self.num_modes]

        if self.noise > 0 and eigvects is not None:
            noise_factor = self.noise
            proportional_noise = torch.randn_like(eigvects) * eigvects * noise_factor
            eigvects += proportional_noise

            proportional_noise = torch.randn_like(emb) * emb * noise_factor
            emb += proportional_noise
            eigvects /= torch.norm(eigvects, dim=0, keepdim=True)

        seq_length = torch.count_nonzero(coverage, dim=0)

        if eigvects is not None:
            eigvects *= seq_length**0.5  # -> the norm of the 3N mode is sqrt(N)

        full_name = f"{sample_name}"

        sample = {
            "coverage": coverage,
            "emb": emb,
            "name": name,
            "sequence": sequence,
            "full_name": full_name,
        }

        if eigvals is not None:
            sample["eigvals"] = eigvals
        if eigvects is not None:
            sample["eigvects"] = eigvects
        if ref is not None:
            sample["ref"] = ref

        return sample
