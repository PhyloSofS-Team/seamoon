# Precompute the torch dictionaries used for training
# containing the embeddings and the ground truth data
from transformers import T5Tokenizer, T5EncoderModel
import esm
from tqdm import tqdm
import numpy as np
import torch
import os
import re
import json
from Bio import AlignIO, PDB
from Bio.SeqUtils import seq1
import sys
import argparse

def move_to_cpu(data_dict):
    for key, value in data_dict.items():
        if isinstance(value, torch.Tensor) and value.is_cuda:
            data_dict[key] = value.cpu()
    return data_dict

def eigss_svd(X):
    # input shape (nb_conf, nb_res*3)
    # Vt.T dimensions (nb_res*3, modes)
    _, S, Vt = np.linalg.svd(X.T, full_matrices=False)
    explained_variance_ = (S**2) / (len(X) - 1)
    total_var = explained_variance_.sum()
    explained_variance_ratio = explained_variance_ / total_var
    return explained_variance_ratio, Vt.T


def to_d3(mat):
    ## mat must be(nb_conf, nb_res*3)
    ## output is (nb_conf, nb_res, 3)
    if isinstance(mat, np.ndarray):
        return np.reshape(mat, (len(mat), len(mat.T) // 3, 3))
    else:
        raise ValueError("Unrecognized matrix type")


def to_d2(mat):
    ## mat must be (nb_conf, nb_res, 3)
    ## output is (nb_conf, nb_res*3)
    if isinstance(mat, np.ndarray):
        shape = np.shape(mat)
        return np.reshape(mat, (shape[0], shape[1] * 3))
    else:
        raise ValueError("Unrecognized matrix type")


def apply_query_coord_on_missing_data(coord_mat, K_mat, query_id):
    # input coord_mat (nb_res*3, nb_conf)
    add_mat = np.logical_not(K_mat).astype(float) * coord_mat[:, query_id, None]
    coord_mat = coord_mat * K_mat
    coord_mat += add_mat
    return coord_mat


def get_query(coordinates, gaps, query_index, coverage=False, normalize=False):
    # Input shape coords: (nb_res*3, nb_conf)
    # Input shape gaps: (nb_res*3, nb_conf)
    query_gaps = gaps[:, query_index : query_index + 1]
    indices = np.where(query_gaps[:, 0] == 1)[0]
    filtered_coordinates = coordinates[indices, :]
    filtered_gaps = gaps[indices, :]
    coordinates = apply_query_coord_on_missing_data(
        filtered_coordinates, filtered_gaps, query_index
    )

    if normalize:
        query_coords = coordinates[:, query_index : query_index + 1]
        deviations = coordinates - query_coords
        squared_deviations = deviations**2
        mean_squared_deviation = np.sum(squared_deviations, axis=1, keepdims=True) / (
            squared_deviations.shape[1] - 1
        )
        std_from_query = np.sqrt(mean_squared_deviation)
        std_from_query[std_from_query == 0] = 1
        coordinates = deviations / std_from_query
    else:
        coordinates = coordinates - coordinates[:, query_index : query_index + 1]

    if coverage:

        filtered_gaps_3d = to_d3(filtered_gaps.T)
        coverage_subset = (
            np.sum(filtered_gaps_3d[..., 0], axis=0) / filtered_gaps_3d.shape[0]
        )
        coordinates_3d = to_d3(coordinates.T)
        coordinates_3d = coordinates_3d * coverage_subset[None, :, None]
        coordinates = to_d2(coordinates_3d).T

    return coordinates


def load_tensor(filename):
    with open(filename, "rb") as f:
        numModels = np.frombuffer(f.read(8), dtype=np.int64)[0]
        numSeqs = np.frombuffer(f.read(8), dtype=np.int64)[0]
        numCoords = np.frombuffer(f.read(8), dtype=np.int64)[0]
        data_shape = (numModels, numSeqs * numCoords)
        data = np.frombuffer(f.read(), dtype=np.float64)
        return data.reshape(data_shape)


def load_mask(filename):
    with open(filename, "rb") as f:
        numModels = np.frombuffer(f.read(8), dtype=np.int64)[0]
        numSeqs = np.frombuffer(f.read(8), dtype=np.int64)[0]
        tensor = (
            np.frombuffer(f.read(numModels * numSeqs), dtype=np.uint8)
            .astype(bool)
            .reshape((numModels, numSeqs))
        )
        tensor_3d = np.repeat(tensor[:, :, np.newaxis], 3, axis=2)
        tensor = np.reshape(tensor_3d, (numModels, numSeqs * 3))
        return tensor


def insert_zeros(vector, indices):
    num_zeros = len(indices)
    shape = list(vector.shape)
    shape[0] += num_zeros

    combined_vector = np.zeros(shape, dtype=vector.dtype)

    indices_arr = np.array(indices)
    original_indices = np.delete(np.arange(shape[0]), indices_arr)

    combined_vector[original_indices, ...] = vector

    return combined_vector


def parse_covariance(coord_file, aln_file, save_w_gaps=False):
    al = AlignIO.read(aln_file, "fasta")
    aln = np.array(al)

    coords = load_tensor(coord_file).T
    mask = load_mask(
        coord_file.replace(".bin", "_mask.bin")
    ).T  # transpose to have (nb_res*3, nb_conf)
    mask_copy = mask.copy()
    col_to_keep = np.where(np.sum(mask, axis=1) > 1)[0]
    mask = mask[col_to_keep]
    coords = coords[col_to_keep]
    mask_copy_3d = to_d3(mask_copy.T)[:, :, 0].T

    col_to_keep_3d = np.where(np.sum(mask_copy_3d, axis=1) > 1)[0]
    aln = aln[:, col_to_keep_3d]
    ref_seq = aln[0]  # Reference sequence is the first sequence
    ref_seq_string = "".join(ref_seq)
    no_gap_or_X_mask = (ref_seq != "-") & (ref_seq != "X")

    insertion = np.where(no_gap_or_X_mask == False)[0]

    filtered_aln = aln[:, no_gap_or_X_mask]
    sequences = ["".join(seq) for seq in filtered_aln]
    num_sequences = len(filtered_aln)
    no_gap_or_X_mask = (filtered_aln != "-") & (filtered_aln != "X")
    coverage = np.sum(no_gap_or_X_mask, axis=0) / num_sequences
    chain_names = [record.id for record in al]

    eigss = []
    data_list = []

    query_gaps = mask[:, 0:1]
    indices = np.where(query_gaps[:, 0] == 1)[0]
    ref_coordinates = coords[indices, 0:1]
    eigenvalues, eigenvectors = eigss_svd(get_query(coords, mask, 0, coverage=True))
    eigenvalues = np.sqrt(eigenvalues)
    eigenvectors = eigenvectors[:, :5]
    eigenvalues = eigenvalues[:5]

    if save_w_gaps is False:
        data_output = (chain_names[0], sequences[0], coverage)
    else:
        coverage = insert_zeros(coverage, insertion)
        eigenvectors_3d = to_d3(eigenvectors.T)
        eigenvectors_3d = np.transpose(eigenvectors_3d, (1, 0, 2))
        eigenvectors_3d = insert_zeros(eigenvectors_3d, insertion)
        eigenvectors_3d = np.transpose(eigenvectors_3d, (1, 0, 2))
        eigenvectors = to_d2(eigenvectors_3d).T
        data_output = (chain_names[0], ref_seq_string, coverage)

    data_list.append([data_output])
    eigs = (eigenvalues, eigenvectors)  # only the 5 first modes
    eigss.append(eigs)
    return data_list, eigss, ref_coordinates


def precompute_w_gt(
    prefixes_txt,
    bin_dir_path,
    aln_dir_path,
    output_dir,
    emb_model="ProstT5",
    save_w_gaps=False,
    serialize_on_cpu = True
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if emb_model == "ProstT5":
        tokenizer = T5Tokenizer.from_pretrained("Rostlab/ProstT5", do_lower_case=False)
        model = T5EncoderModel.from_pretrained("Rostlab/ProstT5").to(device)
        model.half()
        model.eval()

    elif emb_model == "ESM":
        model, alphabet = getattr(esm.pretrained, "esm2_t33_650M_UR50D")()
        model = model.to(device)
        batch_converter = alphabet.get_batch_converter()
        model.eval()

    if save_w_gaps:
        emb_model += "_w_gaps"

    with open(prefixes_txt, "r") as file:
        prefixes = file.readlines()

    for prefix in tqdm(prefixes):
        prefix = prefix.strip()
        coord_file = os.path.join(bin_dir_path, prefix + "_raw_coords_ca.bin")
        aln_file = os.path.join(aln_dir_path, prefix + "_aln.fa")
        data_list, eigss, ref = parse_covariance(
            coord_file, aln_file, save_w_gaps=save_w_gaps
        )

        for data, eigs in zip(data_list, eigss):
            eigvals, eigvects = eigs

            if "ProstT5" in emb_model:
                sequence = data[0][1].upper()
                sequence = " ".join(list(re.sub(r"[UZOB]", "X", sequence)))
                sequence = "<AA2fold> " + sequence
                ids = tokenizer.encode_plus(
                    sequence, add_special_tokens=True, return_tensors="pt"
                ).to(device)

                # Generate embeddings
                with torch.no_grad():
                    embedding_repr = model(
                        ids.input_ids, attention_mask=ids.attention_mask
                    )
                emb = embedding_repr.last_hidden_state.squeeze(0)

            if "ESM" in emb_model:
                batch_labels, batch_strs, batch_tokens = batch_converter(
                    [(data[0][0], data[0][1])]
                )
                batch_tokens = batch_tokens.to(device)
                with torch.no_grad():
                    results = model(
                        batch_tokens,
                        repr_layers=[model.num_layers],
                        return_contacts=False,
                    )
                token_representations = results["representations"][model.num_layers]
                emb = token_representations.squeeze(0)

            ref = torch.from_numpy(ref).to(device).squeeze()

            # Checking for sequence length consistency
            assert (
                emb.shape[0] - 2 == eigvects.shape[0] // 3
            ), f"Sequence length mismatch between emb ({emb.shape[0]}) and eigvects ({eigvects.shape[0]}) for file {file}"
            assert (
                emb.shape[0] - 2 == ref.shape[0] // 3
            ), f"Sequence length mismatch between emb ({emb.shape[0]}) and ref ({ref.shape[0]}) for file {file}"
            sample_data = {
                "data": data,
                "emb": emb,
                "eigvals": torch.from_numpy(eigvals[:5]).to(device),
                "eigvects": torch.from_numpy(eigvects[:, :5]).to(device),
                "ref": ref,
            }

            if serialize_on_cpu:
                sample_data = move_to_cpu(sample_data)

            torch.save(
                sample_data, os.path.join(output_dir, f"{prefix}_{emb_model}_data.pt")
            )


def precompute_from_fasta(input_file_or_fasta, output_dir, emb_model="ProstT5",serialize_on_cpu = True):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if emb_model == "ProstT5":
        tokenizer = T5Tokenizer.from_pretrained("Rostlab/ProstT5", do_lower_case=False)
        model = T5EncoderModel.from_pretrained("Rostlab/ProstT5").to(device)
        model.half()
        model.eval()

    elif emb_model == "ESM":
        model, alphabet = getattr(esm.pretrained, "esm2_t33_650M_UR50D")()
        model = model.to(device)
        batch_converter = alphabet.get_batch_converter()
        model.eval()

    fasta_sequences = []
    sequence_names = []
    if input_file_or_fasta.endswith(".txt"):
        with open(input_file_or_fasta, "r") as file_list:
            fasta_files = [line.strip() for line in file_list.readlines()]

        for fasta_file in fasta_files:
            with open(fasta_file, "r") as file:
                fasta_content = file.readlines()
                fasta_sequences.extend(
                    [seq.strip() for seq in fasta_content if not seq.startswith(">")]
                )
                sequence_names.extend(
                    [seq[1:].strip() for seq in fasta_content if seq.startswith(">")]
                )
    else:
        with open(input_file_or_fasta, "r") as file:
            fasta_content = file.readlines()
            fasta_sequences = [
                seq.strip() for seq in fasta_content if not seq.startswith(">")
            ]
            sequence_names = [
                seq[1:].strip() for seq in fasta_content if seq.startswith(">")
            ]

    for idx, (sequence, seq_name) in enumerate(
        tqdm(zip(fasta_sequences, sequence_names))
    ):
        sequence = sequence.strip()

        if "ProstT5" in emb_model:
            sequence = sequence.upper()
            sequence = " ".join(list(re.sub(r"[UZOB]", "X", sequence)))
            sequence = "<AA2fold> " + sequence
            ids = tokenizer.encode_plus(
                sequence, add_special_tokens=True, return_tensors="pt"
            ).to(device)
            with torch.no_grad():
                embedding_repr = model(ids.input_ids, attention_mask=ids.attention_mask)
            emb = embedding_repr.last_hidden_state.squeeze(0)
        elif "ESM" in emb_model:
            batch_labels, batch_strs, batch_tokens = batch_converter(
                [("seq", sequence)]
            )
            batch_tokens = batch_tokens.to(device)
            with torch.no_grad():
                results = model(
                    batch_tokens, repr_layers=[model.num_layers], return_contacts=False
                )
            emb = results["representations"][model.num_layers].squeeze(0)

        data = [(seq_name, sequence, np.ones(len(sequence)))]

        sample_data = {"data": data, "emb": emb}

        file_name = seq_name.replace(" ", "_").replace("/", "_")
        
        if serialize_on_cpu:
            sample_data = move_to_cpu(sample_data)

        torch.save(
            sample_data, os.path.join(output_dir, f"{file_name}_{emb_model}_data.pt")
        )


def precompute_w_pdb(pdb_file_list, output_dir, emb_model="ProstT5",serialize_on_cpu = True):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if emb_model == "ProstT5":
        tokenizer = T5Tokenizer.from_pretrained("Rostlab/ProstT5", do_lower_case=False)
        emb_model_instance = T5EncoderModel.from_pretrained("Rostlab/ProstT5").to(
            device
        )
        emb_model_instance.half()
        emb_model_instance.eval()

    elif emb_model == "ESM":
        emb_model_instance, alphabet = getattr(esm.pretrained, "esm2_t33_650M_UR50D")()
        emb_model_instance = emb_model_instance.to(device)
        batch_converter = alphabet.get_batch_converter()
        emb_model_instance.eval()

    parser = PDB.PDBParser(QUIET=True)

    with open(pdb_file_list, "r") as file_list:
        pdb_files = [line.strip() for line in file_list.readlines()]

    for pdb_file in tqdm(pdb_files):
        structure = parser.get_structure("pdb", pdb_file)
        first_model = structure[0]
        sequence = []
        ca_coords = []

        for chain in first_model:
            for residue in chain:
                if PDB.is_aa(residue, standard=True):
                    sequence.append(seq1(residue.resname))
                    ca = residue["CA"].get_coord()
                    ca_coords.extend(ca)

        sequence = "".join(sequence)
        ca_coords = torch.tensor(ca_coords, dtype=torch.float32)

        if "ProstT5" in emb_model:
            sequence = sequence.upper()
            sequence = " ".join(list(re.sub(r"[UZOB]", "X", sequence)))
            sequence = "<AA2fold> " + sequence
            ids = tokenizer.encode_plus(
                sequence, add_special_tokens=True, return_tensors="pt"
            ).to(device)
            with torch.no_grad():
                embedding_repr = emb_model_instance(
                    ids.input_ids, attention_mask=ids.attention_mask
                )
            emb = embedding_repr.last_hidden_state.squeeze(0)

        elif "ESM" in emb_model:
            batch_labels, batch_strs, batch_tokens = batch_converter(
                [("seq", sequence)]
            )
            batch_tokens = batch_tokens.to(device)
            with torch.no_grad():
                results = emb_model_instance(
                    batch_tokens,
                    repr_layers=[emb_model_instance.num_layers],
                    return_contacts=False,
                )
            emb = results["representations"][emb_model_instance.num_layers].squeeze(0)

        data = [(pdb_file, sequence, np.ones(len(sequence)))]

        sample_data = {"data": data, "emb": emb, "ref": ca_coords}

        file_name = (
            os.path.basename(pdb_file)
            .replace(".pdb", "")
            .replace(" ", "_")
            .replace("/", "_")
        )
        if serialize_on_cpu:
            sample_data = move_to_cpu(sample_data)

        torch.save(
            sample_data, os.path.join(output_dir, f"{file_name}_{emb_model}_data.pt")
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate ground truth data and compute embeddings."
    )
    parser.add_argument(
        "--prefixes",
        type=str,
        default="data_test/exemple_list.txt",
        help="Path to the text file containing prefixes",
    )
    parser.add_argument(
        "--bin_dir",
        type=str,
        default="data_test/bin_dance",
        help="Path to the binary directory",
    )
    parser.add_argument(
        "--aln_dir",
        type=str,
        default="data_test/aln_dance",
        help="Path to the alignment directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data_test/training_data",
        help="Path to the output directory",
    )
    parser.add_argument(
        "--emb_model",
        type=str,
        default="ProstT5",
        help="Model to use for embeddings (ProstT5 or ESM)",
    )
    args = parser.parse_args()
    precompute_data(
        args.prefixes, args.bin_dir, args.aln_dir, args.output_dir, args.emb_model
    )
