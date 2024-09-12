# SeaMoon: Prediction of Molecular Motions Based on Language Models

SeaMoon is a framework designed to predict molecular linear motions using language models such as **ProstT5** and **ESM**.

## Quick Start

### Setup Environment

1. Create a new conda environment and activate it:

   ```bash
   conda create --name seamoon python=3.11.9
   conda activate seamoon
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

### Test Run

A small test dataset of 100 elements is included in `data_set` to validate all main functions. Start by precomputing embeddings:

```bash
python -m seamoon precompute-w-gt
```

If you wish to skip precomputing, 10 precomputed elements are provided in data_set/training_data. You can launch the **infer** and **evaluate** directly.

- **Infer**:
  ```bash
  python -m seamoon infer
  ```

- **Evaluate**:
  ```bash
  python -m seamoon evaluate
  ```

The full dataset from the paper can be downloaded [here]().

## Usage

### Precompute Embeddings

Precompute embeddings using either FASTA or PDB files, with options to specify the embedding model:

- **From FASTA**:
  ```bash
  python -m seamoon precompute-from-fasta --input-files [path-to-fasta-or-list] --output-dir [output-directory] --emb-model [ProstT5|ESM]
  ```

- **From PDB**:
  ```bash
  python -m seamoon precompute-from-pdb --input-files [path-to-pdb-list] --output-dir [output-directory] --emb-model [ProstT5|ESM]
  ```

- **From DANCE binaries and alignments** (with ground truth to train the model):
  ```bash
  python -m seamoon precompute-w-gt --prefixes [file-with-prefixes] --bin-dir [binary-dir] --aln-dir [alignment-dir] --output-dir [output-directory] --emb-model [ProstT5|ESM]
  ```

NB:The torque minimization requires a structure file, so use **precompute-from-pdb** if you wish to use it. 

### Training

```bash
python -m seamoon train --config-path [path-to-config-file]
```

### Inference

```bash
python -m seamoon infer --model-path [path-to-model] --config-file [path-to-config] --list-path [path-to-list] --precomputed-path [path-to-precomputed-data] --output-path [output-directory] --batch-size [batch-size] --torque-mode [true|false] --device [cuda|cpu]
```

### Evaluation

```bash
python -m seamoon evaluate --model-path [path-to-model] --config-file [path-to-config] --list-path [path-to-list] --precomputed-path [path-to-precomputed-data] --output-path [output-directory] --batch-size [batch-size] --torque-mode [true|false] --device [cuda|cpu]
```

## Citation


## License
