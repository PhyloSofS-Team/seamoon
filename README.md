# SeaMoon: Prediction of Molecular Motions Based on Language Models

SeaMoon is a deep learning framework that predicts protein motions from their amino acid sequences. It leverages embeddings of protein language models, such as the sequence-only-based [ESM-2](https://github.com/facebookresearch/esm) ([Lin et al. 2022](https://www.science.org/doi/abs/10.1126/science.ade2574)), the multimodal [ESM3](https://github.com/evolutionaryscale/esm) ([Hayes et al. 2024](https://www.biorxiv.org/content/10.1101/2024.07.01.600583v1)), or the sequence-structure bilingual [ProstT5](https://github.com/mheinzinger/ProstT5) ([Heinzinger et al. 2023](https://www.biorxiv.org/content/10.1101/2023.07.23.550085v2)). Given a query protein sequence, SeaMoon outputs sets of 3D displacements vectors for each C-alpha atom within an invariant subspace, which can be interpreted as **linear motions**.  

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

3. If you wish to use **--torque-mode** during inference or evaluation, you will need a working version of the [Wolfram Engine](https://www.wolfram.com/engine/). Make sure to specify the path to your `WolframKernel` at line 30 of `eval.py`. We used Wolfram Engine v14.0.

### Test Run

A small test dataset of 100 elements is included in `data_set` to validate all main functions. If you wish to precompute all of them, you can use:

```bash
python -m seamoon precompute-w-gt
```

If you wish to **skip precomputing**, 10 precomputed elements are provided in data_set/training_data. You can launch the **infer** and **evaluate** directly.

- **Infer**:
  ```bash
  python -m seamoon infer
  ```

- **Evaluate**:
  ```bash
  python -m seamoon evaluate
  ```

The full dataset from the paper can be downloaded [here]() (soon).

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

NB:The torque minimization requires a structure file, so use **precompute-from-pdb** or **precompute-w-gt** if you wish to use it. 

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
