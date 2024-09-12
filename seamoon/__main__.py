import typer

app = typer.Typer()


@app.command()
def precompute_w_gt(
    prefixes: str = typer.Option(
        "data_test/example_list.txt",
        "--prefixes",
        "-p",
        help="Path to the text file containing prefixes",
    ),
    bin_dir: str = typer.Option(
        "data_test/bin_dance", "--bin-dir", "-b", help="Path to the binary directory"
    ),
    aln_dir: str = typer.Option(
        "data_test/aln_dance", "--aln-dir", "-a", help="Path to the alignment directory"
    ),
    output_dir: str = typer.Option(
        "data_test/training_data",
        "--output-dir",
        "-o",
        help="Path to the output directory",
    ),
    emb_model: str = typer.Option(
        "ProstT5",
        "--emb-model",
        "-e",
        help="Model to use for embeddings (ProstT5 or ESM)",
    ),
):
    """
    Precompute embeddings and ground truth data for training from the DANCE binaries and alignments files
    """
    from .data.precompute import precompute_w_gt

    precompute_w_gt(prefixes, bin_dir, aln_dir, output_dir, emb_model)


@app.command()
def precompute_from_fasta(
    input_files: str = typer.Option(
        ...,
        "--input-files",
        "-i",
        help="Path to the multi-FASTA file or text file containing paths to individual FASTA files",
    ),
    output_dir: str = typer.Option(
        ..., "--output-dir", "-o", help="Directory to save the computed embeddings"
    ),
    emb_model: str = typer.Option(
        "ProstT5",
        "--emb-model",
        "-e",
        help="Model to use for embeddings (ProstT5 or ESM)",
    ),
):
    """
    Precompute embeddings from a multi-FASTA file or a text file containing paths to individual FASTA files, for inference without torque alignment only
    """
    from .data.precompute import precompute_from_fasta

    precompute_from_fasta(input_files, output_dir, emb_model)


@app.command()
def precompute_w_pdb(
    input_files: str = typer.Option(
        ...,
        "--input-files",
        "-i",
        help="Path to the text file containing paths to PDB files",
    ),
    output_dir: str = typer.Option(
        ...,
        "--output-dir",
        "-o",
        help="Directory to save the computed embeddings and C-alpha coordinates",
    ),
    emb_model: str = typer.Option(
        "ProstT5",
        "--emb-model",
        "-e",
        help="Model to use for embeddings (ProstT5 or ESM)",
    ),
):
    """
    Precompute embeddings and extract C-alpha coordinates from PDB files, for inference only.
    """

    from .data.precompute import precompute_w_pdb

    precompute_w_pdb(input_files, output_dir, emb_model)


@app.command()
def train(
    config_path: str = typer.Option(
        "weights/config_ProstT5.json",
        "--config-path",
        "-c",
        help="Path to the configuration file",
    )
):
    """
    Train the model using the parameters specified in the configuration file.
    """
    from .training.train import train_loop

    train_loop(config_path)


@app.command()
def infer(
    model_path: str = typer.Option(
        "weights/ProstT5_5ref_2024-06-18_22-46-27/best_model_epoch_257.pt",
        "--model-path",
        "-m",
        help="Path to the state dictionary file of the model",
    ),
    config_file: str = typer.Option(
        "weights/config_ProstT5.json",
        "--config-file",
        "-c",
        help="Path to the configuration file",
    ),
    list_path: str = typer.Option(
        "data_test/split/test_list.txt",
        "--list-path",
        "-l",
        help="Path to the list file containing the names of the samples to infer",
    ),
    precomputed_path: str = typer.Option(
        "data_test/training_data",
        "--precomputed-path",
        "-p",
        help="Path to the precomputed data",
    ),
    output_path: str = typer.Option(
        "prediction", "--output-path", "-o", help="Path to the output directory"
    ),
    batch_size: int = typer.Option(
        1, "--batch-size", "-b", help="Batch size to use during inference"
    ),
    torque_mode: bool = typer.Option(
        False,
        "--torque-mode",
        "-t",
        help="Alignment of the ground truth using the torque minimization",
    ),
    device: str = typer.Option(
        "cuda", "--device", "-d", help="Device to use, either 'cuda' or 'cpu'"
    ),
):
    """
    Infer the modes using the specified model and data.
    """

    from .eval.eval import evaluate

    evaluate(
        model_path,
        config_file,
        list_path,
        precomputed_path,
        output_path,
        batch_size,
        torque_mode,
        infer_only=True,
        device=device,
    )


@app.command()
def evaluate(
    model_path: str = typer.Option(
        "weights/ProstT5_5ref_2024-06-18_22-46-27/best_model_epoch_257.pt",
        "--model-path",
        "-m",
        help="Path to the model file",
    ),
    config_file: str = typer.Option(
        "weights/config_ProstT5.json",
        "--config-file",
        "-c",
        help="Path to the configuration file",
    ),
    list_path: str = typer.Option(
        "data_test/split/test_list.txt",
        "--list-path",
        "-l",
        help="Path to the list file containing the names of the samples to infer",
    ),
    precomputed_path: str = typer.Option(
        "data_test/training_data",
        "--precomputed-path",
        "-p",
        help="Path to the precomputed data",
    ),
    output_path: str = typer.Option(
        "prediction", "--output-path", "-o", help="Path to the output directory"
    ),
    batch_size: int = typer.Option(
        1, "--batch-size", "-b", help="Batch size to use during evaluation"
    ),
    torque_mode: bool = typer.Option(
        False,
        "--torque-mode",
        "-t",
        help="Alignment of the ground truth using the torque minimization",
    ),
    device: str = typer.Option(
        "cuda", "--device", "-d", help="Device to use, either 'cuda' or 'cpu'"
    ),
):
    """
    Evaluate the specified model using the specified data.
    """
    from .eval.eval import evaluate

    evaluate(
        model_path,
        config_file,
        list_path,
        precomputed_path,
        output_path,
        batch_size,
        torque_mode,
        infer_only=False,
        device=device,
    )


if __name__ == "__main__":
    app()
