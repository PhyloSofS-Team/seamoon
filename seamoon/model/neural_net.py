import torch
from torch import nn
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F


def lengths_to_mask(lengths, max_len=None, dtype=None):
    """
    Converts a "lengths" tensor to its binary mask representation.

    Based on: https://discuss.pytorch.org/t/how-to-generate-variable-length-mask/23397

    :lengths: N-dimensional tensor
    :returns: N*max_len dimensional tensor. If max_len==None, max_len=max(lengtsh)
    """
    assert len(lengths.shape) == 1, "Length shape should be 1 dimensional."
    max_len = max_len or lengths.max().item()
    mask = torch.arange(max_len, device=lengths.device, dtype=lengths.dtype).expand(
        len(lengths), max_len
    ) < lengths.unsqueeze(1)
    if dtype is not None:
        mask = torch.as_tensor(mask, dtype=dtype, device=lengths.device)
    return mask


class MaskedBatchNorm1d(nn.BatchNorm1d):
    """
    From https://gist.github.com/amiasato/902fc14afa37a7537386f7b0c5537741

    Masked verstion of the 1D Batch normalization.

    Based on: https://github.com/ptrblck/pytorch_misc/blob/20e8ea93bd458b88f921a87e2d4001a4eb753a02/batch_norm_manual.py

    Receives a N-dim tensor of sequence lengths per batch element
    along with the regular input for masking.

    Check pytorch's BatchNorm1d implementation for argument details.
    """

    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
    ):
        super(MaskedBatchNorm1d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats
        )

    def forward(self, inp, lengths):
        self._check_input_dim(inp)

        exponential_average_factor = 0.0

        # We transform the mask into a sort of P(inp) with equal probabilities
        # for all unmasked elements of the tensor, and 0 probability for masked
        # ones.
        mask = lengths_to_mask(lengths, max_len=inp.shape[-1], dtype=inp.dtype)
        n = mask.sum()
        mask = mask / n
        mask = mask.unsqueeze(1).expand(inp.shape)

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training and n > 1:
            # Here lies the trick. Using Var(X) = E[X^2] - E[X]^2 as the biased
            # variance, we do not need to make any tensor shape manipulation.
            # mean = E[X] is simply the sum-product of our "probability" mask with the input...
            mean = (mask * inp).sum([0, 2])
            # ...whereas Var(X) is directly derived from the above formulae
            # This should be numerically equivalent to the biased sample variance
            var = (mask * inp**2).sum([0, 2]) - mean**2
            with torch.no_grad():
                self.running_mean = (
                    exponential_average_factor * mean
                    + (1 - exponential_average_factor) * self.running_mean
                )
                # Update running_var with unbiased var
                self.running_var = (
                    exponential_average_factor * var * n / (n - 1)
                    + (1 - exponential_average_factor) * self.running_var
                )
        else:
            mean = self.running_mean
            var = self.running_var

        inp = (inp - mean[None, :, None]) / (torch.sqrt(var[None, :, None] + self.eps))
        if self.affine:
            inp = inp * self.weight[None, :, None] + self.bias[None, :, None]

        return inp


class ConvHead(nn.Module):
    def __init__(
        self,
        in_features: int,
        num_modes: int,
        hidden_sizes: List[int],
        kernel_sizes: List[int],
        bias=True,
        eos_idx: Optional[int] = None,
        dropout_coeff=0.2,
        qr_reg=False,
        use_bn=False,
        **kwargs
    ):
        super().__init__()
        self.in_features = in_features
        self.num_modes = num_modes
        self.qr_reg = qr_reg
        self.eos_idx = eos_idx

        assert len(hidden_sizes) == len(
            kernel_sizes
        ), "hidden_sizes and kernel_sizes must have the same length"
        conv_layers = []
        bn_layers = []
        in_channels = in_features

        for out_channels, kernel_size in zip(hidden_sizes, kernel_sizes):

            conv_layers.append(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding="same",
                    bias=bias,
                )
            )
            if use_bn:
                bn_layers.append(MaskedBatchNorm1d(out_channels))
            in_channels = out_channels

        self.mode_output_layers = nn.ModuleList(
            [
                nn.Conv1d(in_channels, 3, kernel_size=1, padding="same", bias=bias)
                for _ in range(num_modes)
            ]
        )

        if use_bn:
            # self.embedding_bn = MaskedBatchNorm1d(in_features)
            bn_layers.append(MaskedBatchNorm1d(3))

        self.conv_layers = nn.ModuleList(conv_layers)
        self.bn_layers = nn.ModuleList(bn_layers) if use_bn else None
        self.leaky = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout_coeff)

    def forward(self, embedding, seq_lengths):

        # in the beginning embedding is (batch_size, seq_len, emb_dim)

        batch_size = embedding.size(0)
        seqlen = torch.max(seq_lengths)
        if hasattr(self, "embedding_bn"):
            embedding = self.embedding_bn(embedding.permute(0, 2, 1), seq_lengths)
        else:
            embedding = embedding.permute(0, 2, 1)
        out = embedding
        for n, conv in enumerate(self.conv_layers):
            out = conv(out)
            if self.bn_layers and n != len(self.conv_layers) - 1:
                out = self.bn_layers[n](out, seq_lengths)
            if n != len(self.conv_layers) - 1:
                out = self.leaky(out)
                out = self.dropout(out)

        mode_outputs = [mode_layer(out) for mode_layer in self.mode_output_layers]
        out = torch.stack(mode_outputs, dim=1)

        out = out.permute(0, 1, 3, 2)
        modes = out[:, :, 1:-1, :]

        if self.qr_reg:
            out_reshaped = modes.reshape(batch_size, 3 * seqlen, self.num_modes)
            q_r_product = torch.empty_like(out_reshaped)

            for i in range(batch_size):
                q, r = torch.linalg.qr(out_reshaped[i], mode="reduced")
                q_r_product[i] = q @ torch.diag_embed(torch.diag(r))

            modes = q_r_product.reshape(batch_size, self.num_modes, seqlen, 3)

        return modes


HEADS = {"ConvHead": ConvHead}
