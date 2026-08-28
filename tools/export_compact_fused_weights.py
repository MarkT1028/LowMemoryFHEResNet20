#!/usr/bin/env python3
"""Recover compact Conv+BN weights from the repository's expanded slot files.

The upstream repository stores every convolution diagonal as a 4K/8K/16K
plaintext vector.  Those vectors are tied to the original 32x32 packing.  This
script reverses that expansion once and writes small, resolution-independent
fused-weight files for the native 64x64 implementation.

No model download and no PyTorch installation are required.  NumPy is the only
dependency.
"""

from __future__ import annotations

import argparse
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np


MAGIC = b"FHEWGHT1"


@dataclass(frozen=True)
class LayerSpec:
    output_name: str
    source_prefix: str
    in_channels: int
    out_channels: int
    width: int
    kernel_size: int = 3
    transition: bool = False
    initial: bool = False


LAYERS = [
    LayerSpec("initial", "conv1bn1", 3, 16, 32, initial=True),
    LayerSpec("layer1_conv1", "layer1-conv1bn1", 16, 16, 32),
    LayerSpec("layer1_conv2", "layer1-conv2bn2", 16, 16, 32),
    LayerSpec("layer2_conv1", "layer2-conv1bn1", 16, 16, 32),
    LayerSpec("layer2_conv2", "layer2-conv2bn2", 16, 16, 32),
    LayerSpec("layer3_conv1", "layer3-conv1bn1", 16, 16, 32),
    LayerSpec("layer3_conv2", "layer3-conv2bn2", 16, 16, 32),
    LayerSpec("layer4_conv1", "layer4-conv1bn1", 16, 32, 32, transition=True),
    LayerSpec("layer4_downsample", "layer4dx-conv1bn1", 16, 32, 32, kernel_size=1, transition=True),
    LayerSpec("layer4_conv2", "layer4-conv2bn2", 32, 32, 16),
    LayerSpec("layer5_conv1", "layer5-conv1bn1", 32, 32, 16),
    LayerSpec("layer5_conv2", "layer5-conv2bn2", 32, 32, 16),
    LayerSpec("layer6_conv1", "layer6-conv1bn1", 32, 32, 16),
    LayerSpec("layer6_conv2", "layer6-conv2bn2", 32, 32, 16),
    LayerSpec("layer7_conv1", "layer7-conv1bn1", 32, 64, 16, transition=True),
    LayerSpec("layer7_downsample", "layer7dx-conv1bn1", 32, 64, 16, kernel_size=1, transition=True),
    LayerSpec("layer7_conv2", "layer7-conv2bn2", 64, 64, 8),
    LayerSpec("layer8_conv1", "layer8-conv1bn1", 64, 64, 8),
    LayerSpec("layer8_conv2", "layer8-conv2bn2", 64, 64, 8),
    LayerSpec("layer9_conv1", "layer9-conv1bn1", 64, 64, 8),
    LayerSpec("layer9_conv2", "layer9-conv2bn2", 64, 64, 8),
]


def load_vector(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    return np.loadtxt(path, delimiter=",", dtype=np.float64)


def repeated_nonzero_value(vector: np.ndarray, block: int, area: int) -> float:
    """Return the repeated fused coefficient in one masked channel block."""
    segment = vector[block * area : (block + 1) * area]
    nonzero = segment[np.abs(segment) > 1e-15]
    if nonzero.size == 0:
        return 0.0

    # np.savetxt preserves each repeated float exactly.  The most frequent
    # non-zero value is the coefficient; the other values can only come from a
    # neighbouring block at a cyclic boundary.
    values, counts = np.unique(nonzero, return_counts=True)
    return float(values[np.argmax(counts)])


def extract_initial(spec: LayerSpec, weights_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    area = spec.width * spec.width
    weights = np.zeros((spec.out_channels, spec.in_channels, 9), dtype=np.float64)
    for output_channel in range(spec.out_channels):
        for kernel_index in range(9):
            vector = load_vector(
                weights_dir / f"{spec.source_prefix}-ch{output_channel}-k{kernel_index + 1}.bin"
            )
            for input_channel in range(spec.in_channels):
                weights[output_channel, input_channel, kernel_index] = repeated_nonzero_value(
                    vector, input_channel, area
                )

    bias_vector = load_vector(weights_dir / f"{spec.source_prefix}-bias.bin")
    bias = np.array(
        [repeated_nonzero_value(bias_vector, channel, area) for channel in range(spec.out_channels)],
        dtype=np.float64,
    )
    return weights, bias


def extract_square(spec: LayerSpec, weights_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    area = spec.width * spec.width
    channels = spec.in_channels
    weights = np.zeros((channels, channels, 9), dtype=np.float64)

    for diagonal in range(channels):
        for kernel_index in range(9):
            vector = load_vector(
                weights_dir / f"{spec.source_prefix}-ch{diagonal}-k{kernel_index + 1}.bin"
            )
            vector = np.roll(vector, -(area * diagonal))
            for output_channel in range(channels):
                input_channel = (output_channel + diagonal) % channels
                weights[output_channel, input_channel, kernel_index] = repeated_nonzero_value(
                    vector, output_channel, area
                )

    bias_vector = load_vector(weights_dir / f"{spec.source_prefix}-bias.bin")
    bias = np.array(
        [repeated_nonzero_value(bias_vector, channel, area) for channel in range(channels)],
        dtype=np.float64,
    )
    return weights, bias


def extract_transition(spec: LayerSpec, weights_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    area = spec.width * spec.width
    kernel_elements = spec.kernel_size * spec.kernel_size
    input_channels = spec.in_channels
    weights = np.zeros(
        (spec.out_channels, input_channels, kernel_elements), dtype=np.float64
    )

    for diagonal in range(input_channels):
        for output_group in range(2):
            file_channel = diagonal + output_group * input_channels
            # The second half is shifted by one slot in Algorithm 2 before the
            # even-position mask is applied.
            applied_shift = area * diagonal - output_group
            for kernel_index in range(kernel_elements):
                vector = load_vector(
                    weights_dir
                    / f"{spec.source_prefix}-ch{file_channel}-k{kernel_index + 1}.bin"
                )
                vector = np.roll(vector, -applied_shift)
                for local_output in range(input_channels):
                    output_channel = output_group * input_channels + local_output
                    input_channel = (local_output + diagonal) % input_channels
                    weights[output_channel, input_channel, kernel_index] = repeated_nonzero_value(
                        vector, local_output, area
                    )

    bias = np.zeros(spec.out_channels, dtype=np.float64)
    for output_group in range(2):
        bias_vector = load_vector(
            weights_dir / f"{spec.source_prefix}-bias{output_group + 1}.bin"
        )
        for local_output in range(input_channels):
            bias[output_group * input_channels + local_output] = repeated_nonzero_value(
                bias_vector, local_output, area
            )
    return weights, bias


def write_layer(path: Path, weights: np.ndarray, bias: np.ndarray) -> None:
    out_channels, in_channels, kernel_elements = weights.shape
    kernel_size = int(round(kernel_elements**0.5))
    if kernel_size * kernel_size != kernel_elements:
        raise ValueError(f"Invalid kernel shape: {weights.shape}")

    with path.open("wb") as output:
        output.write(MAGIC)
        output.write(struct.pack("<III", out_channels, in_channels, kernel_size))
        output.write(np.asarray(weights, dtype="<f8").tobytes(order="C"))
        output.write(np.asarray(bias, dtype="<f8").tobytes(order="C"))


def export(weights_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    total_parameters = 0

    for spec in LAYERS:
        print(f"Extracting {spec.output_name} ...", flush=True)
        if spec.initial:
            weights, bias = extract_initial(spec, weights_dir)
        elif spec.transition:
            weights, bias = extract_transition(spec, weights_dir)
        else:
            weights, bias = extract_square(spec, weights_dir)

        if not np.isfinite(weights).all() or not np.isfinite(bias).all():
            raise ValueError(f"Non-finite coefficient in {spec.output_name}")
        if np.count_nonzero(weights) == 0:
            raise ValueError(f"All-zero weights in {spec.output_name}")

        write_layer(output_dir / f"{spec.output_name}.fwgt", weights, bias)
        total_parameters += weights.size + bias.size

    print(
        f"Wrote {len(LAYERS)} layers ({total_parameters:,} fused parameters) to {output_dir}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights-dir", type=Path, default=Path("weights"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("weights") / "compact_fused"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export(args.weights_dir, args.output_dir)
