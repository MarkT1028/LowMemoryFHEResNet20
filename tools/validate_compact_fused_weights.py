#!/usr/bin/env python3
"""Validate compact weights against the upstream 32x32 expanded encoding."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "weights"
COMPACT = WEIGHTS / "compact_fused"


def rotate(vector: np.ndarray, amount: int) -> np.ndarray:
    # OpenFHE's positive packed rotation is a left rotation.
    return np.roll(vector, -amount)


def load_text(name: str) -> np.ndarray:
    return np.loadtxt(WEIGHTS / name, delimiter=",", dtype=np.float64)


def load_compact(name: str) -> tuple[np.ndarray, np.ndarray]:
    with (COMPACT / f"{name}.fwgt").open("rb") as source:
        if source.read(8) != b"FHEWGHT1":
            raise ValueError(f"Invalid compact weight file: {name}")
        out_channels, in_channels, kernel_size = struct.unpack("<III", source.read(12))
        weight_count = out_channels * in_channels * kernel_size * kernel_size
        weights = np.frombuffer(source.read(weight_count * 8), dtype="<f8").copy()
        bias = np.frombuffer(source.read(out_channels * 8), dtype="<f8").copy()
    return weights.reshape(out_channels, in_channels, -1), bias


def spatial_rotations(vector: np.ndarray, width: int, kernel_size: int) -> list[np.ndarray]:
    if kernel_size == 1:
        return [vector]
    return [
        rotate(vector, row * width + column)
        for row in (-1, 0, 1)
        for column in (-1, 0, 1)
    ]


def boundary_mask(width: int, kernel_index: int, channels: int) -> np.ndarray:
    row_offset = kernel_index // 3 - 1
    column_offset = kernel_index % 3 - 1
    single = np.zeros((width, width), dtype=np.float64)
    for row in range(width):
        for column in range(width):
            input_row = row + row_offset
            input_column = column + column_offset
            if 0 <= input_row < width and 0 <= input_column < width:
                single[row, column] = 1.0
    return np.tile(single.reshape(-1), channels)


def stride2_mask(width: int, channels: int) -> np.ndarray:
    single = np.zeros((width, width), dtype=np.float64)
    single[::2, ::2] = 1.0
    return np.tile(single.reshape(-1), channels)


def generic_group(
    source: np.ndarray,
    weights: np.ndarray,
    width: int,
    input_channels: int,
    output_start: int,
    group_channels: int,
) -> np.ndarray:
    area = width * width
    rotations = spatial_rotations(source, width, weights.shape[2] ** 0.5)
    result = None
    for diagonal in range(group_channels):
        diagonal_sum = np.zeros_like(source)
        for kernel_index, rotated in enumerate(rotations):
            encoded = np.zeros_like(source)
            spatial_mask = (
                np.ones_like(source)
                if len(rotations) == 1
                else boundary_mask(width, kernel_index, group_channels)
            )
            for input_local in range(group_channels):
                output_local = (input_local - diagonal) % group_channels
                output_channel = output_start + output_local
                input_channel = input_local
                if output_channel < weights.shape[0] and input_channel < input_channels:
                    begin = input_local * area
                    end = begin + area
                    encoded[begin:end] = (
                        weights[output_channel, input_channel, kernel_index]
                        * spatial_mask[begin:end]
                    )
            diagonal_sum += rotated * encoded
        result = diagonal_sum if result is None else result + diagonal_sum
        result = rotate(result, -area)
    return result


def generic_sharded(
    source: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
    width: int,
    group_channels: int,
) -> list[np.ndarray]:
    """Clear simulation of FHEController::convbn_sharded."""
    area = width * width
    in_channels = weights.shape[1]
    out_channels = weights.shape[0]
    input_shards = (in_channels + group_channels - 1) // group_channels
    output_shards = (out_channels + group_channels - 1) // group_channels
    result = [np.zeros(group_channels * area) for _ in range(output_shards)]

    for input_shard in range(input_shards):
        packed = np.zeros(group_channels * area)
        for local_channel in range(group_channels):
            global_channel = input_shard * group_channels + local_channel
            if global_channel < in_channels:
                packed[local_channel * area : (local_channel + 1) * area] = source[
                    global_channel
                ].reshape(-1)
        rotations = spatial_rotations(packed, width, int(weights.shape[2] ** 0.5))

        for output_shard in range(output_shards):
            pair_result = None
            for diagonal in range(group_channels):
                diagonal_sum = np.zeros_like(packed)
                for kernel_index, rotated in enumerate(rotations):
                    mask = boundary_mask(width, kernel_index, group_channels)
                    if len(rotations) == 1:
                        mask.fill(1.0)
                    encoded = np.zeros_like(packed)
                    for input_local in range(group_channels):
                        output_local = (input_local - diagonal) % group_channels
                        input_channel = input_shard * group_channels + input_local
                        output_channel = output_shard * group_channels + output_local
                        if input_channel < in_channels and output_channel < out_channels:
                            begin = input_local * area
                            encoded[begin : begin + area] = (
                                weights[output_channel, input_channel, kernel_index]
                                * mask[begin : begin + area]
                            )
                    diagonal_sum += rotated * encoded
                pair_result = (
                    diagonal_sum if pair_result is None else pair_result + diagonal_sum
                )
                pair_result = rotate(pair_result, -area)
            result[output_shard] += pair_result

    for output_shard, packed in enumerate(result):
        for local_channel in range(group_channels):
            output_channel = output_shard * group_channels + local_channel
            if output_channel < out_channels:
                packed[local_channel * area : (local_channel + 1) * area] += bias[
                    output_channel
                ]
    return result


def direct_convolution(
    source: np.ndarray, weights: np.ndarray, bias: np.ndarray
) -> np.ndarray:
    out_channels, in_channels, kernel_elements = weights.shape
    kernel_size = int(kernel_elements**0.5)
    radius = kernel_size // 2
    _, width, _ = source.shape
    result = np.zeros((out_channels, width, width))
    for output_channel in range(out_channels):
        result[output_channel].fill(bias[output_channel])
        for input_channel in range(in_channels):
            for kernel_index in range(kernel_elements):
                row_offset = kernel_index // kernel_size - radius
                column_offset = kernel_index % kernel_size - radius
                coefficient = weights[output_channel, input_channel, kernel_index]
                for row in range(width):
                    input_row = row + row_offset
                    if not 0 <= input_row < width:
                        continue
                    for column in range(width):
                        input_column = column + column_offset
                        if 0 <= input_column < width:
                            result[output_channel, row, column] += (
                                source[input_channel, input_row, input_column]
                                * coefficient
                            )
    return result


def validate_native_sharding(name: str, width: int, group_channels: int) -> None:
    rng = np.random.default_rng(640128)
    weights, bias = load_compact(name)
    source = rng.normal(size=(weights.shape[1], width, width))
    expected = direct_convolution(source, weights, bias)
    packed = generic_sharded(source, weights, bias, width, group_channels)
    actual = np.concatenate(
        [shard.reshape(group_channels, width, width) for shard in packed], axis=0
    )[: weights.shape[0]]
    error = float(np.max(np.abs(actual - expected)))
    print(f"{name} native sharding: max error {error:.3e}")
    if error > 1e-10:
        raise AssertionError(f"{name} native sharding")


def compact_stride2(shard: np.ndarray, width: int, group_channels: int) -> np.ndarray:
    """Clear simulation of one shard in downsample_stride2_sharded."""
    area = width * width
    compact_width = width // 2
    compact_area = area // 4
    horizontal = shard.copy()
    for step in (1 << power for power in range(int(np.log2(compact_width)))):
        horizontal = horizontal + rotate(horizontal, step)
        if step * 2 < compact_width:
            mask = np.array(
                [1.0 if index % (4 * step) < 2 * step else 0.0 for index in range(shard.size)]
            )
            horizontal *= mask

    compacted_rows = np.zeros_like(shard)
    for row in range(compact_width):
        mask = np.zeros_like(shard)
        for channel in range(group_channels):
            begin = channel * area + row * compact_width
            mask[begin : begin + compact_width] = 1.0
        compacted_rows += horizontal * mask
        if row + 1 < compact_width:
            horizontal = rotate(horizontal, 2 * width - compact_width)

    compacted_channels = np.zeros_like(shard)
    for channel in range(group_channels):
        mask = np.zeros_like(shard)
        begin = channel * area
        mask[begin : begin + compact_area] = 1.0
        compacted_channels += compacted_rows * mask
        compacted_channels = rotate(compacted_channels, -(area - compact_area))
    return rotate(compacted_channels, (area - compact_area) * group_channels)


def validate_native_downsample(width: int, group_channels: int, output_group: int) -> None:
    rng = np.random.default_rng(width * 1000 + group_channels)
    shard_count = output_group // group_channels
    source = rng.normal(size=(shard_count, group_channels, width, width))
    source[:, :, 1::2, :] = 0.0
    source[:, :, :, 1::2] = 0.0
    compact_area = (width // 2) ** 2
    chunks = [compact_stride2(shard.reshape(-1), width, group_channels) for shard in source]
    merged = []
    merge_factor = output_group // group_channels
    for target in range(shard_count // merge_factor):
        packed = np.zeros(output_group * compact_area)
        for position in range(merge_factor):
            chunk = chunks[target * merge_factor + position]
            packed += rotate(chunk, -position * group_channels * compact_area)
        merged.append(packed)
    actual = np.concatenate(
        [chunk.reshape(output_group, width // 2, width // 2) for chunk in merged]
    )
    expected = source[:, :, ::2, ::2].reshape(-1, width // 2, width // 2)
    error = float(np.max(np.abs(actual - expected)))
    print(f"downsample {width}->{width // 2}, group {group_channels}->{output_group}: max error {error:.3e}")
    if error > 1e-10:
        raise AssertionError("native stride-2 packing")


def validate_final_layer_128() -> None:
    """Check the four-shard global average + fully-connected decomposition."""
    rng = np.random.default_rng(128010)
    source = rng.normal(size=(64, 32, 32))
    fc = load_text("fc.bin").reshape(64, 10)
    expected = source.mean(axis=(1, 2)) @ fc

    actual = np.zeros(10)
    for shard in range(4):
        first = shard * 16
        last = first + 16
        actual += source[first:last].mean(axis=(1, 2)) @ fc[first:last]

    error = float(np.max(np.abs(actual - expected)))
    print(f"128x128 four-shard final layer: max error {error:.3e}")
    if error > 1e-10:
        raise AssertionError("128x128 final layer")


def validate_square(name: str, source_prefix: str, channels: int, width: int) -> None:
    rng = np.random.default_rng(20240828)
    source = rng.normal(size=channels * width * width)
    rotations = spatial_rotations(source, width, 3)
    area = width * width

    expected = None
    for diagonal in range(channels):
        diagonal_sum = np.zeros_like(source)
        for kernel_index, rotated in enumerate(rotations):
            diagonal_sum += rotated * load_text(
                f"{source_prefix}-ch{diagonal}-k{kernel_index + 1}.bin"
            )
        expected = diagonal_sum if expected is None else expected + diagonal_sum
        expected = rotate(expected, -area)
    expected += load_text(f"{source_prefix}-bias.bin")

    weights, bias = load_compact(name)
    actual = generic_group(source, weights, width, channels, 0, channels)
    actual += np.repeat(bias, area)
    error = float(np.max(np.abs(actual - expected)))
    print(f"{name}: max error {error:.3e}")
    if error > 1e-10:
        raise AssertionError(name)


def validate_transition(
    name: str,
    source_prefix: str,
    input_channels: int,
    width: int,
    kernel_size: int,
) -> None:
    rng = np.random.default_rng(640064)
    source = rng.normal(size=input_channels * width * width)
    rotations = spatial_rotations(source, width, kernel_size)
    area = width * width
    weights, bias = load_compact(name)
    keep = stride2_mask(width, input_channels)

    for output_group in range(2):
        expected = None
        for diagonal in range(input_channels):
            diagonal_sum = np.zeros_like(source)
            for kernel_index, rotated in enumerate(rotations):
                file_channel = diagonal + output_group * input_channels
                diagonal_sum += rotated * load_text(
                    f"{source_prefix}-ch{file_channel}-k{kernel_index + 1}.bin"
                )
            expected = diagonal_sum if expected is None else expected + diagonal_sum
            expected = rotate(expected, -area)
        expected += load_text(f"{source_prefix}-bias{output_group + 1}.bin")

        actual = generic_group(
            source,
            weights,
            width,
            input_channels,
            output_group * input_channels,
            input_channels,
        )
        actual += np.repeat(
            bias[output_group * input_channels : (output_group + 1) * input_channels],
            area,
        )
        actual *= keep
        error = float(np.max(np.abs(actual - expected)))
        print(f"{name} group {output_group}: max error {error:.3e}")
        if error > 1e-10:
            raise AssertionError(f"{name} group {output_group}")


if __name__ == "__main__":
    validate_square("layer1_conv1", "layer1-conv1bn1", 16, 32)
    validate_transition("layer4_conv1", "layer4-conv1bn1", 16, 32, 3)
    validate_transition("layer4_downsample", "layer4dx-conv1bn1", 16, 32, 1)
    validate_transition("layer7_conv1", "layer7-conv1bn1", 32, 16, 3)
    validate_transition("layer7_downsample", "layer7dx-conv1bn1", 32, 16, 1)
    validate_native_sharding("initial", 8, 4)
    validate_native_sharding("initial", 8, 1)
    validate_native_sharding("layer4_conv1", 8, 4)
    validate_native_sharding("layer5_conv1", 4, 16)
    validate_native_downsample(128, 1, 4)
    validate_native_downsample(64, 4, 16)
    validate_native_downsample(32, 16, 64)
    validate_final_layer_128()
    print("Compact fused weights match the upstream expanded packing.")
