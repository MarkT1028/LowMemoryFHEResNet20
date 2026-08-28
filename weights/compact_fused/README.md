# Compact fused weights

These `FHEWGHT1` files contain the same convolution-plus-batch-normalization
coefficients as the upstream expanded plaintext files, but without a fixed
spatial layout. This lets the native 64x64 path build CKKS plaintext diagonals
for its own tensor width at runtime.

Each little-endian file contains:

1. the 8-byte magic value `FHEWGHT1`;
2. three `uint32` values: output channels, input channels, kernel size;
3. row-major `float64` weights `[out][in][kernel][kernel]`;
4. one `float64` bias per output channel.

Regenerate and verify them from the repository root with:

```bash
python3 tools/export_compact_fused_weights.py
python3 tools/validate_compact_fused_weights.py
```
