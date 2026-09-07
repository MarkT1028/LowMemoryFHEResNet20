# Encrypted Image Classification with Low Memory Footprint using Fully Homomorphic Encryption
<center>
<img src="imgs/console.png" alt="Console presentation image" width=85% >
</center>

<img src="https://github.com/narger-ef/LowMemoryFHEResNet20/actions/workflows/cmake-multi-platform.yml/badge.svg" alt="CMake build result" width=30% >

---

This repository contains a OpenFHE-based project that implements an encrypted version of the ResNet20 model, used to classify encrypted CIFAR-10 images.

The reference paper for this work is [Encrypted Image Classification with Low Memory Footprint using Fully Homomorphic Encryption](https://doi.org/10.1142/S0129065724500254). A preprint is available on [ePrint Archive](https://eprint.iacr.org/2024/460)

The key idea behind this work is to propose a solution to run a CNN in relative small time ($<5$ minutes on my MacBook M1 Pro with 16GB RAM) and, moreover, to use a small amount of memory. 

De Castro et al. [6] showed that memory is currently the main bottleneck to be addressed in FHE circuits, although most of the works do not consider it as a metric when building FHE solutions.

Existing works use a lot of memory ([4]: $\approx$ 100GB, [5]: $\approx$ 500GB), while this implementation uses less than 16GB, making it usable by normal users.

## Technical details

The circuit is based on the RNS-CKKS implementation [1] in OpenFHE [2].
We propose an approach to convolutions called _Optimized Vector Encoding_, which enabled to evaluate a convolution using only five Automorphism Keys, needed to rotate the values of the ciphertext. These are the heaviest objects in memory, therefore by minimizing the use of these keys, it is possible to reduce the memory footprint of the application.

Experiments show that it is possible to evaluate the circuit in less than 5 minutes (in [3] it requires more than 6 minutes) and by using a small amount of RAM, from 10GB to 15GB, depending on the desired precision and speed.



## Architecture

The program simulates a server-client interaction in which the server is assumed to be honest-but-curious. 

Both client and server agree on a pair public-secrey key that is based on Ring Learning With Errors (RLWE) [3]: a post-quantum hard problem defined as follows:

Given a polynomial ring $\mathcal{R} = Z[X]/(X^N + 1)$ and small Gaussian distribution $\chi$:

* Secret key: $s \gets \chi$ is a polynomial with random coefficients in $\mathcal{R}$
* Public key: $(a, b)$, where $a$ is a random polynomial in $\mathcal{R}$, and $b = a \cdot s + e$, with $e \gets \chi$

The idea is to use $b$, which, without the secret key $s$ would look like a random element, to encrypt the image.

1) The client encrypts the image using the public key.

<img src="imgs/arch1.png" alt="Architecture description 1" width=45%>


2) The server performs computations on it (following the definition of Fully Homomorphic Encryption)

<img src="imgs/arch2.png" alt="Architecture description 2" width=45%>


3) The server returns an encrypted vector containing the output of the last fully connected layer. The client is able to decrypt it and see the result

<img src="imgs/arch3.png" alt="Architecture description 3" width=45%>

4) The client finds the index of the maximum value and, using a dictionary, find the classified label

<img src="imgs/arch4.png" alt="Architecture description 4" width=45%>


## Reproduce on another Windows computer (WSL2)

This section is the portable setup guide for every experiment branch in this
fork. Only one clone is required. Switch branches in that clone, then rebuild.
The words **plaintext weights** below refer only to model parameters: the input
image is encrypted in every FHE inference branch.

### 0) Choose the experiment branch

| Resolution | Model-parameter mode | Branch | Key directory |
|---|---|---|---|
| 32x32 | original plaintext weights | `main` | `keys_exp3` |
| 32x32 | CKKS-encrypted weights | `feature/32x32-encrypted-weights` | `keys_exp3` |
| 64x64 | original plaintext weights | `feature/native-64x64` | `keys_exp3_64` |
| 64x64 | CKKS-encrypted weights | `feature/64x64-encrypted-weights` | `keys_exp3_64` |
| 128x128 | original plaintext weights | `feature/native-128x128` | `keys_exp3_128` |
| 128x128 | CKKS-encrypted weights | `feature/128x128-encrypted-weights` | `keys_exp3_128` |
| 256x256 | original plaintext weights | `feature/native-256x256` | `keys_exp3_256` |
| 256x256 | CKKS-encrypted weights | `feature/256x256-encrypted-weights` | `keys_exp3_256` |

The plaintext- and encrypted-weight branches at the same resolution use the
same OpenFHE context and evaluation keys. Generate that resolution's key set
only once. Never mix files from different key directories.

### 1) Hardware and free-space planning

This project is CPU-only; CUDA and a discrete GPU are not required. Eight CPU
cores work, while approximately 16 logical cores are recommended. These are
practical WSL2 allocations rather than cryptographic minimums:

| Resolution | Recommended WSL RAM | Recommended swap | Measured Experiment 3 key size |
|---|---:|---:|---:|
| 32x32 | 16 GiB or more | 8 GiB | about 28 GiB |
| 64x64 | 18 GiB or more | 8-12 GiB | about 31 GiB |
| 128x128 | 18-21 GiB or more | 12 GiB | about 30 GiB |
| 256x256 | 24 GiB preferred | 16 GiB | about 52 GiB |

A 256x256 probe has reached about 20.85 GiB resident memory. A known 32 GiB
Windows host completed it with 21 GiB assigned to WSL plus 12 GiB swap, but
that is tight; 48-64 GiB host RAM is preferable for 256x256. Reserve at least
60 GiB free space for one 32/64/128 key set or 90 GiB for 256x256. Keeping all
four Experiment 3 key directories requires about 141 GiB before builds, logs,
and checkpoints, so reserve roughly 200 GiB.

### 2) Install and size WSL2

Run this once in an Administrator PowerShell window if Ubuntu is not installed:

```powershell
wsl --install -d Ubuntu-24.04
```

Create or edit `%UserProfile%\.wslconfig` on Windows. For a 32 GiB host with
16 logical processors, the configuration already validated for this project is:

```ini
[wsl2]
memory=21GB
processors=16
swap=12GB
```

Do not assign more processors than the new computer has. After changing the
file, apply it from PowerShell and reopen Ubuntu:

```powershell
wsl --shutdown
```

Inside Ubuntu, confirm the resources and disk space:

```bash
free -h
nproc
df -h
```

Cloning below the WSL home directory is usually faster than `/mnt/c` or
`/mnt/e`. If the WSL virtual disk lacks space, a large Windows drive may be used,
but key loading and checkpoint I/O can be slower.

### 3) Install the compiler and OpenFHE 1.0.4

Run inside Ubuntu:

```bash
sudo apt update
sudo apt install -y build-essential cmake git libomp-dev pkg-config

mkdir -p "$HOME/src"
git clone --branch v1.0.4 --depth 1 --recursive \
  https://github.com/openfheorg/openfhe-development.git \
  "$HOME/src/openfhe-1.0.4"

cmake -S "$HOME/src/openfhe-1.0.4" \
  -B "$HOME/src/openfhe-1.0.4/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_UNITTESTS=OFF \
  -DBUILD_EXAMPLES=OFF \
  -DBUILD_BENCHMARKS=OFF

cmake --build "$HOME/src/openfhe-1.0.4/build" -j "$(nproc)"
sudo cmake --install "$HOME/src/openfhe-1.0.4/build"
sudo ldconfig
```

Use OpenFHE **v1.0.4** for these experiments. Newer releases can change CKKS
level consumption and may reproduce the `DropLastElement` failure described
later in this README.

### 4) Clone this fork and build the selected branch

The following example selects the 64x64 plaintext-weight branch. Replace the
branch name with the exact entry from the table in step 0:

```bash
cd "$HOME"
git clone https://github.com/MarkT1028/LowMemoryFHEResNet20.git
cd LowMemoryFHEResNet20
git fetch --all --prune
git switch feature/native-64x64
git branch --show-current

cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build -j "$(nproc)"
mkdir -p logs
cd build
```

After switching to another experiment branch, return to the repository root and
run the two `cmake` commands again. Deleting the existing build directory is not
normally necessary.

### 5) Generate the matching Experiment 3 keys

Run exactly one block from the `build` directory. Key generation is required
once per resolution and can take several minutes. It intentionally stops rather
than overwriting an existing key directory.

```bash
# 32x32: main or feature/32x32-encrypted-weights
set -o pipefail
/usr/bin/time -v ./LowMemoryFHEResNet20 generate_keys 3 verbose 1 \
  2>&1 | tee ../logs/keygen_exp3_32.log
```

```bash
# 64x64: feature/native-64x64 or feature/64x64-encrypted-weights
set -o pipefail
/usr/bin/time -v ./LowMemoryFHEResNet20 \
  generate_keys 3 resolution 64 verbose 1 \
  2>&1 | tee ../logs/keygen_exp3_64.log
```

```bash
# 128x128: feature/native-128x128 or feature/128x128-encrypted-weights
set -o pipefail
/usr/bin/time -v ./LowMemoryFHEResNet20 \
  generate_keys 3 resolution 128 verbose 1 \
  2>&1 | tee ../logs/keygen_exp3_128.log
```

```bash
# 256x256: feature/native-256x256 or feature/256x256-encrypted-weights
set -o pipefail
/usr/bin/time -v ./LowMemoryFHEResNet20 \
  generate_keys 3 resolution 256 verbose 1 \
  2>&1 | tee ../logs/keygen_exp3_256.log
```

A successful command ends with `Exit status: 0` and creates the key directory
shown in step 0. Key directories are ignored by Git and contain a secret key;
do not commit or upload them. When moving keys to another computer, copy the
entire directory as one unit and protect `secret-key.txt`.

### 6) Run the matching inference

Enable pipeline failure reporting once, then run the command that corresponds
to the currently checked-out branch:

```bash
set -o pipefail
```

```bash
# main: 32x32, original plaintext model weights
/usr/bin/time -v ./LowMemoryFHEResNet20 \
  load_keys 3 input "inputs/luis.png" verbose 1 \
  2>&1 | tee ../logs/infer_exp3_32_plain_weights.log
```

```bash
# feature/32x32-encrypted-weights
/usr/bin/time -v ./LowMemoryFHEResNet20 \
  load_keys 3 weights encrypted input "inputs/luis.png" verbose 1 \
  2>&1 | tee ../logs/infer_exp3_32_encrypted_weights.log
```

```bash
# feature/native-64x64
/usr/bin/time -v ./LowMemoryFHEResNet20 \
  load_keys 3 resolution 64 input "inputs/cat_64x64.png" verbose 1 \
  2>&1 | tee ../logs/infer_exp3_64_plain_weights.log
```

```bash
# feature/64x64-encrypted-weights
/usr/bin/time -v ./LowMemoryFHEResNet20 \
  load_keys 3 resolution 64 weights encrypted \
  input "inputs/cat_64x64.png" verbose 1 \
  2>&1 | tee ../logs/infer_exp3_64_encrypted_weights.log
```

```bash
# feature/native-128x128
/usr/bin/time -v ./LowMemoryFHEResNet20 \
  load_keys 3 resolution 128 input "inputs/dog_128x128.png" verbose 1 \
  2>&1 | tee ../logs/infer_exp3_128_plain_weights.log
```

```bash
# feature/128x128-encrypted-weights
/usr/bin/time -v ./LowMemoryFHEResNet20 \
  load_keys 3 resolution 128 weights encrypted \
  input "inputs/dog_128x128.png" verbose 1 \
  2>&1 | tee ../logs/infer_exp3_128_encrypted_weights.log
```

```bash
# feature/native-256x256
/usr/bin/time -v ./LowMemoryFHEResNet20 \
  load_keys 3 resolution 256 input "inputs/horse_256x256.png" verbose 1 \
  2>&1 | tee ../logs/infer_exp3_256_plain_weights.log
```

```bash
# feature/256x256-encrypted-weights
/usr/bin/time -v ./LowMemoryFHEResNet20 \
  load_keys 3 resolution 256 weights encrypted \
  input "inputs/horse_256x256.png" verbose 1 \
  2>&1 | tee ../logs/infer_exp3_256_encrypted_weights.log
```

Always use an image whose dimensions exactly match the selected resolution.
Success means the program prints ten output values, a predicted class and index,
then `/usr/bin/time` reports `Exit status: 0`. With `set -o pipefail`, a failure
inside the program is not hidden by `tee`. The 128x128 encrypted-weight run has
taken about two hours on the development machine; 256x256 runs can take several
hours. Keep the terminal and computer awake.

For encrypted-weight branches, a quick operation check is available after key
generation:

```bash
./LowMemoryFHEResNet20 load_keys 3 resolution 64 \
  test_encrypted_weights verbose 1
```

Change `64` to `128` or `256` on those branches. On the 32x32 encrypted-weight
branch, omit `resolution 64`. The test must print
`Encrypted model-parameter self-test: PASS` before a long inference is started.

## How to run

### Native 256x256 path

This branch adds an exact `256x256` encrypted path. The input is **not** resized
to 128x128 or 32x32 before encryption. Its packing is:

- input: `3 x 256 x 256` in three ciphertexts, one channel per ciphertext;
- stage 1: `16 x 256 x 256` in sixteen ciphertexts;
- stage 2: `32 x 128 x 128` in eight ciphertexts, four channels each;
- stage 3: `64 x 64 x 64` in four ciphertexts, sixteen channels each;
- every ciphertext uses 65536 CKKS slots and a `2^17` ring dimension.

Because the ring is twice as large as the 128x128 path, 256x256 needs an
independent key directory. `keys_exp3_128` cannot be reused. On a 32 GiB Windows
host, allocate as much WSL memory as practical and keep swap enabled. The
development machine's current 21 GiB WSL RAM plus 12 GiB swap completed the
Experiment 3 key generation in about 10:55 with a 20.85 GiB peak resident set
and no swap use. Keep the swap allocation as safety margin for full inference;
it is substantially slower and more disk-intensive than 128x128.

The bundled test input is the exact RGB image `inputs/horse_256x256.png`.
Generate Experiment 3 keys from `build`:

```bash
set -o pipefail
/usr/bin/time -v ./LowMemoryFHEResNet20 \
  generate_keys 3 resolution 256 verbose 1 \
  2>&1 | tee ../logs/keygen_exp3_256.log
```

This creates `keys_exp3_256` without changing any 32x32, 64x64, or 128x128
keys. Before committing to the full run, validate both the final aggregation
and the precision-protection path:

```bash
./LowMemoryFHEResNet20 load_keys 3 resolution 256 probe_final verbose 1
./LowMemoryFHEResNet20 load_keys 3 resolution 256 probe_refresh verbose 1
```

On the development machine these passed with maximum absolute errors of about
`7.54e-10` and `6.98e-2`, respectively. Run the horse image with encrypted
model parameters using:

```bash
set -o pipefail
/usr/bin/time -v ./LowMemoryFHEResNet20 \
  load_keys 3 resolution 256 weights encrypted \
  input "inputs/horse_256x256.png" verbose 1 \
  2>&1 | tee ../logs/infer_exp3_256_encrypted_weights.log
```

#### Native 256x256 encrypted-model-parameter experiment

Encrypted parameters are the default on this branch. Every compact Conv+BN
fused diagonal, every fused bias, and each of the four final fully-connected
weight partitions is encrypted as a CKKS ciphertext before it is used. Packing
masks, fixed residual scales, refresh factors, and activation coefficients are
circuit constants rather than learned model parameters, so they remain
plaintext.

Parameter ciphertexts are created on demand and released after the associated
operation. This avoids retaining tens of thousands of large `2^17`-ring
ciphertexts simultaneously. A full 256x256 run performs approximately 47,540
ciphertext multiplicative parameter operations and 268 encrypted-bias
additions. It is therefore expected to run for several hours and to be
substantially slower than the plaintext-weight 256x256 baseline.

The existing `keys_exp3_256` can be reused; its serialized EvalMult key supports
the ciphertext-ciphertext multiplications. Do not delete or regenerate that key
directory. Run the inexpensive operation test first:

```bash
./LowMemoryFHEResNet20 load_keys 3 resolution 256 \
  test_encrypted_weights verbose 1
```

Before a complete inference, verify the precision-sensitive final and refresh
paths with encrypted fully-connected weights:

```bash
./LowMemoryFHEResNet20 load_keys 3 resolution 256 \
  weights encrypted probe_final verbose 1

./LowMemoryFHEResNet20 load_keys 3 resolution 256 \
  weights encrypted probe_refresh verbose 1
```

On the development machine, the encrypted-parameter operation self-test passed
with a maximum absolute error of `1.59e-8`. The encrypted-FC final probe passed
at `9.44e-10`, and the encrypted-FC refresh probe passed at `6.98e-2`, below its
`1.5e-1` tolerance and without a CKKS decryption failure.

For an A/B comparison, keep the image, key set, executable, and WSL settings
fixed and change only the parameter mode:

```bash
set -o pipefail
/usr/bin/time -v ./LowMemoryFHEResNet20 \
  load_keys 3 resolution 256 weights plaintext \
  input "inputs/horse_256x256.png" verbose 1 \
  2>&1 | tee ../logs/infer_exp3_256_plain_weights.log

/usr/bin/time -v ./LowMemoryFHEResNet20 \
  load_keys 3 resolution 256 weights encrypted \
  input "inputs/horse_256x256.png" verbose 1 \
  2>&1 | tee ../logs/infer_exp3_256_encrypted_weights.log
```

> [!IMPORTANT]
> This branch is a single-process performance experiment. It reads the clear
> compact `.fwgt` files and encrypts each encoded parameter immediately before
> use. The homomorphic circuit genuinely receives ciphertext parameters, but a
> complete model-confidential deployment would encrypt the model offline and
> would not make the clear weight files available to the inference process.

The 256x256 path saves these independent recovery points:

- `checkpoints/native256-stage2-v1-shard[0-7].bin` after Stage 2;
- `checkpoints/native256-layer9-pre-relu-v1-shard[0-3].bin` immediately before
  the final ReLU;
- `checkpoints/native256-stage3-unrefreshed-v1-shard[0-3].bin` after the final
  ReLU;
- `checkpoints/native256-stage3-scaled-v1-shard[0-3].bin` immediately before
  the final fully-connected layer.

Use the same recovery commands as the 128x128 path, but select resolution 256:

```bash
./LowMemoryFHEResNet20 load_keys 3 resolution 256 resume_final verbose 1
./LowMemoryFHEResNet20 load_keys 3 resolution 256 resume_refresh verbose 1
./LowMemoryFHEResNet20 load_keys 3 resolution 256 resume_relu verbose 1
./LowMemoryFHEResNet20 load_keys 3 resolution 256 resume_stage3 verbose 1
```

The 128x128 precision failure is handled from the start: the final activation
uses a degree-59 Chebyshev ReLU on `[-2, 2]`, preserves the existing positive
`0.10` factor, stores a clean pre-ReLU checkpoint, scales by another `1/16`
before a two-iteration bootstrap, and restores the combined `1/160` factor only
after client-side decryption. The 64x64 global average is scaled by `1/4096`
before its rotate-and-add reduction.

Horse is CIFAR-10 index `7`. This repository's weights were trained on 32x32
CIFAR-10 images, so a horse label at 256x256 is a useful functional check, not
a guarantee of accuracy. A scientific comparison must evaluate a labelled set
consistently at every resolution.

`resolution 256` is the default on this branch.

### Native 128x128 path

This branch adds native `128x128` and `64x64` inference paths while preserving
the original `32x32` implementation. A larger input is not resized back to
CIFAR-10 resolution. For 128x128, the ciphertext layout is:

- input: `3 x 128 x 128` in three ciphertexts, one channel per ciphertext;
- stage 1: `16 x 128 x 128` in sixteen ciphertexts;
- stage 2: `32 x 64 x 64` in eight ciphertexts;
- stage 3: `64 x 32 x 32` in four ciphertexts;
- all ciphertexts use at most 16384 logical CKKS slots, so the ring dimension
  remains `2^16`.

The included test image is `inputs/dog_128x128.png`. Dimensions are checked
strictly. Generate the independent 128x128 Experiment 3 key set from `build`:

```bash
/usr/bin/time -v ./LowMemoryFHEResNet20 \
  generate_keys 3 resolution 128 verbose 1 \
  2>&1 | tee ../logs/keygen_exp3_128.log
```

This writes `keys_exp3_128`; it does not modify `keys_exp3` or `keys_exp3_64`.
Run the bundled dog image with:

```bash
set -o pipefail
/usr/bin/time -v ./LowMemoryFHEResNet20 \
  load_keys 3 resolution 128 \
  input "inputs/dog_128x128.png" verbose 1 \
  2>&1 | tee ../logs/infer_exp3_128.log
```

The existing `keys_exp3_128` generated by an earlier version of this branch can
be reused; the precision fix does not require new evaluation keys. The current
version saves independent recovery points around every precision-sensitive
final step:

- `checkpoints/native128-stage2-v4-shard[0-7].bin` is written after Stage 2;
- `checkpoints/native128-layer9-pre-relu-v4-shard[0-3].bin` is written after
  the last bootstrap and immediately before the final ReLU;
- `checkpoints/native128-stage3-unrefreshed-v6-shard[0-3].bin` is written after
  the final ReLU and before the final normalization and bootstrap;
- `checkpoints/native128-stage3-scaled-v6-shard[0-3].bin` is written after the
  final refresh and immediately before the fully-connected layer.

If the final layer is interrupted, retry only that part instead of the
hour-long CNN stages:

```bash
set -o pipefail
/usr/bin/time -v ./LowMemoryFHEResNet20 \
  load_keys 3 resolution 128 resume_final verbose 1 \
  2>&1 | tee ../logs/infer_exp3_128_resume_final.log
```

If the normalization or final bootstrap is interrupted, resume from the
unrefreshed checkpoint and rerun the refresh plus final layer:

```bash
set -o pipefail
/usr/bin/time -v ./LowMemoryFHEResNet20 \
  load_keys 3 resolution 128 resume_refresh verbose 1 \
  2>&1 | tee ../logs/infer_exp3_128_resume_refresh.log
```

If the final ReLU needs to be retried, use its clean pre-activation checkpoint:

```bash
set -o pipefail
/usr/bin/time -v ./LowMemoryFHEResNet20 \
  load_keys 3 resolution 128 resume_relu verbose 1 \
  2>&1 | tee ../logs/infer_exp3_128_resume_relu.log
```

Stage 3 itself can be restarted from the Stage 2 checkpoint with
`resume_stage3`. The recovery commands load only the exact versions listed
above. Older checkpoints are not loaded because they may contain a high-error
or out-of-range ciphertext; they may be left on disk or removed manually.

Before committing to a complete run, the final rotations, four-ciphertext
aggregation, and decryption can be checked with a small synthetic probe:

```bash
set -o pipefail
/usr/bin/time -v ./LowMemoryFHEResNet20 \
  load_keys 3 resolution 128 probe_final verbose 1 \
  2>&1 | tee ../logs/final_probe_exp3_128.log
```

The probe intentionally does not classify the dog and its predicted label has
no semantic meaning; success means it prints ten finite output values and exits
with status `0`. `set -o pipefail` is important because otherwise `tee` can
return status `0` even when the classifier before it aborts.

For the 128x128 path, the final ReLU uses a degree-59 Chebyshev approximation on
`[-1.25, 1.25]`. The wider interval covers the observed Stage 3 activations;
using the default `[-1, 1]` interval caused unstable polynomial extrapolation
when a shard was slightly above `1.0`. It also retains the `0.10` scale already
present in the residual branch instead of multiplying the ciphertext and its
approximation error by ten. Stage 3 is then scaled by another `1/16`, explicitly
rescaled, and refreshed with OpenFHE's two-iteration bootstrap. The combined
positive factor of `1/160` is restored on the client after decryption, so it
does not change the predicted class. The global average is also scaled before
its rotate-and-add reduction.

The more expensive `probe_refresh` mode exercises the widened degree-59 final
ReLU, normalization, two-iteration bootstrap, final aggregation, and numerical
comparison together:

```bash
set -o pipefail
/usr/bin/time -v ./LowMemoryFHEResNet20 \
  load_keys 3 resolution 128 probe_refresh verbose 1 \
  2>&1 | tee ../logs/refresh_probe_exp3_128.log
```

The CIFAR-10 label for dog is index `5`. The included network was trained on
32x32 CIFAR-10 images, however, so a dog prediction at 128x128 is an experiment,
not a guaranteed correctness check. A real accuracy comparison requires a
labelled test set evaluated consistently at every resolution.

The 128x128 path remains available explicitly. It uses substantially more
ciphertexts than 64x64, so allow considerably more RAM and runtime. Key
generation is also disk-intensive.

### Native 64x64 path

The previous 64x64 layout remains available:

- input: `3 x 64 x 64`, padded to four channel blocks in one ciphertext;
- stage 1: `16 x 64 x 64` in four ciphertexts;
- stage 2: `32 x 32 x 32` in two ciphertexts;
- stage 3: `64 x 16 x 16` in one ciphertext;
- all ciphertexts use at most 16384 logical CKKS slots, so the ring dimension
  remains `2^16`.

The included test image is `inputs/cat_64x64.png`. Image dimensions are checked
strictly: the 64x64 path rejects any image that is not exactly 64x64. The model
weights are still the CIFAR-10 weights trained on 32x32 inputs, so timing and
memory comparisons are meaningful, but classification accuracy at 64x64 must
be measured separately and should not be assumed to match the paper's 32x32
result.

From the `build` directory, generate a separate 64x64 key set:

```bash
./LowMemoryFHEResNet20 generate_keys 3 resolution 64 verbose 1
```

This writes `keys_exp3_64` and never overwrites `keys_exp3`. Key generation is
disk- and memory-intensive. Make sure WSL has ample free disk space and RAM
before starting it.

Run the bundled 64x64 image:

```bash
/usr/bin/time -v ./LowMemoryFHEResNet20 load_keys 3 resolution 64 \
  input "inputs/cat_64x64.png" verbose 1
```

The upstream 32x32 path also remains available explicitly:

```bash
./LowMemoryFHEResNet20 load_keys 3 resolution 32 input "inputs/luis.png"
```

The small resolution-independent fused files in `weights/compact_fused` are
derived exactly from the upstream expanded plaintext weights. They can be
recreated and checked without PyTorch or a model download:

```bash
python3 tools/export_compact_fused_weights.py
python3 tools/validate_compact_fused_weights.py
```

> [!IMPORTANT]
> With newer versions of OpenFHE, a `DropLastElement: Removing last element of DCRTPoly renders it invalid.` error may pop up. This happens because of an additional multiplication performed by the bootstrapping. I suggest you two solutions:
> - Relax the security parameters to (NotSet) and increase the circuit depth by 1. This will not give you a 128-bits secure circuit anymore, something less.
> - Increase the ring dimension to $2^{17}$ and increase the circuit depth by 1. This will still give you $>128$-bits of security, but time and memory will be doubled.

### Prerequisites
Linux or Mac operative system, with at least 16GB of RAM.

In order to run the program, you need to install:
- `cmake`
- `g++` or `clang`
- `OpenFHE` ([how to install OpenFHE](https://openfhe-development.readthedocs.io/en/latest/sphinx_rsts/intro/installation/installation.html)), this work has been tested on v1.0.4

### 1) Build the project

Setup the project using this command:
```
mkdir build
cmake -B "build" -S LowMemoryFHEResNet20
```
Then build it using
```
cmake --build "build" --target
```

### 2) Execute the project

After building, go to the created `build` folder:

```
cd build
```
and run it with the following command:
```
./LowMemoryFHEResNet20
```

### 3) Custom arguments

- `generate_keys`, type `int`, a value in `[1, 2, 3, 4]`
- `load_keys`, type: `int` a value in `[1, 2, 3, 4]`
- `input`, type: `string`, the filename of a custom image. It **MUST** match the selected resolution exactly (`32x32`, `64x64`, `128x128`, or `256x256`) and may be `.jpg` or `.png`; inputs are decoded as RGB
- `verbose` a value in `[-1, 0, 1, 2]`, the first shows no information, the last shows a lot of messages
- `plain`: added when the user wants the plain result too. This comparison is currently available only for the original 32x32 path. It requires `torch`, `torchvision`, `PIL`, and `numpy`.
- `resolution`, type: `int`, either `32`, `64`, `128`, or `256`; this branch defaults to `256`
- `probe_final`: with `resolution 128` or `256`, run a synthetic final-layer check without the three CNN stages
- `resume_final`: with `resolution 128` or `256`, load the four saved Stage 3 ciphertexts and rerun only the final layer
- `probe_refresh`: with `resolution 128` or `256`, stress-test the normalized Stage 3 bootstrap and final layer
- `resume_refresh`: with `resolution 128` or `256`, load the unrefreshed Stage 3 checkpoint and rerun its refresh plus the final layer
- `resume_relu`: with `resolution 128` or `256`, load the final pre-ReLU checkpoint and rerun the final ReLU, refresh, and final layer
- `resume_stage3`: with `resolution 128` or `256`, load the Stage 2 checkpoint and rerun Stage 3 plus the final layer
- `weights`, followed by `encrypted` or `plaintext`; this branch defaults to encrypted model parameters for native high-resolution inference
- `test_encrypted_weights`: run a small ciphertext-weight multiplication and encrypted-bias addition test instead of full inference; use it with `load_keys`

#### Some examples 

The first execution should be launched with the `generate_keys` argument, using the preferred set of parameters. Check the paper to see the differences between them. The original 32x32 example is selected explicitly below:
```
./LowMemoryFHEResNet20 generate_keys 1 resolution 32
```
This command create the required keys and stores them in a new folder called `keys_exp1`, in the root folder of the project.

The 32x32 path classifies the default image in `inputs/luis.png`. We can, however, use custom arguments.
We can use a set of serialized context and keys with the argument `load_keys` as follows:

```
./LowMemoryFHEResNet20 load_keys 1 resolution 32
```
This command loads context and keys from the folder `keys_exp1`, located in the root folder of the project, and runs an inference on the default image.
Then, in order to load a custom image, we use the argument `input` as follows:

```
./LowMemoryFHEResNet20 load_keys 1 resolution 32 input "inputs/vale.jpg"
```
Even for this argument, the starting position will be the root of the project.
We can also compare the result with the plain version of the model, using the `plain` keyword:

```
./LowMemoryFHEResNet20 load_keys 1 resolution 32 input "inputs/vale.jpg" plain
```

This command will launch a Python script at the end of the encrypted comptations, giving the plain output (which will differ from the encrypted one according to the parameters, check the paper for the precision values of each set of parameters).
Notice that `plain` requires a few things in order to be used:

- `python3`
- `torch`
- `torchvision`
- `PIL`
- `numpy`

## Interpreting the output
The output of the encrypted model is a vector consisting of 10 elements. In order to interpret it, it is enough to find the index of the maximum element. A sample output could be:

```
output = [-2.633, -1.091,  6.063, -4.093, -0.5967, 7.252, -2.156, -1.085, -0.9119, -0.7291]
```
In this case, the maximum value is at position 5. Just translate it using the following dictionary (from ResNet20 pretrained on CIFAR-10):

| Index of max 	| Class      	|
|--------------	|------------	|
| 0            	| Airplane   	|
| 1            	| Automobile 	|
| 2            	| Bird       	|
| 3            	| Cat        	|
| 4            	| Deer       	|
| 5            	| Dog        	|
| 6            	| Frog       	|
| 7            	| Horse      	|
| 8            	| Ship       	|
| 9            	| Truck      	|

In the sample output, the input image was my dog Vale:

> <img src="inputs/vale.jpg" alt="ResNet dog input image" width=20%>

---

Another output could be

```
output = [-0.719, -4.19, -0.252, 12.04, -4.979, 4.413, -0.5173, -1.038, -2.229, -2.504]
```

In this case, the index of max is 3, which is nice, since the input image was Luis the cat:

> <img src="inputs/luis.png" alt="ResNet cat input image" width=20%>

So it was correct!

## Comparing to the plain model

In the `notebook` folder, it is possible to find different useful notebooks that can be used in order to compute the precision of a computation, with respect to the plain model, in details for each layer. 

## Citing
In case you want to cite our work, feel free to do it using the following BibTeX entry:

```
@article{rovidaleporati,
   author = {Rovida, Lorenzo and Leporati, Alberto},
   title = {Encrypted Image Classification with Low Memory Footprint Using Fully Homomorphic Encryption},
   journal = {International Journal of Neural Systems},
   volume = {34},
   number = {05},
   pages = {2450025},
   year = {2024},
   doi = {10.1142/S0129065724500254}
}
```

## Authors

- Lorenzo Rovida (`lorenzo.rovida@unimib.it`)
- Alberto Leporati (`alberto.leporati@unimib.it`)

Made with <3  at [Bicocca Security Lab](https://www.bislab.unimib.it), at University of Milan-Bicocca.

<img src="imgs/lab_logo.png" alt="BisLab logo" width=20%>


### Declaration

This is a proof of concept and, even though parameters are created with $\lambda \geq 128$ security bits (according to [Homomorphic Encryption Standards](https://homomorphicencryption.org/standard)), this circuit is intended for educational purposes only.


## Bibliography

[1] Kim, A., Papadimitriou, A., & Polyakov, Y. (2022). 
Approximate Homomorphic Encryption with Reduced Approximation Error. In: Galbraith, S.D. (eds) Topics in Cryptology – CT-RSA 2022. CT-RSA 2022. Lecture Notes in Computer Science, vol 13161. Springer, Cham.

[2] Al Badawi, A., Bates, J., Bergamaschi, F., Cousins, D. B., Erabelli, S., Genise, N., Halevi, S., Hunt, H., Kim, A., Lee, Y., Liu, Z., Micciancio, D., Quah, I., Polyakov, Y., R.V., S., Rohloff, K., Saylor, J., Suponitsky, D., Triplett, M., Zucca, V. (2022). *OpenFHE: Open-Source Fully Homomorphic Encryption Library*. Proceedings of the 10th Workshop on Encrypted Computing & Applied Homomorphic Cryptography, 53–63.

[3] Lyubashevsky, V., Peikert, C., & Regev, O. (2010). *On Ideal Lattices and Learning with Errors over Rings*. In: Gilbert, H. (eds) Advances in Cryptology – EUROCRYPT 2010. EUROCRYPT 2010. Lecture Notes in Computer Science, vol 6110. Springer, Berlin, Heidelberg.

[4] Kim, D., & Guyot, C. (2023). *Optimized Privacy-Preserving CNN Inference With Fully Homomorphic Encryption*. In IEEE Transactions on Information Forensics and Security, vol. 18, pp. 2175-2187.

[5] Lee, E., Lee, J. W., Lee, J., Kim, Y. S., Kim, Y., No, J. S., & Choi, W. (2022, June). *Low-complexity deep convolutional neural networks on fully homomorphic encryption using multiplexed parallel convolutions*. In International Conference on Machine Learning (pp. 12403-12422). PMLR.

[6] De Castro, L., Agrawal, R., Yazicigil, R., Chandrakasan, A., Vaikuntanathan, V., Juvekar, C., & Joshi, A. (2021). Does Fully Homomorphic Encryption Need Compute Acceleration?
