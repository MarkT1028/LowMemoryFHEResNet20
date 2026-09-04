#include <iostream>
#include <cmath>
#include <iomanip>
#include <sys/stat.h>

#include "FHEController.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define GREEN_TEXT "\033[1;32m"
#define RED_TEXT "\033[1;31m"
#define RESET_COLOR "\033[0m"


void check_arguments(int argc, char *argv[]);
vector<double> read_image(const char *filename, int expected_size = 32);

void executeResNet20();
void executeResNet64();
void executeResNet128();
void executeResNet256();
void executeResNet128FinalFromCheckpoint();
void executeResNet128RefreshFromCheckpoint();
void executeResNet128ReluFromCheckpoint();
void executeResNet128Stage3FromCheckpoint();
void executeResNet256FinalFromCheckpoint();
void executeResNet256RefreshFromCheckpoint();
void executeResNet256ReluFromCheckpoint();
void executeResNet256Stage3FromCheckpoint();
void executeFinalProbeHighResolution(int resolution);
void executeRefreshProbeHighResolution(int resolution);
void generate_evaluation_keys64();
void generate_evaluation_keys128();
void generate_evaluation_keys256();
void save_tensor_checkpoint(const EncryptedTensor& tensor, const string& prefix);
EncryptedTensor load_tensor_checkpoint(const TensorLayout& layout, const string& prefix);
void print_ciphertext_stats(const Ctxt& ciphertext, const string& label);

Ctxt initial_layer(const Ctxt& in);
Ctxt layer1(const Ctxt& in);
Ctxt layer2(const Ctxt& in);
Ctxt layer3(const Ctxt& in);
Ctxt final_layer(const Ctxt& in);

EncryptedTensor residual_block_native(const EncryptedTensor& in,
                                      const string& weight_prefix,
                                      double first_scale,
                                      double second_scale,
                                      const string& title);
EncryptedTensor layer1_native(const EncryptedTensor& in);
EncryptedTensor layer2_native(const EncryptedTensor& in,
                              int output_channels_per_ciphertext);
EncryptedTensor layer3_native(const EncryptedTensor& in,
                              int output_channels_per_ciphertext);
EncryptedTensor layer2_high_resolution(EncryptedTensor in,
                                       int original_resolution);
EncryptedTensor layer3_128(EncryptedTensor in);
EncryptedTensor layer3_256(EncryptedTensor in);
EncryptedTensor final_residual_block_128(const EncryptedTensor& in,
                                         bool timing);
EncryptedTensor final_residual_block_256(const EncryptedTensor& in,
                                         bool timing);
EncryptedTensor refresh_stage3_high_resolution(EncryptedTensor in,
                                                bool timing);
Ctxt final_layer64(const EncryptedTensor& in);
Ctxt final_layer_high_resolution(const EncryptedTensor& in,
                                 int expected_width,
                                 double decrypted_output_scale = 1.0);

FHEController controller;

constexpr double HIGHRES_STAGE3_REFRESH_SCALE = 1.0 / 16.0;
constexpr double HIGHRES_DECRYPTED_OUTPUT_SCALE = 160.0;
constexpr int HIGHRES_FINAL_RELU_DEGREE = 59;
constexpr double NATIVE128_FINAL_RELU_BOUND = 1.25;
constexpr int NATIVE256_SLOTS = 1 << 16;
constexpr double NATIVE256_FINAL_RELU_BOUND = 2.0;

int generate_context;
string input_filename;
int verbose;
bool test;
bool plain;
bool resume_final128;
bool resume_refresh128;
bool resume_relu128;
bool resume_stage3_128;
bool probe_final128;
bool probe_refresh128;
int input_resolution;
bool test_encrypted_weights;

/*
 * TODO:
 * 1) Migliorare convbn sfruttando tutti gli slot del ciphertext
 */

int main(int argc, char *argv[]) {
    //TODO: possibile che il bootstrap a 8192 ci metta lo stesso tempo? indaga

    check_arguments(argc, argv);

    if (test) {
        controller.test_context();
        exit(0);
    }

    if (generate_context == -1) {
        cerr << "You either have to use the argument \"generate_keys\" or \"load_keys\"!\nIf it is your first time, you could try "
                "with \"./LowMemoryFHEResNet20 generate_keys 1\"\nCheck the README.md.\nAborting. :-(" << endl;
        exit(1);
    }


    if (generate_context > 0) {
        const int context_log_ring = input_resolution == 256 ? 17 : 16;
        const int context_batch_slots = input_resolution == 256
            ? NATIVE256_SLOTS
            : (1 << 14);
        switch (generate_context) {
            case 1:
                controller.generate_context(
                    context_log_ring, 52, 48, 2, 3, 3, 59, true,
                    context_batch_slots);
                break;
            case 2:
                controller.generate_context(
                    context_log_ring, 50, 46, 3, 4, 4, 200, true,
                    context_batch_slots);
                break;
            case 3:
                controller.generate_context(
                    context_log_ring, 50, 46, 3, 5, 4, 119, true,
                    context_batch_slots);
                break;
            case 4:
                controller.generate_context(
                    context_log_ring, 48, 44, 2, 4, 4, 59, true,
                    context_batch_slots);
                break;
            default:
                controller.generate_context(true);
                break;
        }

        if (verbose > 1) cout << "Basic context built. Now generating bootstrapping and rotations keys..." << endl;

        if (verbose > 1) cout << "(It may take a while, depending on the machine)" << endl;


        if (input_resolution == 256) {
            generate_evaluation_keys256();
            cout << "256x256 context created correctly." << endl;
            exit(0);
        }
        if (input_resolution == 64) {
            generate_evaluation_keys64();
            cout << "64x64 context created correctly." << endl;
            exit(0);
        }
        if (input_resolution == 128) {
            generate_evaluation_keys128();
            cout << "128x128 context created correctly." << endl;
            exit(0);
        }

        controller.generate_bootstrapping_and_rotation_keys({1, -1, 32, -32, -1024},
                                                            16384,
                                                            true,
                                                            "rotations-layer1.bin");
        //After each serialization I release and re-load the context, otherwise OpenFHE gives a weird error (something
        //like "4kb missing"), but I have no time to investigate :D
        if (verbose > 1) cout << "1/6 done." << endl;
        controller.clear_context(16384);
        controller.load_context(false);
        controller.generate_rotation_keys({1, 2, 4, 8, 64-16, -(1024 - 256), (1024 - 256) * 32, -8192},
                                          true,
                                          "rotations-layer2-downsample.bin");
        if (verbose > 1) cout << "2/6 done." << endl;
        controller.clear_context(0);
        controller.load_context(false);
        controller.generate_bootstrapping_and_rotation_keys({1, -1, 16, -16, -256},
                                          8192,
                                          true,
                                          "rotations-layer2.bin");
        if (verbose > 1) cout << "3/6 done." << endl;
        controller.clear_context(8192);
        controller.load_context(false);
        controller.generate_rotation_keys({1, 2, 4, 32 - 8, -(256 - 64), (256 - 64) * 64, -4096},
                                          true,
                                          "rotations-layer3-downsample.bin");
        if (verbose > 1) cout << "4/6 done." << endl;
        controller.clear_context(0);
        controller.load_context(false);
        controller.generate_bootstrapping_and_rotation_keys({1, -1, 8, -8, -64},
                                          4096,
                                          true,
                                          "rotations-layer3.bin");
        if (verbose > 1)cout << "5/6 done." << endl;
        controller.clear_context(4096);
        controller.load_context(false);
        controller.generate_rotation_keys({1, 2, 4, 8, 16, 32, -15, 64, 128, 256, 512, 1024, 2048}, true, "rotations-finallayer.bin");
        if (verbose > 1) cout << "6/6 done!" << endl;

        controller.clear_context(0);
        controller.load_context(false);

        cout << "Context created correctly." << endl;
        exit(0);

    } else {
        controller.load_context(verbose > 1);
    }

    if (test_encrypted_weights) {
        controller.num_slots = input_resolution == 256
            ? NATIVE256_SLOTS
            : (1 << 14);
        exit(controller.test_encrypted_model_parameter_ops() ? 0 : 1);
    }

    if ((input_resolution == 128 || input_resolution == 256) &&
        probe_refresh128) {
        executeRefreshProbeHighResolution(input_resolution);
    } else if ((input_resolution == 128 || input_resolution == 256) &&
               probe_final128) {
        executeFinalProbeHighResolution(input_resolution);
    } else if (input_resolution == 128 && resume_refresh128) {
        executeResNet128RefreshFromCheckpoint();
    } else if (input_resolution == 128 && resume_relu128) {
        executeResNet128ReluFromCheckpoint();
    } else if (input_resolution == 128 && resume_stage3_128) {
        executeResNet128Stage3FromCheckpoint();
    } else if (input_resolution == 128 && resume_final128) {
        executeResNet128FinalFromCheckpoint();
    } else if (input_resolution == 256 && resume_refresh128) {
        executeResNet256RefreshFromCheckpoint();
    } else if (input_resolution == 256 && resume_relu128) {
        executeResNet256ReluFromCheckpoint();
    } else if (input_resolution == 256 && resume_stage3_128) {
        executeResNet256Stage3FromCheckpoint();
    } else if (input_resolution == 256 && resume_final128) {
        executeResNet256FinalFromCheckpoint();
    } else if (input_resolution == 256) {
        executeResNet256();
    } else if (input_resolution == 128) {
        executeResNet128();
    } else if (input_resolution == 64) {
        executeResNet64();
    } else {
        executeResNet20();
    }

    if (verbose >= 0 && input_resolution != 32) {
        controller.print_model_parameter_stats();
    }
}

void generate_evaluation_keys64() {
    // Native 64x64 keeps a 2^16 ring and shards every intermediate tensor into
    // ciphertexts of exactly 16384 logical CKKS slots.
    controller.generate_bootstrapping_and_rotation_keys(
        {1, -1, 64, -64, -4096},
        16384,
        true,
        "rotations-layer1.bin");
    if (verbose > 1) cout << "1/6 done." << endl;

    controller.clear_context(16384);
    controller.load_context(false);
    controller.generate_rotation_keys(
        {1, 2, 4, 8, 16, 96, -3072, 12288, -4096},
        true,
        "rotations-layer2-downsample.bin");
    if (verbose > 1) cout << "2/6 done." << endl;

    controller.clear_context(0);
    controller.load_context(false);
    controller.generate_bootstrapping_and_rotation_keys(
        {1, -1, 32, -32, -1024},
        16384,
        true,
        "rotations-layer2.bin");
    if (verbose > 1) cout << "3/6 done." << endl;

    controller.clear_context(16384);
    controller.load_context(false);
    controller.generate_rotation_keys(
        {1, 2, 4, 8, 48, -768, 12288, -4096},
        true,
        "rotations-layer3-downsample.bin");
    if (verbose > 1) cout << "4/6 done." << endl;

    controller.clear_context(0);
    controller.load_context(false);
    controller.generate_bootstrapping_and_rotation_keys(
        {1, -1, 16, -16, -256},
        16384,
        true,
        "rotations-layer3.bin");
    if (verbose > 1) cout << "5/6 done." << endl;

    controller.clear_context(16384);
    controller.load_context(false);
    controller.generate_rotation_keys(
        {1, 2, 4, 8, 16, 32, 64, 128, -15,
         256, 512, 1024, 2048, 4096, 8192},
        true,
        "rotations-finallayer.bin");
    if (verbose > 1) cout << "6/6 done!" << endl;

    controller.clear_context(0);
    controller.load_context(false);
}

void generate_evaluation_keys128() {
    // The 128x128 path still caps every ciphertext at 16384 slots. The first
    // two key files include their following stride-2 packing rotations so the
    // large transition tensors can be compacted without keeping two branches
    // alive or reloading another multi-gigabyte bootstrapping key set.
    controller.generate_bootstrapping_and_rotation_keys(
        {1, -1, 2, 4, 8, 16, 32, 128, -128, 192, -4096},
        16384,
        true,
        "rotations-layer1.bin");
    if (verbose > 1) cout << "1/4 done." << endl;

    controller.clear_context(16384);
    controller.load_context(false);
    controller.generate_bootstrapping_and_rotation_keys(
        {1, -1, 2, 4, 8, 16, 64, -64, 96,
         -4096, -3072, 12288},
        16384,
        true,
        "rotations-layer2.bin");
    if (verbose > 1) cout << "2/4 done." << endl;

    controller.clear_context(16384);
    controller.load_context(false);
    controller.generate_bootstrapping_and_rotation_keys(
        {1, -1, 32, -32, -1024},
        16384,
        true,
        "rotations-layer3.bin");
    if (verbose > 1) cout << "3/4 done." << endl;

    controller.clear_context(16384);
    controller.load_context(false);
    controller.generate_rotation_keys(
        {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, -15,
         1024, 2048, 4096, 8192},
        true,
        "rotations-finallayer.bin");
    if (verbose > 1) cout << "4/4 done!" << endl;

    controller.clear_context(0);
    controller.load_context(false);
}

void generate_evaluation_keys256() {
    // A 256x256 channel occupies 65536 CKKS slots, so this path uses the full
    // slot capacity of a 2^17 ring. Each key file also includes the rotations
    // needed by the following stride-2 compaction, which avoids retaining two
    // very large bootstrapping key sets at once.
    controller.num_slots = NATIVE256_SLOTS;
    controller.generate_bootstrapping_and_rotation_keys(
        {1, -1, 2, 4, 8, 16, 32, 64, 128, -128,
         256, -256, 384, -16384},
        NATIVE256_SLOTS,
        true,
        "rotations-layer1.bin");
    if (verbose > 1) cout << "1/4 done." << endl;

    controller.clear_context(NATIVE256_SLOTS);
    controller.load_context(false);
    controller.num_slots = NATIVE256_SLOTS;
    controller.generate_bootstrapping_and_rotation_keys(
        {1, -1, 2, 4, 8, 16, 32, 128, -128, 192,
         -12288, -16384, 49152},
        NATIVE256_SLOTS,
        true,
        "rotations-layer2.bin");
    if (verbose > 1) cout << "2/4 done." << endl;

    controller.clear_context(NATIVE256_SLOTS);
    controller.load_context(false);
    controller.num_slots = NATIVE256_SLOTS;
    controller.generate_bootstrapping_and_rotation_keys(
        {1, -1, 64, -64, -4096},
        NATIVE256_SLOTS,
        true,
        "rotations-layer3.bin");
    if (verbose > 1) cout << "3/4 done." << endl;

    controller.clear_context(NATIVE256_SLOTS);
    controller.load_context(false);
    controller.num_slots = NATIVE256_SLOTS;
    controller.generate_rotation_keys(
        {1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
         1024, 2048, 4096, 8192, 16384, 32768, -15},
        true,
        "rotations-finallayer.bin");
    if (verbose > 1) cout << "4/4 done!" << endl;

    controller.clear_context(0);
    controller.load_context(false);
}

void executeResNet64() {
    if (verbose >= 0) {
        cout << "Encrypted ResNet20 native 64x64 classification started." << endl;
        cout << "Packing: 1 -> 4 -> 2 -> 1 ciphertexts; 16384 slots per ciphertext." << endl;
    }

    if (input_filename.empty()) {
        input_filename = "../inputs/cat_64x64.png";
        if (verbose >= 0) {
            cout << "You did not set any input, I use " << GREEN_TEXT
                 << input_filename << RESET_COLOR << "." << endl;
        }
    } else if (verbose >= 0) {
        cout << "I am going to encrypt and classify " << GREEN_TEXT
             << input_filename << RESET_COLOR << "." << endl;
    }

    vector<double> input_image = read_image(input_filename.c_str(), 64);
    controller.num_slots = 16384;
    TensorLayout input_layout{64, 3, 4, 16384};
    EncryptedTensor current = controller.encrypt_tensor(
        input_image,
        input_layout,
        controller.circuit_depth - 4 - get_relu_depth(controller.relu_degree));

    bool timing = verbose > 1;
    controller.load_bootstrapping_and_rotation_keys(
        "rotations-layer1.bin", 16384, timing);

    auto start = start_time();
    current = controller.convbn_sharded(
        current, "../weights/compact_fused/initial.fwgt", 16, 0.90, false, timing);
    current = controller.relu_tensor(current, 0.90, timing);

    auto start_layer = start_time();
    current = layer1_native(current);
    if (verbose > 0) print_duration(start_layer, "64x64 stage 1 took:");

    start_layer = start_time();
    current = layer2_native(current, 16);
    if (verbose > 0) print_duration(start_layer, "64x64 stage 2 took:");

    start_layer = start_time();
    current = layer3_native(current, 64);
    if (verbose > 0) print_duration(start_layer, "64x64 stage 3 took:");

    final_layer64(current);
    if (verbose > 0) {
        print_duration_yellow(start, "The native 64x64 circuit evaluation took: ");
    }
}

void executeResNet128() {
    if (verbose >= 0) {
        cout << "Encrypted ResNet20 native 128x128 classification started." << endl;
        cout << "Packing: 3 -> 16 -> 8 -> 4 ciphertexts; 16384 slots per ciphertext." << endl;
    }

    if (input_filename.empty()) {
        input_filename = "../inputs/dog_128x128.png";
        if (verbose >= 0) {
            cout << "You did not set any input, I use " << GREEN_TEXT
                 << input_filename << RESET_COLOR << "." << endl;
        }
    } else if (verbose >= 0) {
        cout << "I am going to encrypt and classify " << GREEN_TEXT
             << input_filename << RESET_COLOR << "." << endl;
    }

    vector<double> input_image = read_image(input_filename.c_str(), 128);
    controller.num_slots = 16384;
    TensorLayout input_layout{128, 3, 1, 16384};
    EncryptedTensor current = controller.encrypt_tensor(
        input_image,
        input_layout,
        controller.circuit_depth - 4 - get_relu_depth(controller.relu_degree));

    bool timing = verbose > 1;
    controller.load_bootstrapping_and_rotation_keys(
        "rotations-layer1.bin", 16384, timing);

    auto start = start_time();
    current = controller.convbn_sharded(
        current, "../weights/compact_fused/initial.fwgt", 16, 0.90, false, timing);
    current = controller.relu_tensor(current, 0.90, timing);

    auto start_layer = start_time();
    current = layer1_native(current);
    if (verbose > 0) print_duration(start_layer, "128x128 stage 1 took:");

    start_layer = start_time();
    current = layer2_high_resolution(std::move(current), 128);
    if (verbose > 0) print_duration(start_layer, "128x128 stage 2 took:");
    save_tensor_checkpoint(current, "native128-stage2-v4");

    start_layer = start_time();
    current = layer3_128(std::move(current));
    if (verbose > 0) print_duration(start_layer, "128x128 stage 3 took:");

    save_tensor_checkpoint(current, "native128-stage3-scaled-v6");
    final_layer_high_resolution(
        current, 32, HIGHRES_DECRYPTED_OUTPUT_SCALE);
    if (verbose > 0) {
        print_duration_yellow(start, "The native 128x128 circuit evaluation took: ");
    }
}

void executeResNet256() {
    if (verbose >= 0) {
        cout << "Encrypted ResNet20 native 256x256 classification started."
             << endl;
        cout << "Packing: 3 -> 16 -> 8 -> 4 ciphertexts; 65536 slots per "
                "ciphertext; ring dimension 2^17."
             << endl;
        cout << "This path needs substantially more RAM than 128x128; keep "
                "WSL swap enabled."
             << endl;
        cout << "Model parameters are "
             << (controller.model_parameters_are_encrypted()
                 ? "encrypted CKKS ciphertexts."
                 : "CKKS plaintexts.")
             << endl;
    }

    if (input_filename.empty()) {
        input_filename = "../inputs/horse_256x256.png";
        if (verbose >= 0) {
            cout << "You did not set any input, I use " << GREEN_TEXT
                 << input_filename << RESET_COLOR << "." << endl;
        }
    } else if (verbose >= 0) {
        cout << "I am going to encrypt and classify " << GREEN_TEXT
             << input_filename << RESET_COLOR << "." << endl;
    }

    vector<double> input_image = read_image(input_filename.c_str(), 256);
    controller.num_slots = NATIVE256_SLOTS;
    TensorLayout input_layout{256, 3, 1, NATIVE256_SLOTS};
    EncryptedTensor current = controller.encrypt_tensor(
        input_image,
        input_layout,
        controller.circuit_depth - 4 - get_relu_depth(controller.relu_degree));

    bool timing = verbose > 1;
    controller.load_bootstrapping_and_rotation_keys(
        "rotations-layer1.bin", NATIVE256_SLOTS, timing);

    auto start = start_time();
    current = controller.convbn_sharded(
        current, "../weights/compact_fused/initial.fwgt", 16, 0.90, false,
        timing);
    current = controller.relu_tensor(current, 0.90, timing);

    auto start_layer = start_time();
    current = layer1_native(current);
    if (verbose > 0) print_duration(start_layer, "256x256 stage 1 took:");

    start_layer = start_time();
    current = layer2_high_resolution(std::move(current), 256);
    if (verbose > 0) print_duration(start_layer, "256x256 stage 2 took:");
    save_tensor_checkpoint(current, "native256-stage2-v1");

    start_layer = start_time();
    current = layer3_256(std::move(current));
    if (verbose > 0) print_duration(start_layer, "256x256 stage 3 took:");

    save_tensor_checkpoint(current, "native256-stage3-scaled-v1");
    final_layer_high_resolution(
        current, 64, HIGHRES_DECRYPTED_OUTPUT_SCALE);
    if (verbose > 0) {
        print_duration_yellow(
            start, "The native 256x256 circuit evaluation took: ");
    }
}

void save_tensor_checkpoint(const EncryptedTensor& tensor,
                            const string& prefix) {
    struct stat checkpoint_dir;
    if (stat("../checkpoints", &checkpoint_dir) != 0) {
        if (mkdir("../checkpoints", 0777) != 0) {
            throw runtime_error("Could not create ../checkpoints");
        }
    } else if (!S_ISDIR(checkpoint_dir.st_mode)) {
        throw runtime_error("../checkpoints exists but is not a directory");
    }

    for (size_t shard = 0; shard < tensor.shards.size(); shard++) {
        string filename = "../checkpoints/" + prefix + "-shard" +
                          to_string(shard) + ".bin";
        if (!Serial::SerializeToFile(
                filename, tensor.shards[shard], SerType::BINARY)) {
            throw runtime_error("Could not write checkpoint: " + filename);
        }
    }

    if (verbose >= 0) {
        cout << "Saved checkpoint set '" << prefix << "' with "
             << tensor.shards.size() << " ciphertexts in ../checkpoints/."
             << endl;
    }
}

EncryptedTensor load_tensor_checkpoint(const TensorLayout& layout,
                                       const string& prefix) {
    EncryptedTensor tensor;
    tensor.layout = layout;
    tensor.shards.resize(layout.ciphertext_count());

    for (size_t shard = 0; shard < tensor.shards.size(); shard++) {
        string filename = "../checkpoints/" + prefix + "-shard" +
                          to_string(shard) + ".bin";
        if (!Serial::DeserializeFromFile(
                filename, tensor.shards[shard], SerType::BINARY)) {
            throw runtime_error(
                "Could not load checkpoint: " + filename +
                ". Run a complete inference for the selected resolution first.");
        }
    }

    if (verbose >= 0) {
        cout << "Loaded checkpoint set '" << prefix << "' with "
             << tensor.shards.size() << " ciphertexts from ../checkpoints/."
             << endl;
    }
    return tensor;
}

void print_ciphertext_stats(const Ctxt& ciphertext, const string& label) {
    if (verbose < 2) {
        return;
    }

    vector<double> values = controller.decrypt_tovector(
        ciphertext, controller.num_slots);
    double minimum = values[0];
    double maximum = values[0];
    double maximum_absolute = abs(values[0]);
    size_t non_finite = 0;
    for (double value : values) {
        minimum = min(minimum, value);
        maximum = max(maximum, value);
        maximum_absolute = max(maximum_absolute, abs(value));
        if (!isfinite(value)) {
            non_finite++;
        }
    }
    cout << label << ": level=" << ciphertext->GetLevel()
         << ", min=" << scientific << setprecision(6) << minimum
         << ", max=" << maximum
         << ", max_abs=" << maximum_absolute
         << ", non_finite=" << non_finite
         << defaultfloat << endl;
}

void executeResNet128FinalFromCheckpoint() {
    if (verbose >= 0) {
        cout << "Resuming the native 128x128 final layer from checkpoint." << endl;
    }

    controller.num_slots = 16384;
    TensorLayout stage3_layout{32, 64, 16, 16384};
    EncryptedTensor current = load_tensor_checkpoint(
        stage3_layout, "native128-stage3-scaled-v6");
    for (size_t shard = 0; shard < current.shards.size(); shard++) {
        print_ciphertext_stats(
            current.shards[shard],
            "Checkpoint shard " + to_string(shard));
    }

    auto start = start_time();
    final_layer_high_resolution(
        current, 32, HIGHRES_DECRYPTED_OUTPUT_SCALE);
    if (verbose > 0) {
        print_duration_yellow(start, "The resumed 128x128 final layer took: ");
    }
}

void executeResNet128RefreshFromCheckpoint() {
    if (verbose >= 0) {
        cout << "Resuming the native 128x128 Stage 3 refresh from checkpoint."
             << endl;
    }

    controller.num_slots = 16384;
    TensorLayout stage3_layout{32, 64, 16, 16384};
    EncryptedTensor current = load_tensor_checkpoint(
        stage3_layout, "native128-stage3-unrefreshed-v6");
    for (size_t shard = 0; shard < current.shards.size(); shard++) {
        print_ciphertext_stats(
            current.shards[shard],
            "Unrefreshed checkpoint shard " + to_string(shard));
    }
    controller.load_bootstrapping_and_rotation_keys(
        "rotations-layer3.bin", 16384, verbose > 1);

    auto start = start_time();
    current = refresh_stage3_high_resolution(
        std::move(current), verbose > 1);
    save_tensor_checkpoint(current, "native128-stage3-scaled-v6");
    final_layer_high_resolution(
        current, 32, HIGHRES_DECRYPTED_OUTPUT_SCALE);
    if (verbose > 0) {
        print_duration_yellow(
            start, "The resumed 128x128 refresh and final layer took: ");
    }
}

void executeResNet128ReluFromCheckpoint() {
    if (verbose >= 0) {
        cout << "Resuming the native 128x128 final ReLU from checkpoint."
             << endl;
    }

    controller.num_slots = 16384;
    TensorLayout stage3_layout{32, 64, 16, 16384};
    EncryptedTensor current = load_tensor_checkpoint(
        stage3_layout, "native128-layer9-pre-relu-v4");
    for (size_t shard = 0; shard < current.shards.size(); shard++) {
        print_ciphertext_stats(
            current.shards[shard],
            "Pre-ReLU checkpoint shard " + to_string(shard));
    }
    auto start = start_time();
    current = controller.relu_tensor_wide(
        current,
        -NATIVE128_FINAL_RELU_BOUND,
        NATIVE128_FINAL_RELU_BOUND,
        HIGHRES_FINAL_RELU_DEGREE,
        1.0,
        verbose > 1);
    for (size_t shard = 0; shard < current.shards.size(); shard++) {
        print_ciphertext_stats(
            current.shards[shard],
            "Post-ReLU checkpoint shard " + to_string(shard));
    }
    save_tensor_checkpoint(current, "native128-stage3-unrefreshed-v6");
    controller.load_bootstrapping_and_rotation_keys(
        "rotations-layer3.bin", 16384, verbose > 1);
    current = refresh_stage3_high_resolution(
        std::move(current), verbose > 1);
    save_tensor_checkpoint(current, "native128-stage3-scaled-v6");
    final_layer_high_resolution(
        current, 32, HIGHRES_DECRYPTED_OUTPUT_SCALE);
    if (verbose > 0) {
        print_duration_yellow(
            start, "The resumed final ReLU, refresh, and final layer took: ");
    }
}

void executeResNet128Stage3FromCheckpoint() {
    if (verbose >= 0) {
        cout << "Resuming the native 128x128 Stage 3 from checkpoint." << endl;
    }

    controller.num_slots = 16384;
    TensorLayout stage2_layout{64, 32, 4, 16384};
    EncryptedTensor current = load_tensor_checkpoint(
        stage2_layout, "native128-stage2-v4");
    controller.load_bootstrapping_and_rotation_keys(
        "rotations-layer2.bin", 16384, verbose > 1);

    auto start = start_time();
    current = layer3_128(std::move(current));
    save_tensor_checkpoint(current, "native128-stage3-scaled-v6");
    final_layer_high_resolution(
        current, 32, HIGHRES_DECRYPTED_OUTPUT_SCALE);
    if (verbose > 0) {
        print_duration_yellow(
            start, "The resumed 128x128 Stage 3 and final layer took: ");
    }
}

void executeResNet256FinalFromCheckpoint() {
    if (verbose >= 0) {
        cout << "Resuming the native 256x256 final layer from checkpoint."
             << endl;
    }

    controller.num_slots = NATIVE256_SLOTS;
    TensorLayout stage3_layout{64, 64, 16, NATIVE256_SLOTS};
    EncryptedTensor current = load_tensor_checkpoint(
        stage3_layout, "native256-stage3-scaled-v1");
    for (size_t shard = 0; shard < current.shards.size(); shard++) {
        print_ciphertext_stats(
            current.shards[shard],
            "Checkpoint shard " + to_string(shard));
    }

    auto start = start_time();
    final_layer_high_resolution(
        current, 64, HIGHRES_DECRYPTED_OUTPUT_SCALE);
    if (verbose > 0) {
        print_duration_yellow(start, "The resumed 256x256 final layer took: ");
    }
}

void executeResNet256RefreshFromCheckpoint() {
    if (verbose >= 0) {
        cout << "Resuming the native 256x256 Stage 3 refresh from checkpoint."
             << endl;
    }

    controller.num_slots = NATIVE256_SLOTS;
    TensorLayout stage3_layout{64, 64, 16, NATIVE256_SLOTS};
    EncryptedTensor current = load_tensor_checkpoint(
        stage3_layout, "native256-stage3-unrefreshed-v1");
    for (size_t shard = 0; shard < current.shards.size(); shard++) {
        print_ciphertext_stats(
            current.shards[shard],
            "Unrefreshed checkpoint shard " + to_string(shard));
    }
    controller.load_bootstrapping_and_rotation_keys(
        "rotations-layer3.bin", NATIVE256_SLOTS, verbose > 1);

    auto start = start_time();
    current = refresh_stage3_high_resolution(
        std::move(current), verbose > 1);
    save_tensor_checkpoint(current, "native256-stage3-scaled-v1");
    final_layer_high_resolution(
        current, 64, HIGHRES_DECRYPTED_OUTPUT_SCALE);
    if (verbose > 0) {
        print_duration_yellow(
            start, "The resumed 256x256 refresh and final layer took: ");
    }
}

void executeResNet256ReluFromCheckpoint() {
    if (verbose >= 0) {
        cout << "Resuming the native 256x256 final ReLU from checkpoint."
             << endl;
    }

    controller.num_slots = NATIVE256_SLOTS;
    TensorLayout stage3_layout{64, 64, 16, NATIVE256_SLOTS};
    EncryptedTensor current = load_tensor_checkpoint(
        stage3_layout, "native256-layer9-pre-relu-v1");
    for (size_t shard = 0; shard < current.shards.size(); shard++) {
        print_ciphertext_stats(
            current.shards[shard],
            "Pre-ReLU checkpoint shard " + to_string(shard));
    }

    auto start = start_time();
    current = controller.relu_tensor_wide(
        current,
        -NATIVE256_FINAL_RELU_BOUND,
        NATIVE256_FINAL_RELU_BOUND,
        HIGHRES_FINAL_RELU_DEGREE,
        1.0,
        verbose > 1);
    for (size_t shard = 0; shard < current.shards.size(); shard++) {
        print_ciphertext_stats(
            current.shards[shard],
            "Post-ReLU checkpoint shard " + to_string(shard));
    }
    save_tensor_checkpoint(current, "native256-stage3-unrefreshed-v1");
    controller.load_bootstrapping_and_rotation_keys(
        "rotations-layer3.bin", NATIVE256_SLOTS, verbose > 1);
    current = refresh_stage3_high_resolution(
        std::move(current), verbose > 1);
    save_tensor_checkpoint(current, "native256-stage3-scaled-v1");
    final_layer_high_resolution(
        current, 64, HIGHRES_DECRYPTED_OUTPUT_SCALE);
    if (verbose > 0) {
        print_duration_yellow(
            start,
            "The resumed 256x256 final ReLU, refresh, and final layer took: ");
    }
}

void executeResNet256Stage3FromCheckpoint() {
    if (verbose >= 0) {
        cout << "Resuming the native 256x256 Stage 3 from checkpoint." << endl;
    }

    controller.num_slots = NATIVE256_SLOTS;
    TensorLayout stage2_layout{128, 32, 4, NATIVE256_SLOTS};
    EncryptedTensor current = load_tensor_checkpoint(
        stage2_layout, "native256-stage2-v1");
    controller.load_bootstrapping_and_rotation_keys(
        "rotations-layer2.bin", NATIVE256_SLOTS, verbose > 1);

    auto start = start_time();
    current = layer3_256(std::move(current));
    save_tensor_checkpoint(current, "native256-stage3-scaled-v1");
    final_layer_high_resolution(
        current, 64, HIGHRES_DECRYPTED_OUTPUT_SCALE);
    if (verbose > 0) {
        print_duration_yellow(
            start, "The resumed 256x256 Stage 3 and final layer took: ");
    }
}

void executeFinalProbeHighResolution(int resolution) {
    if (verbose >= 0) {
        cout << "Running a synthetic " << resolution << "x" << resolution
             << " final-layer probe." << endl;
        cout << "This checks the existing final rotation keys and aggregation path "
             << "without running the three CNN stages." << endl;
    }

    const int slots = resolution == 256 ? NATIVE256_SLOTS : (1 << 14);
    const int feature_width = resolution / 4;
    controller.num_slots = slots;
    TensorLayout stage3_layout{feature_width, 64, 16, slots};
    vector<double> synthetic_values(
        stage3_layout.channels * stage3_layout.area(), 0.0);
    for (int channel = 0; channel < stage3_layout.channels; channel++) {
        for (int pixel = 0; pixel < stage3_layout.area(); pixel++) {
            synthetic_values[channel * stage3_layout.area() + pixel] =
                0.01 + 0.0001 * channel +
                0.000001 * (pixel % feature_width);
        }
    }

    EncryptedTensor synthetic = controller.encrypt_tensor(
        synthetic_values, stage3_layout, 0);
    auto start = start_time();
    Ctxt encrypted_result = final_layer_high_resolution(
        synthetic, feature_width);

    vector<double> fc_weights = read_values_from_file("../weights/fc.bin");
    if (fc_weights.size() < 640) {
        throw runtime_error("The fully-connected weight file is truncated");
    }
    vector<double> expected(10, 0.0);
    for (int channel = 0; channel < stage3_layout.channels; channel++) {
        double channel_average = 0.0;
        for (int pixel = 0; pixel < stage3_layout.area(); pixel++) {
            channel_average +=
                synthetic_values[channel * stage3_layout.area() + pixel];
        }
        channel_average /= stage3_layout.area();
        for (int class_index = 0; class_index < 10; class_index++) {
            expected[class_index] +=
                channel_average * fc_weights[channel * 10 + class_index];
        }
    }

    vector<double> actual = controller.decrypt_tovector(encrypted_result, 10);
    double max_error = 0.0;
    for (int class_index = 0; class_index < 10; class_index++) {
        max_error = max(max_error,
                        abs(actual[class_index] - expected[class_index]));
    }
    cout << "Final-layer probe maximum absolute error: "
         << scientific << setprecision(6) << max_error << defaultfloat << endl;
    if (max_error > 1e-3) {
        throw runtime_error(
            "The high-resolution final-layer probe exceeded the 1e-3 error tolerance");
    }
    if (verbose > 0) {
        print_duration_yellow(
            start,
            "The " + to_string(resolution) + "x" + to_string(resolution) +
                " final-layer probe took: ");
    }
}

void executeRefreshProbeHighResolution(int resolution) {
    const int slots = resolution == 256 ? NATIVE256_SLOTS : (1 << 14);
    const int feature_width = resolution / 4;
    const double relu_bound = resolution == 256
        ? NATIVE256_FINAL_RELU_BOUND
        : NATIVE128_FINAL_RELU_BOUND;
    const double error_tolerance = resolution == 256 ? 1.5e-1 : 7.5e-2;
    if (verbose >= 0) {
        cout << "Running a synthetic " << resolution << "x" << resolution
             << " Stage 3 refresh and final-layer probe." << endl;
        cout << "The probe includes the degree-"
             << HIGHRES_FINAL_RELU_DEGREE << " final ReLU on [-"
             << relu_bound << ", " << relu_bound
             << "] with its retained 0.10 scale." << endl;
    }

    controller.num_slots = slots;
    TensorLayout stage3_layout{feature_width, 64, 16, slots};
    vector<double> synthetic_values(
        stage3_layout.channels * stage3_layout.area(), 0.0);
    for (int channel = 0; channel < stage3_layout.channels; channel++) {
        for (int pixel = 0; pixel < stage3_layout.area(); pixel++) {
            if (resolution == 256) {
                synthetic_values[channel * stage3_layout.area() + pixel] =
                    -1.0 + 0.04 * channel +
                    0.003 * (pixel % feature_width);
            } else {
                synthetic_values[channel * stage3_layout.area() + pixel] =
                    -0.58 + 0.025 * channel +
                    0.0015 * (pixel % feature_width);
            }
        }
    }

    EncryptedTensor synthetic = controller.encrypt_tensor(
        synthetic_values,
        stage3_layout,
        17);
    synthetic = controller.relu_tensor_wide(
        synthetic,
        -relu_bound,
        relu_bound,
        HIGHRES_FINAL_RELU_DEGREE,
        1.0,
        verbose > 1);
    controller.load_bootstrapping_and_rotation_keys(
        "rotations-layer3.bin", slots, verbose > 1);

    EncryptedTensor refreshed = refresh_stage3_high_resolution(
        std::move(synthetic), verbose > 1);
    for (size_t shard = 0; shard < refreshed.shards.size(); shard++) {
        print_ciphertext_stats(
            refreshed.shards[shard],
            "Refresh probe shard " + to_string(shard));
    }

    auto start = start_time();
    Ctxt encrypted_result = final_layer_high_resolution(
        refreshed, feature_width, HIGHRES_DECRYPTED_OUTPUT_SCALE);
    vector<double> actual = controller.decrypt_tovector(encrypted_result, 10);
    for (double& value : actual) {
        value *= HIGHRES_DECRYPTED_OUTPUT_SCALE;
    }

    vector<double> fc_weights = read_values_from_file("../weights/fc.bin");
    if (fc_weights.size() < 640) {
        throw runtime_error("The fully-connected weight file is truncated");
    }
    vector<double> expected(10, 0.0);
    for (int channel = 0; channel < stage3_layout.channels; channel++) {
        double channel_average = 0.0;
        for (int pixel = 0; pixel < stage3_layout.area(); pixel++) {
            channel_average += max(
                0.0,
                synthetic_values[channel * stage3_layout.area() + pixel]);
        }
        channel_average /= stage3_layout.area();
        channel_average *= 10.0;
        for (int class_index = 0; class_index < 10; class_index++) {
            expected[class_index] +=
                channel_average * fc_weights[channel * 10 + class_index];
        }
    }

    double max_error = 0.0;
    for (int class_index = 0; class_index < 10; class_index++) {
        max_error = max(max_error,
                        abs(actual[class_index] - expected[class_index]));
    }
    cout << "Refresh probe maximum absolute error after client-side rescaling: "
         << scientific << setprecision(6) << max_error << defaultfloat << endl;
    // This comparison includes the deliberate degree-59 approximation of the
    // non-smooth ReLU as well as CKKS error. The wider 256 interval has a
    // proportionally larger approximation tolerance.
    if (max_error > error_tolerance) {
        throw runtime_error(
            "The high-resolution refresh probe exceeded its error tolerance");
    }
    if (verbose > 0) {
        print_duration_yellow(
            start,
            "The " + to_string(resolution) + "x" + to_string(resolution) +
                " refresh and final-layer probe took: ");
    }
}

EncryptedTensor residual_block_native(const EncryptedTensor& in,
                                      const string& weight_prefix,
                                      double first_scale,
                                      double second_scale,
                                      const string& title) {
    bool timing = verbose > 1;
    if (timing) cout << "---Start: " << title << "---" << endl;
    auto start = start_time();

    EncryptedTensor res = controller.convbn_sharded(
        in, weight_prefix + "_conv1.fwgt", in.layout.channels,
        first_scale, false, timing);
    res = controller.bootstrap_tensor(res, timing);
    res = controller.relu_tensor(res, first_scale, timing);
    res = controller.convbn_sharded(
        res, weight_prefix + "_conv2.fwgt", in.layout.channels,
        second_scale, false, timing);
    res = controller.add_tensor(res, controller.mult_tensor(in, second_scale));
    res = controller.bootstrap_tensor(res, timing);
    res = controller.relu_tensor(res, second_scale, timing);

    if (timing) {
        print_duration(start, "Total");
        cout << "---End  : " << title << "---" << endl;
    }
    return res;
}

EncryptedTensor layer1_native(const EncryptedTensor& in) {
    string resolution = to_string(in.layout.width) + "x" + to_string(in.layout.width);
    EncryptedTensor res = residual_block_native(
        in, "../weights/compact_fused/layer1", 1.00, 0.52,
        resolution + " Stage 1 - Block 1");
    res = residual_block_native(
        res, "../weights/compact_fused/layer2", 0.55, 0.36,
        resolution + " Stage 1 - Block 2");
    return residual_block_native(
        res, "../weights/compact_fused/layer3", 0.63, 0.42,
        resolution + " Stage 1 - Block 3");
}

EncryptedTensor layer2_native(const EncryptedTensor& in,
                              int output_channels_per_ciphertext) {
    bool timing = verbose > 1;
    if (timing) cout << "---Start: 64x64 Stage 2 - Block 1---" << endl;
    auto start = start_time();

    EncryptedTensor boot_in = controller.bootstrap_tensor(in, timing);
    EncryptedTensor left = controller.convbn_sharded(
        boot_in, "../weights/compact_fused/layer4_conv1.fwgt",
        32, 0.57, true, timing);
    EncryptedTensor right = controller.convbn_sharded(
        boot_in, "../weights/compact_fused/layer4_downsample.fwgt",
        32, 0.40, true, timing);

    controller.clear_bootstrapping_and_rotation_keys(16384);
    controller.load_rotation_keys("rotations-layer2-downsample.bin", timing);
    left = controller.downsample_stride2_sharded(
        std::move(left), output_channels_per_ciphertext, timing);
    right = controller.downsample_stride2_sharded(
        std::move(right), output_channels_per_ciphertext, timing);

    controller.clear_rotation_keys();
    controller.load_bootstrapping_and_rotation_keys(
        "rotations-layer2.bin", 16384, timing);
    left = controller.bootstrap_tensor(left, timing);
    left = controller.relu_tensor(left, 0.57, timing);
    left = controller.convbn_sharded(
        left, "../weights/compact_fused/layer4_conv2.fwgt",
        32, 0.40, false, timing);
    EncryptedTensor res = controller.add_tensor(left, right);
    res = controller.bootstrap_tensor(res, timing);
    res = controller.relu_tensor(res, 0.40, timing);

    if (timing) {
        print_duration(start, "Total");
        cout << "---End  : 64x64 Stage 2 - Block 1---" << endl;
    }

    res = residual_block_native(
        res, "../weights/compact_fused/layer5", 0.76, 0.37,
        "64x64 Stage 2 - Block 2");
    return residual_block_native(
        res, "../weights/compact_fused/layer6", 0.63, 0.25,
        "64x64 Stage 2 - Block 3");
}

EncryptedTensor layer3_native(const EncryptedTensor& in,
                              int output_channels_per_ciphertext) {
    bool timing = verbose > 1;
    if (timing) cout << "---Start: 64x64 Stage 3 - Block 1---" << endl;
    auto start = start_time();

    EncryptedTensor boot_in = controller.bootstrap_tensor(in, timing);
    EncryptedTensor left = controller.convbn_sharded(
        boot_in, "../weights/compact_fused/layer7_conv1.fwgt",
        64, 0.63, true, timing);
    EncryptedTensor right = controller.convbn_sharded(
        boot_in, "../weights/compact_fused/layer7_downsample.fwgt",
        64, 0.40, true, timing);

    controller.clear_bootstrapping_and_rotation_keys(16384);
    controller.load_rotation_keys("rotations-layer3-downsample.bin", timing);
    left = controller.downsample_stride2_sharded(
        std::move(left), output_channels_per_ciphertext, timing);
    right = controller.downsample_stride2_sharded(
        std::move(right), output_channels_per_ciphertext, timing);

    controller.clear_rotation_keys();
    controller.load_bootstrapping_and_rotation_keys(
        "rotations-layer3.bin", 16384, timing);
    left = controller.bootstrap_tensor(left, timing);
    left = controller.relu_tensor(left, 0.63, timing);
    left = controller.convbn_sharded(
        left, "../weights/compact_fused/layer7_conv2.fwgt",
        64, 0.40, false, timing);
    EncryptedTensor res = controller.add_tensor(left, right);
    res = controller.bootstrap_tensor(res, timing);
    res = controller.relu_tensor(res, 0.40, timing);

    if (timing) {
        print_duration(start, "Total");
        cout << "---End  : 64x64 Stage 3 - Block 1---" << endl;
    }

    res = residual_block_native(
        res, "../weights/compact_fused/layer8", 0.57, 0.33,
        "64x64 Stage 3 - Block 2");
    res = residual_block_native(
        res, "../weights/compact_fused/layer9", 0.69, 0.10,
        "64x64 Stage 3 - Block 3");
    return controller.bootstrap_tensor(res, timing);
}

EncryptedTensor layer2_high_resolution(EncryptedTensor in,
                                       int original_resolution) {
    bool timing = verbose > 1;
    const string resolution = to_string(original_resolution) + "x" +
                              to_string(original_resolution);
    if (timing) {
        cout << "---Start: " << resolution << " Stage 2 - Block 1---"
             << endl;
    }
    auto start = start_time();

    // This function owns its input so bootstrapping can replace each shard
    // immediately instead of retaining another 16-ciphertext tensor.
    for (Ctxt& shard : in.shards) {
        shard = controller.bootstrap(shard, timing);
    }

    EncryptedTensor left = controller.convbn_sharded(
        in, "../weights/compact_fused/layer4_conv1.fwgt",
        32, 0.57, true, timing);
    left = controller.downsample_stride2_sharded(std::move(left), 4, timing);

    EncryptedTensor right = controller.convbn_sharded(
        in, "../weights/compact_fused/layer4_downsample.fwgt",
        32, 0.40, true, timing);
    in.shards.clear();
    in.shards.shrink_to_fit();
    right = controller.downsample_stride2_sharded(std::move(right), 4, timing);

    controller.clear_bootstrapping_and_rotation_keys(in.layout.slots);
    controller.load_bootstrapping_and_rotation_keys(
        "rotations-layer2.bin", in.layout.slots, timing);
    left = controller.bootstrap_tensor(left, timing);
    left = controller.relu_tensor(left, 0.57, timing);
    left = controller.convbn_sharded(
        left, "../weights/compact_fused/layer4_conv2.fwgt",
        32, 0.40, false, timing);
    EncryptedTensor res = controller.add_tensor(left, right);
    res = controller.bootstrap_tensor(res, timing);
    res = controller.relu_tensor(res, 0.40, timing);

    if (timing) {
        print_duration(start, "Total");
        cout << "---End  : " << resolution << " Stage 2 - Block 1---"
             << endl;
    }

    res = residual_block_native(
        res, "../weights/compact_fused/layer5", 0.76, 0.37,
        resolution + " Stage 2 - Block 2");
    return residual_block_native(
        res, "../weights/compact_fused/layer6", 0.63, 0.25,
        resolution + " Stage 2 - Block 3");
}

EncryptedTensor layer3_128(EncryptedTensor in) {
    bool timing = verbose > 1;
    if (timing) cout << "---Start: 128x128 Stage 3 - Block 1---" << endl;
    auto start = start_time();

    for (Ctxt& shard : in.shards) {
        shard = controller.bootstrap(shard, timing);
    }

    EncryptedTensor left = controller.convbn_sharded(
        in, "../weights/compact_fused/layer7_conv1.fwgt",
        64, 0.63, true, timing);
    left = controller.downsample_stride2_sharded(std::move(left), 16, timing);

    EncryptedTensor right = controller.convbn_sharded(
        in, "../weights/compact_fused/layer7_downsample.fwgt",
        64, 0.40, true, timing);
    in.shards.clear();
    in.shards.shrink_to_fit();
    right = controller.downsample_stride2_sharded(std::move(right), 16, timing);

    controller.clear_bootstrapping_and_rotation_keys(16384);
    controller.load_bootstrapping_and_rotation_keys(
        "rotations-layer3.bin", 16384, timing);
    left = controller.bootstrap_tensor(left, timing);
    left = controller.relu_tensor(left, 0.63, timing);
    left = controller.convbn_sharded(
        left, "../weights/compact_fused/layer7_conv2.fwgt",
        64, 0.40, false, timing);
    EncryptedTensor res = controller.add_tensor(left, right);
    res = controller.bootstrap_tensor(res, timing);
    res = controller.relu_tensor(res, 0.40, timing);

    if (timing) {
        print_duration(start, "Total");
        cout << "---End  : 128x128 Stage 3 - Block 1---" << endl;
    }

    res = residual_block_native(
        res, "../weights/compact_fused/layer8", 0.57, 0.33,
        "128x128 Stage 3 - Block 2");
    res = final_residual_block_128(res, timing);

    // Keep the expensive CNN result before its final refresh. If a future
    // OpenFHE parameter adjustment is needed, resume_refresh can retry from
    // here without evaluating the three ResNet stages again.
    save_tensor_checkpoint(res, "native128-stage3-unrefreshed-v6");
    return refresh_stage3_high_resolution(std::move(res), timing);
}

EncryptedTensor layer3_256(EncryptedTensor in) {
    bool timing = verbose > 1;
    if (timing) cout << "---Start: 256x256 Stage 3 - Block 1---" << endl;
    auto start = start_time();

    // Replace each shard in place to keep the larger 2^17-ring tensor within
    // the WSL memory budget.
    for (Ctxt& shard : in.shards) {
        shard = controller.bootstrap(shard, timing);
    }

    EncryptedTensor left = controller.convbn_sharded(
        in, "../weights/compact_fused/layer7_conv1.fwgt",
        64, 0.63, true, timing);
    left = controller.downsample_stride2_sharded(std::move(left), 16, timing);

    EncryptedTensor right = controller.convbn_sharded(
        in, "../weights/compact_fused/layer7_downsample.fwgt",
        64, 0.40, true, timing);
    in.shards.clear();
    in.shards.shrink_to_fit();
    right = controller.downsample_stride2_sharded(std::move(right), 16, timing);

    controller.clear_bootstrapping_and_rotation_keys(NATIVE256_SLOTS);
    controller.load_bootstrapping_and_rotation_keys(
        "rotations-layer3.bin", NATIVE256_SLOTS, timing);
    left = controller.bootstrap_tensor(left, timing);
    left = controller.relu_tensor(left, 0.63, timing);
    left = controller.convbn_sharded(
        left, "../weights/compact_fused/layer7_conv2.fwgt",
        64, 0.40, false, timing);
    EncryptedTensor res = controller.add_tensor(left, right);
    res = controller.bootstrap_tensor(res, timing);
    res = controller.relu_tensor(res, 0.40, timing);

    if (timing) {
        print_duration(start, "Total");
        cout << "---End  : 256x256 Stage 3 - Block 1---" << endl;
    }

    res = residual_block_native(
        res, "../weights/compact_fused/layer8", 0.57, 0.33,
        "256x256 Stage 3 - Block 2");
    res = final_residual_block_256(res, timing);

    save_tensor_checkpoint(res, "native256-stage3-unrefreshed-v1");
    return refresh_stage3_high_resolution(std::move(res), timing);
}

EncryptedTensor final_residual_block_128(const EncryptedTensor& in,
                                         bool timing) {
    if (timing) cout << "---Start: 128x128 Stage 3 - Block 3---" << endl;
    auto start = start_time();

    EncryptedTensor res = controller.convbn_sharded(
        in, "../weights/compact_fused/layer9_conv1.fwgt",
        in.layout.channels, 0.69, false, timing);
    res = controller.bootstrap_tensor(res, timing);
    res = controller.relu_tensor(res, 0.69, timing);
    res = controller.convbn_sharded(
        res, "../weights/compact_fused/layer9_conv2.fwgt",
        in.layout.channels, 0.10, false, timing);
    res = controller.add_tensor(res, controller.mult_tensor(in, 0.10));
    res = controller.bootstrap_tensor(res, timing);

    // Save the clean pre-activation so a final-ReLU adjustment never requires
    // rerunning the CNN. Keep the 0.10 factor already present in the residual
    // branch instead of amplifying the ciphertext and its approximation error
    // by ten. Use a lower-degree polynomial on [-1.25, 1.25], because the real
    // shard-2 activation can slightly exceed 1.0 and Chebyshev extrapolation
    // outside [-1, 1] destabilizes that shard. ReLU is positively homogeneous;
    // the retained factor is restored only after client-side decryption.
    save_tensor_checkpoint(res, "native128-layer9-pre-relu-v4");
    res = controller.relu_tensor_wide(
        res,
        -NATIVE128_FINAL_RELU_BOUND,
        NATIVE128_FINAL_RELU_BOUND,
        HIGHRES_FINAL_RELU_DEGREE,
        1.0,
        timing);

    if (timing) {
        print_duration(start, "Total");
        cout << "---End  : 128x128 Stage 3 - Block 3---" << endl;
    }
    return res;
}

EncryptedTensor final_residual_block_256(const EncryptedTensor& in,
                                         bool timing) {
    if (timing) cout << "---Start: 256x256 Stage 3 - Block 3---" << endl;
    auto start = start_time();

    EncryptedTensor res = controller.convbn_sharded(
        in, "../weights/compact_fused/layer9_conv1.fwgt",
        in.layout.channels, 0.69, false, timing);
    res = controller.bootstrap_tensor(res, timing);
    res = controller.relu_tensor(res, 0.69, timing);
    res = controller.convbn_sharded(
        res, "../weights/compact_fused/layer9_conv2.fwgt",
        in.layout.channels, 0.10, false, timing);
    res = controller.add_tensor(res, controller.mult_tensor(in, 0.10));
    res = controller.bootstrap_tensor(res, timing);

    // Preserve a clean recovery point before the precision-sensitive final
    // activation. The 256x256 path starts with the stable degree-59 ReLU from
    // the 128x128 fix, but uses a wider [-2, 2] domain so a slightly larger
    // activation cannot trigger unstable Chebyshev extrapolation.
    save_tensor_checkpoint(res, "native256-layer9-pre-relu-v1");
    res = controller.relu_tensor_wide(
        res,
        -NATIVE256_FINAL_RELU_BOUND,
        NATIVE256_FINAL_RELU_BOUND,
        HIGHRES_FINAL_RELU_DEGREE,
        1.0,
        timing);

    if (timing) {
        print_duration(start, "Total");
        cout << "---End  : 256x256 Stage 3 - Block 3---" << endl;
    }
    return res;
}

EncryptedTensor refresh_stage3_high_resolution(EncryptedTensor res,
                                                bool timing) {
    // The final ReLU leaves each shard at level 24/25 and its output is
    // deliberately kept at one tenth of the original activation. Consume
    // the remaining level while scaling into a stable bootstrapping interval.
    // The positive factor is restored only after decryption, so it does not
    // change the predicted class and does not amplify ciphertext noise in the
    // final homomorphic reduction.
    EncryptedTensor refreshed;
    refreshed.layout = res.layout;
    refreshed.shards.reserve(res.shards.size());
    for (const Ctxt& shard : res.shards) {
        Ctxt depleted_and_scaled = controller.mult(
            shard, HIGHRES_STAGE3_REFRESH_SCALE);
        depleted_and_scaled = controller.rescale(depleted_and_scaled);
        refreshed.shards.push_back(
            controller.bootstrap(depleted_and_scaled, 17, timing));
    }
    return refreshed;
}

Ctxt final_layer64(const EncryptedTensor& in) {
    if (in.layout.width != 16 || in.layout.channels != 64 ||
        in.layout.channels_per_ciphertext != 64 || in.shards.size() != 1) {
        throw invalid_argument("Unexpected encrypted tensor layout before 64x64 final layer");
    }

    controller.clear_bootstrapping_and_rotation_keys(16384);
    controller.load_rotation_keys("rotations-finallayer.bin", verbose > 1);
    controller.num_slots = 16384;

    const Ctxt& packed = in.shards[0];
    Ptxt weight = controller.encode(
        read_fc_weight("../weights/fc.bin", 256),
        packed->GetLevel(),
        controller.num_slots);

    Ctxt res = controller.rotsum(packed, 256);
    res = controller.mult(
        res, controller.mask_mod(256, res->GetLevel(), 1.0 / 256.0));
    res = controller.repeat(res, 16);
    res = controller.mult_model_parameter(res, weight);
    res = controller.rotsum_padded_blocks(res, 256, 64);

    if (verbose >= 0) {
        cout << "Decrypting the output..." << endl;
        controller.print(res, 10, "Output: ");
    }

    vector<double> clear_result = controller.decrypt_tovector(res, 10);
    auto max_element_iterator = max_element(clear_result.begin(), clear_result.end());
    int index_max = distance(clear_result.begin(), max_element_iterator);
    if (verbose >= 0) {
        cout << "The input image is classified as " << YELLOW_TEXT
             << utils::get_class(index_max) << RESET_COLOR << endl;
        cout << "The index of max element is " << YELLOW_TEXT
             << index_max << RESET_COLOR << endl;
        if (plain) {
            cout << "Plain PyTorch comparison is only available for the original 32x32 path." << endl;
        }
    }
    return res;
}

Ctxt final_layer_high_resolution(const EncryptedTensor& in,
                                 int expected_width,
                                 double decrypted_output_scale) {
    if (in.layout.width != expected_width || in.layout.channels != 64 ||
        in.layout.channels_per_ciphertext != 16 || in.shards.size() != 4) {
        throw invalid_argument(
            "Unexpected encrypted tensor layout before high-resolution final layer");
    }

    const int area = in.layout.area();
    controller.clear_bootstrapping_and_rotation_keys(in.layout.slots);
    controller.load_rotation_keys("rotations-finallayer.bin", verbose > 1);
    controller.num_slots = in.layout.slots;

    Ctxt final_res;
    for (int shard = 0; shard < static_cast<int>(in.shards.size()); shard++) {
        const Ctxt& packed = in.shards[shard];
        print_ciphertext_stats(
            packed, "Final shard " + to_string(shard) + " input");
        Ptxt weight = controller.encode(
            read_fc_weight_range("../weights/fc.bin", shard * 16, 16, area),
            packed->GetLevel(),
            controller.num_slots);

        // Scale before the ten rotate-and-add steps. This computes the same
        // global average as scaling afterwards, while also reducing the CKKS
        // approximation error accumulated by the spatial reduction.
        Ctxt partial = controller.mult(packed, 1.0 / area);
        print_ciphertext_stats(
            partial, "Final shard " + to_string(shard) + " scaled");
        partial = controller.rotsum(partial, area);
        print_ciphertext_stats(
            partial, "Final shard " + to_string(shard) + " spatial sum");
        partial = controller.mult(
            partial,
            controller.mask_mod(area, partial->GetLevel(), 1.0));
        print_ciphertext_stats(
            partial, "Final shard " + to_string(shard) + " masked");
        partial = controller.repeat(partial, 16);
        print_ciphertext_stats(
            partial, "Final shard " + to_string(shard) + " repeated");
        partial = controller.mult_model_parameter(partial, weight);
        print_ciphertext_stats(
            partial, "Final shard " + to_string(shard) + " weighted");
        partial = controller.rotsum_padded_blocks(partial, area, 16);
        print_ciphertext_stats(
            partial, "Final shard " + to_string(shard) + " channel sum");
        final_res = final_res ? controller.add(final_res, partial) : partial;
        print_ciphertext_stats(
            final_res, "Final accumulated through shard " + to_string(shard));
    }

    if (verbose >= 0) cout << "Decrypting the output..." << endl;
    vector<double> clear_result = controller.decrypt_tovector(final_res, 10);
    for (double& value : clear_result) {
        value *= decrypted_output_scale;
    }
    auto max_element_iterator = max_element(clear_result.begin(), clear_result.end());
    int index_max = distance(clear_result.begin(), max_element_iterator);
    if (verbose >= 0) {
        cout << "Output: [ ";
        cout << fixed << setprecision(3);
        for (size_t index = 0; index < clear_result.size(); index++) {
            if (index != 0) cout << ", ";
            cout << clear_result[index];
        }
        cout << " ]" << defaultfloat << endl;
        cout << "The input image is classified as " << YELLOW_TEXT
             << utils::get_class(index_max) << RESET_COLOR << endl;
        cout << "The index of max element is " << YELLOW_TEXT
             << index_max << RESET_COLOR << endl;
        if (plain) {
            cout << "Plain PyTorch comparison is only available for the original 32x32 path." << endl;
        }
    }
    return final_res;
}

void executeResNet20() {
    if (verbose >= 0) cout << "Encrypted ResNet20 classification started." << endl;

    Ctxt firstLayer, resLayer1, resLayer2, resLayer3, finalRes;

    bool print_intermediate_values = false;
    bool print_bootstrap_precision = false;

    if (verbose > 1) {
        print_intermediate_values = true;
        print_bootstrap_precision = true;
    }

    if (input_filename.empty()) {
        input_filename = "../inputs/luis.png";
        if (verbose >= 0) cout << "You did not set any input, I use " << GREEN_TEXT << "../inputs/luis.png" << RESET_COLOR << "." << endl;
    } else {
        if (verbose >= 0) cout << "I am going to encrypt and classify " << GREEN_TEXT<< input_filename << RESET_COLOR << "." << endl;
    }

    vector<double> input_image = read_image(input_filename.c_str());

    Ctxt in = controller.encrypt(input_image, controller.circuit_depth - 4 - get_relu_depth(controller.relu_degree));

    controller.load_bootstrapping_and_rotation_keys("rotations-layer1.bin", 16384, verbose > 1);

    if (print_bootstrap_precision){
        controller.bootstrap_precision(controller.encrypt(input_image, controller.circuit_depth - 2));
    }

    auto start = start_time();

    firstLayer = initial_layer(in);
    if (print_intermediate_values) controller.print(firstLayer, 16384, "Initial layer: ");
  
    /*
     * Layer 1: 16 channels of 32x32
     */
    auto startLayer = start_time();
    resLayer1 = layer1(firstLayer);
    Serial::SerializeToFile("../checkpoints/layer1.bin", resLayer1, SerType::BINARY);
    if (print_intermediate_values) controller.print(resLayer1, 16384, "Layer 1: ");
    if (verbose > 0) print_duration(startLayer, "Layer 1 took:");

    /*
     * Layer 2: 32 channels of 16x16
     */
    startLayer = start_time();
    Serial::DeserializeFromFile("../checkpoints/layer1.bin", resLayer1, SerType::BINARY);
    resLayer2 = layer2(resLayer1);
    Serial::SerializeToFile("../checkpoints/layer2.bin", resLayer2, SerType::BINARY);
    if (print_intermediate_values) controller.print(resLayer2, 8192, "Layer 2: ");
    if (verbose > 0) print_duration(startLayer, "Layer 2 took:");

    /*
     * Layer 2: 64 channels of 8x8
     */
    startLayer = start_time();
    Serial::DeserializeFromFile("../checkpoints/layer2.bin", resLayer2, SerType::BINARY);
    resLayer3 = layer3(resLayer2);
    Serial::SerializeToFile("../checkpoints/layer3.bin", resLayer3, SerType::BINARY);
    if (print_intermediate_values) controller.print(resLayer3, 4096, "Layer 3: ");
    if (verbose > 0) print_duration(startLayer, "Layer 3 took:");


    Serial::DeserializeFromFile("../checkpoints/layer3.bin", resLayer3, SerType::BINARY);
    finalRes = final_layer(resLayer3);
    Serial::SerializeToFile("../checkpoints/finalres.bin", finalRes, SerType::BINARY);

    if (verbose > 0) print_duration_yellow(start, "The evaluation of the whole circuit took: ");
}

Ctxt initial_layer(const Ctxt& in) {
    double scale = 0.90;

    Ctxt res = controller.convbn_initial(in, scale, verbose > 1);
    res = controller.relu(res, scale, verbose > 1);

    return res;
}

Ctxt final_layer(const Ctxt& in) {
    controller.clear_bootstrapping_and_rotation_keys(4096);
    controller.load_rotation_keys("rotations-finallayer.bin", false);

    controller.num_slots = 4096;

    Ptxt weight = controller.encode(read_fc_weight("../weights/fc.bin"), in->GetLevel(), controller.num_slots);

    Ctxt res = controller.rotsum(in, 64);
    res = controller.mult(res, controller.mask_mod(64, res->GetLevel(), 1.0 / 64.0));

    //From here, I need 10 repetitons, but I use 16 since *repeat* goes exponentially
    res = controller.repeat(res, 16);
    res = controller.mult(res, weight);
    res = controller.rotsum_padded(res, 64);

    if (verbose >= 0) {
        cout << "Decrypting the output..." << endl;
        controller.print(res, 10, "Output: ");
    }

    vector<double> clear_result = controller.decrypt_tovector(res, 10);

    //Index of the max element
    auto max_element_iterator = std::max_element(clear_result.begin(), clear_result.end());
    int index_max = distance(clear_result.begin(), max_element_iterator);

    if (verbose >= 0) {
        cout << "The input image is classified as " << YELLOW_TEXT << utils::get_class(index_max) << RESET_COLOR << "" << endl;
        cout << "The index of max element is " << YELLOW_TEXT << index_max << RESET_COLOR << "" << endl;
        if (plain) {
            string command = "python3 ../src/plain/script.py \"" + input_filename + "\"";
            int return_sys = system(command.c_str());
            if (return_sys == 1) {
                cout << "There was an error launching src/plain/script.py. Run it from Python in order to debug it." << endl;
            }
        }
    }


    return res;
}

Ctxt layer3(const Ctxt& in) {
    double scaleSx = 0.63;
    double scaleDx = 0.40;

    bool timing = verbose > 1;

    if (verbose > 1) cout << "---Start: Layer3 - Block 1---" << endl;
    auto start = start_time();
    Ctxt boot_in = controller.bootstrap(in, timing);

    vector<Ctxt> res1sx = controller.convbn3264sx(boot_in, 7, 1, scaleSx, timing); //Questo è lento
    vector<Ctxt> res1dx = controller.convbn3264dx(boot_in, 7, 1, scaleDx, timing); //Questo è lento

    controller.clear_bootstrapping_and_rotation_keys(8192);
    controller.load_rotation_keys("rotations-layer3-downsample.bin", timing);

    //N.B. questo downsampling usa un chain index in meno - posso accelerare convbn3264sx
    Ctxt fullpackSx = controller.downsample256to64(res1sx[0], res1sx[1]);
    Ctxt fullpackDx = controller.downsample256to64(res1dx[0], res1dx[1]);
    res1sx.clear();
    res1dx.clear();

    controller.clear_rotation_keys();
    controller.load_bootstrapping_and_rotation_keys("rotations-layer3.bin", 4096, verbose > 1);

    controller.num_slots = 4096;
    fullpackSx = controller.bootstrap(fullpackSx, timing);

    fullpackSx = controller.relu(fullpackSx, scaleSx, timing);
    fullpackSx = controller.convbn3(fullpackSx, 7, 2, scaleDx, timing);
    Ctxt res1 = controller.add(fullpackSx, fullpackDx);
    res1 = controller.bootstrap(res1, timing);
    res1 = controller.relu(res1, scaleDx, timing);
    if (verbose > 1) print_duration(start, "Total");
    if (verbose > 1) cout << "---End  : Layer3 - Block 1---" << endl;

    double scale = 0.57;


    if (verbose > 1) cout << "---Start: Layer3 - Block 2---" << endl;
    start = start_time();
    Ctxt res2;
    res2 = controller.convbn3(res1, 8, 1, scale, timing);
    res2 = controller.bootstrap(res2, timing);
    res2 = controller.relu(res2, scale, timing);

    scale = 0.33;

    res2 = controller.convbn3(res2, 8, 2, scale, timing);
    res2 = controller.add(res2, controller.mult(res1, scale));
    res2 = controller.bootstrap(res2, timing);
    res2 = controller.relu(res2, scale, timing);
    if (verbose > 1) print_duration(start, "Total");
    if (verbose > 1) cout << "---End  : Layer3 - Block 2---" << endl;

    scale = 0.69;

    if (verbose > 1) cout << "---Start: Layer3 - Block 3---" << endl;
    start = start_time();
    Ctxt res3;

    res3 = controller.convbn3(res2, 9, 1, scale, timing);
    res3 = controller.bootstrap(res3, timing);
    res3 = controller.relu(res3, scale, timing);

    scale = 0.1;

    res3 = controller.convbn3(res3, 9, 2, scale, timing);
    res3 = controller.add(res3, controller.mult(res2, scale));
    res3 = controller.bootstrap(res3, timing);
    res3 = controller.relu(res3, scale, timing);
    res3 = controller.bootstrap(res3, timing);

    if (verbose > 1) print_duration(start, "Total");
    if (verbose > 1) cout << "---End  : Layer3 - Block 3---" << endl;


    return res3;
}

Ctxt layer2(const Ctxt& in) {

    double scaleSx = 0.57;
    double scaleDx = 0.40;


    bool timing = verbose > 1;

    if (verbose > 1) cout << "---Start: Layer2 - Block 1---" << endl;
    auto start = start_time();
    Ctxt boot_in = controller.bootstrap(in, timing);

    vector<Ctxt> res1sx = controller.convbn1632sx(boot_in, 4, 1, scaleSx, timing); //Questo è lento

    vector<Ctxt> res1dx = controller.convbn1632dx(boot_in, 4, 1, scaleDx, timing); //Questo è lento


    controller.clear_bootstrapping_and_rotation_keys(16384);
    controller.load_rotation_keys("rotations-layer2-downsample.bin", timing);

    Ctxt fullpackSx = controller.downsample1024to256(res1sx[0], res1sx[1]);
    Ctxt fullpackDx = controller.downsample1024to256(res1dx[0], res1dx[1]);


    res1sx.clear();
    res1dx.clear();

    controller.clear_rotation_keys();
    controller.load_bootstrapping_and_rotation_keys("rotations-layer2.bin", 8192, verbose > 1);

    controller.num_slots = 8192;
    fullpackSx = controller.bootstrap(fullpackSx, timing);

    fullpackSx = controller.relu(fullpackSx, scaleSx, timing);

    //I use the scale of the right branch since they will be added together
    fullpackSx = controller.convbn2(fullpackSx, 4, 2, scaleDx, timing);
    Ctxt res1 = controller.add(fullpackSx, fullpackDx);
    res1 = controller.bootstrap(res1, timing);
    res1 = controller.relu(res1, scaleDx, timing);
    if (verbose > 1) print_duration(start, "Total");
    if (verbose > 1) cout << "---End  : Layer2 - Block 1---" << endl;

    double scale = 0.76;

    if (verbose > 1) cout << "---Start: Layer2 - Block 2---" << endl;
    start = start_time();
    Ctxt res2;
    res2 = controller.convbn2(res1, 5, 1, scale, timing);
    res2 = controller.bootstrap(res2, timing);
    res2 = controller.relu(res2, scale, timing);

    scale = 0.37;

    res2 = controller.convbn2(res2, 5, 2, scale, timing);
    res2 = controller.add(res2, controller.mult(res1, scale));
    res2 = controller.bootstrap(res2, timing);
    res2 = controller.relu(res2, scale, timing);
    if (verbose > 1) print_duration(start, "Total");
    if (verbose > 1) cout << "---End  : Layer2 - Block 2---" << endl;

    scale = 0.63;

    if (verbose > 1) cout << "---Start: Layer2 - Block 3---" << endl;
    start = start_time();
    Ctxt res3;
    res3 = controller.convbn2(res2, 6, 1, scale, timing);
    res3 = controller.bootstrap(res3, timing);
    res3 = controller.relu(res3, scale, timing);
  
    scale = 0.25;

    res3 = controller.convbn2(res3, 6, 2, scale, timing);
    res3 = controller.add(res3, controller.mult(res2, scale));
    res3 = controller.bootstrap(res3, timing);
    res3 = controller.relu(res3, scale, timing);
    if (verbose > 1) print_duration(start, "Total");
    if (verbose > 1) cout << "---End  : Layer2 - Block 3---" << endl;

    return res3;
}

Ctxt layer1(const Ctxt& in) {
    bool timing = verbose > 1;
    double scale = 1.00;


    if (verbose > 1) cout << "---Start: Layer1 - Block 1---" << endl;
    auto start = start_time();
    Ctxt res1;
    res1 = controller.convbn(in, 1, 1, scale, timing);
    res1 = controller.bootstrap(res1, timing);
    res1 = controller.relu(res1, scale, timing);

    scale = 0.52;

    res1 = controller.convbn(res1, 1, 2, scale, timing);
    res1 = controller.add(res1, controller.mult(in, scale));
    res1 = controller.bootstrap(res1, timing);
    res1 = controller.relu(res1, scale, timing);
    if (verbose > 1) print_duration(start, "Total");
    if (verbose > 1) cout << "---End  : Layer1 - Block 1---" << endl;

    scale = 0.55;


    if (verbose > 1) cout << "---Start: Layer1 - Block 2---" << endl;
    start = start_time();
    Ctxt res2;
    res2 = controller.convbn(res1, 2, 1, scale, timing);
    res2 = controller.bootstrap(res2, timing);
    res2 = controller.relu(res2, scale, timing);

    scale = 0.36;

    res2 = controller.convbn(res2, 2, 2, scale, timing);
    res2 = controller.add(res2, controller.mult(res1, scale));
    res2 = controller.bootstrap(res2, timing);
    res2 = controller.relu(res2, scale, timing);
    if (verbose > 1) print_duration(start, "Total");
    if (verbose > 1) cout << "---End  : Layer1 - Block 2---" << endl;
  
    scale = 0.63;

    if (verbose > 1) cout << "---Start: Layer1 - Block 3---" << endl;
    start = start_time();
    Ctxt res3;
    res3 = controller.convbn(res2, 3, 1, scale, timing);
    res3 = controller.bootstrap(res3, timing);
    res3 = controller.relu(res3, scale, timing);

    scale = 0.42;
  
    res3 = controller.convbn(res3, 3, 2, scale, timing);
    res3 = controller.add(res3, controller.mult(res2, scale));
    res3 = controller.bootstrap(res3, timing);
    res3 = controller.relu(res3, scale, timing);

    if (verbose > 1) print_duration(start, "Total");
    if (verbose > 1) cout << "---End  : Layer1 - Block 3---" << endl;

    return res3;
}

void check_arguments(int argc, char *argv[]) {
    generate_context = -1;
    verbose = 0;
    test = false;
    plain = false;
    resume_final128 = false;
    resume_refresh128 = false;
    resume_relu128 = false;
    resume_stage3_128 = false;
    probe_final128 = false;
    probe_refresh128 = false;
    input_resolution = 256;
    test_encrypted_weights = false;

    for (int i = 1; i < argc; ++i) {
        // Parse options that affect all later path decisions first.
        if (string(argv[i]) == "verbose") {
            if (i + 1 < argc) {
                verbose = atoi(argv[i + 1]);
            }
        }
        if (string(argv[i]) == "resolution" && i + 1 < argc) {
            input_resolution = atoi(argv[i + 1]);
        }
    }

    if (input_resolution != 32 && input_resolution != 64 &&
        input_resolution != 128 && input_resolution != 256) {
        cerr << "This branch supports 'resolution 32', 'resolution 64', "
                "'resolution 128', and 'resolution 256'."
             << endl;
        exit(1);
    }
    string resolution_suffix = input_resolution == 32
        ? ""
        : "_" + to_string(input_resolution);

    for (int i = 1; i < argc; ++i) {
        if (string(argv[i]) == "load_keys") {
            if (i + 1 < argc) {
                controller.parameters_folder = "keys_exp" + string(argv[i + 1]);
                controller.parameters_folder += resolution_suffix;
                if (verbose > 1) cout << "Context folder set to: \"" << controller.parameters_folder << "\"." << endl;
                generate_context = 0;
            }
        }

        if (string(argv[i]) == "test") {
            test = true;
        }

        if (string(argv[i]) == "test_encrypted_weights") {
            test_encrypted_weights = true;
        }

        if (string(argv[i]) == "resume_final") {
            resume_final128 = true;
        }

        if (string(argv[i]) == "resume_refresh") {
            resume_refresh128 = true;
        }

        if (string(argv[i]) == "resume_relu") {
            resume_relu128 = true;
        }

        if (string(argv[i]) == "resume_stage3") {
            resume_stage3_128 = true;
        }

        if (string(argv[i]) == "probe_final") {
            probe_final128 = true;
        }

        if (string(argv[i]) == "probe_refresh") {
            probe_refresh128 = true;
        }

        if (string(argv[i]) == "generate_keys") {
            if (i + 1 < argc) {
                string folder = "";
                if (string(argv[i+1]) == "1") {
                    folder = "keys_exp1";
                    generate_context = 1;
                } else if (string(argv[i+1]) == "2") {
                    folder = "keys_exp2";
                    generate_context = 2;
                } else if (string(argv[i+1]) == "3") {
                    folder = "keys_exp3";
                    generate_context = 3;
                } else if (string(argv[i+1]) == "4") {
                    folder = "keys_exp4";
                    generate_context = 4;
                } else {
                    cerr << "Set a proper value for 'generate_keys'. For instance, use '1'. Check the README.md" << endl;
                    exit(1);
                }

                folder += resolution_suffix;

                struct stat sb;
                if (stat(("../" + folder).c_str(), &sb) == 0) {
                    cerr << "The keys folder \"" << folder << "\" already exists, I will abort.";
                    exit(1);
                }
                else {
                    mkdir(("../" + folder).c_str(), 0777);
                }

                controller.parameters_folder = folder;
                if (verbose > 1) cout << "Context folder set to: \"" << controller.parameters_folder << "\"." << endl;
            }
        }
        if (string(argv[i]) == "input") {
            if (i + 1 < argc) {
                input_filename = "../" + string(argv[i + 1]);
                if (verbose > 1) cout << "Input image set to: \"" << input_filename << "\"." << endl;
            }
        }

        if (string(argv[i]) == "plain") {
            plain = true;
        }

        if (string(argv[i]) == "weights") {
            if (i + 1 >= argc) {
                cerr << "The 'weights' argument requires either 'encrypted' or 'plaintext'."
                     << endl;
                exit(1);
            }

            string mode = string(argv[i + 1]);
            if (mode == "encrypted") {
                controller.set_encrypt_model_parameters(true);
            } else if (mode == "plaintext") {
                controller.set_encrypt_model_parameters(false);
            } else {
                cerr << "Unknown weight mode '" << mode
                     << "'. Use 'weights encrypted' or 'weights plaintext'."
                     << endl;
                exit(1);
            }
        }

    }

    int special_modes = static_cast<int>(resume_final128) +
                        static_cast<int>(resume_refresh128) +
                        static_cast<int>(resume_relu128) +
                        static_cast<int>(resume_stage3_128) +
                        static_cast<int>(probe_final128) +
                        static_cast<int>(probe_refresh128);
    if (special_modes > 1) {
        cerr << "Use only one resume or probe mode at a time." << endl;
        exit(1);
    }
    if (test_encrypted_weights && special_modes > 0) {
        cerr << "Do not combine 'test_encrypted_weights' with a resume or probe mode."
             << endl;
        exit(1);
    }
    if ((probe_final128 || probe_refresh128) &&
        input_resolution != 128 && input_resolution != 256) {
        cerr << "The synthetic probe modes require 'resolution 128' or "
                "'resolution 256'."
             << endl;
        exit(1);
    }
    if ((resume_final128 || resume_refresh128 || resume_relu128 ||
         resume_stage3_128) &&
        input_resolution != 128 && input_resolution != 256) {
        cerr << "The checkpoint resume modes require 'resolution 128' or "
                "'resolution 256'."
             << endl;
        exit(1);
    }

}

vector<double> read_image(const char *filename, int expected_size) {
    int width = 0;
    int height = 0;
    int channels_in_file = 0;
    unsigned char* image_data = stbi_load(
        filename, &width, &height, &channels_in_file, 3);

    if (!image_data) {
        cerr << "Could not load the image in " << filename << endl;
        exit(1);
    }

    if (width != expected_size || height != expected_size) {
        cerr << "Expected an exact " << expected_size << "x" << expected_size
             << " image, but " << filename << " is " << width << "x" << height
             << ". Resize it explicitly before encrypted inference." << endl;
        stbi_image_free(image_data);
        exit(1);
    }

    vector<double> imageVector;
    imageVector.reserve(width * height * 3);

    for (int i = 0; i < width * height; ++i) {
        //Channel R
        imageVector.push_back(static_cast<double>(image_data[3 * i]) / 255.0f);
    }
    for (int i = 0; i < width * height; ++i) {
        //Channel G
        imageVector.push_back(static_cast<double>(image_data[1 + 3 * i]) / 255.0f);
    }
    for (int i = 0; i < width * height; ++i) {
        //Channel B
        imageVector.push_back(static_cast<double>(image_data[2 + 3 * i]) / 255.0f);
    }

    stbi_image_free(image_data);

    return imageVector;
}
