#include <iostream>
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
void generate_evaluation_keys64();
void generate_evaluation_keys128();

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
EncryptedTensor layer2_128(EncryptedTensor in);
EncryptedTensor layer3_128(EncryptedTensor in);
Ctxt final_layer64(const EncryptedTensor& in);
Ctxt final_layer128(const EncryptedTensor& in);

FHEController controller;

int generate_context;
string input_filename;
int verbose;
bool test;
bool plain;
int input_resolution;

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
        switch (generate_context) {
            case 1:
                controller.generate_context(16, 52, 48, 2, 3, 3, 59, true);
                break;
            case 2:
                controller.generate_context(16, 50, 46, 3, 4, 4, 200, true);
                break;
            case 3:
                controller.generate_context(16, 50, 46, 3, 5, 4, 119, true);
                break;
            case 4:
                controller.generate_context(16, 48, 44, 2, 4, 4, 59, true);
                break;
            default:
                controller.generate_context(true);
                break;
        }

        if (verbose > 1) cout << "Basic context built. Now generating bootstrapping and rotations keys..." << endl;

        if (verbose > 1) cout << "(It may take a while, depending on the machine)" << endl;


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

    if (input_resolution == 128) {
        executeResNet128();
    } else if (input_resolution == 64) {
        executeResNet64();
    } else {
        executeResNet20();
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
    current = layer2_128(std::move(current));
    if (verbose > 0) print_duration(start_layer, "128x128 stage 2 took:");

    start_layer = start_time();
    current = layer3_128(std::move(current));
    if (verbose > 0) print_duration(start_layer, "128x128 stage 3 took:");

    final_layer128(current);
    if (verbose > 0) {
        print_duration_yellow(start, "The native 128x128 circuit evaluation took: ");
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

EncryptedTensor layer2_128(EncryptedTensor in) {
    bool timing = verbose > 1;
    if (timing) cout << "---Start: 128x128 Stage 2 - Block 1---" << endl;
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

    controller.clear_bootstrapping_and_rotation_keys(16384);
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
        cout << "---End  : 128x128 Stage 2 - Block 1---" << endl;
    }

    res = residual_block_native(
        res, "../weights/compact_fused/layer5", 0.76, 0.37,
        "128x128 Stage 2 - Block 2");
    return residual_block_native(
        res, "../weights/compact_fused/layer6", 0.63, 0.25,
        "128x128 Stage 2 - Block 3");
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
    res = residual_block_native(
        res, "../weights/compact_fused/layer9", 0.69, 0.10,
        "128x128 Stage 3 - Block 3");
    return controller.bootstrap_tensor(res, timing);
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
    res = controller.mult(res, weight);
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

Ctxt final_layer128(const EncryptedTensor& in) {
    if (in.layout.width != 32 || in.layout.channels != 64 ||
        in.layout.channels_per_ciphertext != 16 || in.shards.size() != 4) {
        throw invalid_argument("Unexpected encrypted tensor layout before 128x128 final layer");
    }

    controller.clear_bootstrapping_and_rotation_keys(16384);
    controller.load_rotation_keys("rotations-finallayer.bin", verbose > 1);
    controller.num_slots = 16384;

    Ctxt final_res;
    for (int shard = 0; shard < static_cast<int>(in.shards.size()); shard++) {
        const Ctxt& packed = in.shards[shard];
        Ptxt weight = controller.encode(
            read_fc_weight_range("../weights/fc.bin", shard * 16, 16, 1024),
            packed->GetLevel(),
            controller.num_slots);

        Ctxt partial = controller.rotsum(packed, 1024);
        partial = controller.mult(
            partial,
            controller.mask_mod(1024, partial->GetLevel(), 1.0 / 1024.0));
        partial = controller.repeat(partial, 16);
        partial = controller.mult(partial, weight);
        partial = controller.rotsum_padded_blocks(partial, 1024, 16);
        final_res = final_res ? controller.add(final_res, partial) : partial;
    }

    if (verbose >= 0) {
        cout << "Decrypting the output..." << endl;
        controller.print(final_res, 10, "Output: ");
    }

    vector<double> clear_result = controller.decrypt_tovector(final_res, 10);
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
    input_resolution = 128;

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

    if (input_resolution != 32 && input_resolution != 64 && input_resolution != 128) {
        cerr << "This branch supports 'resolution 32', 'resolution 64', and 'resolution 128'." << endl;
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
