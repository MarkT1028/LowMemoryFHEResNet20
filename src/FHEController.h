//
// Created by Lorenzo on 24/10/23.
//

#ifndef PAPERRESNET_FHECONTROLLER_H
#define PAPERRESNET_FHECONTROLLER_H

#include "openfhe.h"
#include "ciphertext-ser.h"
#include "scheme/ckksrns/ckksrns-ser.h"
#include "ciphertext-ser.h"
#include "cryptocontext-ser.h"
#include "key/key-ser.h"
#include <thread>

#include "Utils.h"

using namespace lbcrypto;
using namespace std;
using namespace std::chrono;

using namespace utils;

using Ptxt = Plaintext;
using Ctxt = Ciphertext<DCRTPoly>;

struct TensorLayout {
    int width = 0;
    int channels = 0;
    int channels_per_ciphertext = 0;
    int slots = 0;

    int area() const {
        return width * width;
    }

    int ciphertext_count() const {
        return (channels + channels_per_ciphertext - 1) / channels_per_ciphertext;
    }
};

struct EncryptedTensor {
    TensorLayout layout;
    vector<Ctxt> shards;
};

struct FusedConvWeights {
    int out_channels = 0;
    int in_channels = 0;
    int kernel_size = 0;
    vector<double> weights;
    vector<double> bias;

    double at(int output_channel, int input_channel, int kernel_index) const {
        int kernel_elements = kernel_size * kernel_size;
        return weights[((output_channel * in_channels) + input_channel) * kernel_elements + kernel_index];
    }
};

class FHEController {
    CryptoContext<DCRTPoly> context;

public:
    int circuit_depth;
    int num_slots;

    FHEController() {}

    /*
     * Context generating/loading stuff
     */
    void generate_context(bool serialize = false);
    void generate_context(int log_ring,
                          int log_scale,
                          int log_primes,
                          int digits_hks,
                          int cts_levels,
                          int stc_levels,
                          int relu_deg,
                          bool serialize = false,
                          int batch_slots = 1 << 14);
    void load_context(bool verbose = true);
    void test_context();
    bool test_encrypted_model_parameter_ops();

    /*
     * Generating bootstrapping and rotation keys stuff
     */
    void generate_bootstrapping_keys(int bootstrap_slots);
    void generate_rotation_keys(vector<int> rotations, bool serialize = false, string filename = "");
    void generate_bootstrapping_and_rotation_keys(vector<int> rotations,
                                                  int bootstrap_slots,
                                                  bool serialize,
                                                  const string& filename);


    void load_bootstrapping_and_rotation_keys(const string& filename, int bootstrap_slots, bool verbose);
    void load_rotation_keys(const string& filename, bool verbose);
    void clear_bootstrapping_and_rotation_keys(int bootstrap_num_slots);
    void clear_rotation_keys();
    void clear_context(int bootstrapping_key_slots);


    /*
     * CKKS Encoding/Decoding/Encryption/Decryption
     */
    Ptxt encode(const vector<double>& vec, int level, int plaintext_num_slots);
    Ptxt encode(double val, int level, int plaintext_num_slots);
    Ctxt encrypt(const vector<double>& vec, int level = 0, int plaintext_num_slots = 0);
    Ctxt encrypt_ptxt(const Ptxt& p);
    Ptxt decrypt(const Ctxt& c);
    vector<double> decrypt_tovector(const Ctxt& c, int slots);


    /*
     * Homomorphic operations
     */
    Ctxt add(const Ctxt& c1, const Ctxt& c2);
    Ctxt mult(const Ctxt& c, double d);
    Ctxt mult(const Ctxt& c, const Ptxt& p);
    Ctxt rescale(const Ctxt& c);
    Ctxt mult_model_parameter(const Ctxt& c, const Ptxt& parameter);
    Ctxt add_model_parameter(const Ctxt& c, const Ptxt& parameter);
    void set_encrypt_model_parameters(bool enabled);
    bool model_parameters_are_encrypted() const;
    void print_model_parameter_stats() const;
    Ctxt bootstrap(const Ctxt& c, bool timing = false);
    Ctxt bootstrap(const Ctxt& c, int precision, bool timing = false);
    Ctxt relu(const Ctxt& c, double scale, bool timing = false);
    Ctxt relu_wide(const Ctxt& c, double a, double b, int degree, double scale, bool timing = false);

    /*
     * I/O
     */
    Ctxt read_input(const string& filename, double scale = 1);
    void print(const Ctxt& c, int slots = 0, string prefix = "");
    void print_padded(const Ctxt& c, int slots = 0, int padding = 1, string prefix = "");
    void print_min_max(const Ctxt& c);

    /*
     * Convolutional Neural Network functions
     */
    Ctxt convbn_initial(const Ctxt &in, double scale = 0.5, bool timing = false);
    Ctxt convbn(const Ctxt &in, int layer, int n, double scale = 0.5, bool timing = false);
    Ctxt convbn2(const Ctxt &in, int layer, int n, double scale = 0.5, bool timing = false);
    Ctxt convbn3(const Ctxt &in, int layer, int n, double scale = 0.5, bool timing = false);
    vector<Ctxt> convbn1632sx(const Ctxt &in, int layer, int n, double scale = 0.5, bool timing = false);
    vector<Ctxt> convbn1632dx(const Ctxt &in, int layer, int n, double scale = 0.5, bool timing = false);
    vector<Ctxt> convbn3264sx(const Ctxt &in, int layer, int n, double scale = 0.5, bool timing = false);
    vector<Ctxt> convbn3264dx(const Ctxt &in, int layer, int n, double scale = 0.5, bool timing = false);

    Ctxt downsample1024to256(const Ctxt& c1, const Ctxt& c2);
    Ctxt downsample256to64(const Ctxt &c1, const Ctxt &c2);

    Ctxt rotsum(const Ctxt &in, int slots);
    Ctxt rotsum_padded(const Ctxt &in, int slots);
    Ctxt rotsum_padded_blocks(const Ctxt& in, int block_size, int blocks);

    Ctxt repeat(const Ctxt &in, int slots);

    /*
     * Resolution-independent channel-sharded tensors.  The native 64x64 path
     * keeps every ciphertext at or below 16384 logical slots, so it can reuse
     * the paper's 2^16 ring dimension instead of increasing memory fourfold.
     */
    EncryptedTensor encrypt_tensor(const vector<double>& values,
                                   const TensorLayout& layout,
                                   int level = 0);
    EncryptedTensor convbn_sharded(const EncryptedTensor& in,
                                   const string& compact_weight_file,
                                   int output_channels,
                                   double scale = 0.5,
                                   bool stride2_output = false,
                                   bool timing = false);
    EncryptedTensor downsample_stride2_sharded(EncryptedTensor in,
                                               int output_channels_per_ciphertext,
                                               bool timing = false);
    EncryptedTensor bootstrap_tensor(const EncryptedTensor& in, bool timing = false);
    EncryptedTensor relu_tensor(const EncryptedTensor& in, double scale, bool timing = false);
    EncryptedTensor relu_tensor_wide(const EncryptedTensor& in,
                                     double lower_bound,
                                     double upper_bound,
                                     int degree,
                                     double scale,
                                     bool timing = false);
    EncryptedTensor add_tensor(const EncryptedTensor& left, const EncryptedTensor& right);
    EncryptedTensor mult_tensor(const EncryptedTensor& in, double value);

    //TODO: studia sta roba
    Ctxt convbnV2(const Ctxt &in, int layer, int n, double scale = 0.5, bool timing = false);
    Ctxt convbn1632sxV2(const Ctxt &in, int layer, int n, double scale = 0.5, bool timing = false);
    Ctxt convbn1632dxV2(const Ctxt &in, int layer, int n, double scale = 0.5, bool timing = false);


    /*
     * Masking things
     */
    Ptxt gen_mask(int n, int level);
    Ptxt mask_first_n(int n, int level);
    Ptxt mask_second_n(int n, int level);
    Ptxt mask_first_n_mod(int n, int padding, int pos, int level);
    Ptxt mask_first_n_mod2(int n, int padding, int pos, int level);
    Ptxt mask_channel(int n, int level);
    Ptxt mask_channel_2(int n, int level);
    Ptxt mask_from_to(int from, int to, int level);

    Ptxt mask_mod(int n, int level, double custom_val);

    void bootstrap_precision(const Ctxt& c);

    int relu_degree = 119;
    string parameters_folder = "NO_FOLDER";

private:
    Ctxt encrypt_model_parameter(const Ptxt& parameter, bool multiplicative);

    KeyPair<DCRTPoly> key_pair;
    vector<uint32_t> level_budget = {4, 4};
    bool encrypt_model_parameters = true;
    uint64_t encrypted_weight_count = 0;
    uint64_t encrypted_bias_count = 0;
    chrono::nanoseconds model_parameter_encryption_time = chrono::nanoseconds::zero();

    FusedConvWeights load_fused_conv_weights(const string& filename) const;
    vector<Ctxt> spatial_rotations(const Ctxt& in, int width, int kernel_size);
    Ptxt sharded_conv_diagonal(const FusedConvWeights& weights,
                               const TensorLayout& input_layout,
                               int input_shard,
                               int output_shard,
                               int output_channels,
                               int diagonal,
                               int kernel_index,
                               int level,
                               double scale,
                               bool stride2_output);
    Ptxt sharded_bias(const FusedConvWeights& weights,
                      const TensorLayout& output_layout,
                      int output_shard,
                      int level,
                      double scale,
                      bool stride2_output);
    Ptxt row_compaction_mask(int width,
                             int channels_per_ciphertext,
                             int row,
                             int level);
    Ptxt channel_compaction_mask(int width,
                                 int channels_per_ciphertext,
                                 int channel,
                                 int level);


};


#endif //PAPERRESNET_FHECONTROLLER_H
