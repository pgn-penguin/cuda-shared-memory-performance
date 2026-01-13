#include <vector>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel 使用 shared memory，所有 threads 共用一塊（以權重為例）
template <typename scalar_t>
__global__ void custom_conv_shared_forward_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int length,
    int kernel_size,
    int stride,
    int padding,
    int groups) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int total_threads = gridDim.x * blockDim.x;

    int output_length = (length + 2 * padding - kernel_size) / stride + 1;
    int channels_per_group = in_channels / groups;
    int filters_per_group  = out_channels / groups;

    // 動態 shared memory 宣告（bytes 由啟動第三參數提供）
    extern __shared__ __align__(sizeof(scalar_t)) unsigned char shared_mem[];
    scalar_t* shared_weight = reinterpret_cast<scalar_t*>(shared_mem);

    int total_weight = out_channels * channels_per_group * kernel_size;
    // 將 global weight 複製到 shared memory（以 blockDim.x 為 stride）
    for (int i = threadIdx.x; i < total_weight; i += blockDim.x) {
        shared_weight[i] = weight[i];
    }

    // 確保所有權重已載入後再開始計算
    __syncthreads();

    // 主計算：一個 thread 可負責多個輸出元素（grid-stride loop）
    int total_outputs = batch_size * out_channels * output_length;
    for (int idx = tid; idx < total_outputs; idx += total_threads) {
        int l_out  = idx % output_length;
        int c_out = (idx / output_length) % out_channels;
        int b_out  = idx / (out_channels * output_length);
        int group_idx = c_out / filters_per_group;

        scalar_t sum = 0;

        for (int c_in = 0; c_in < channels_per_group; ++c_in) {
            int in_c = group_idx * channels_per_group + c_in;
            for (int k = 0; k < kernel_size; ++k) {
                int l_in = l_out * stride + k - padding;
                if (l_in >= 0 && l_in < length) {
                    scalar_t input_val = input[b_out * in_channels * length +
                                               in_c * length + l_in];
                    int weight_idx = c_out * (channels_per_group * kernel_size)
                                   + c_in * kernel_size + k;
                    sum += input_val * shared_weight[weight_idx];
                }
            }
        }

        // 完成累加後再寫回
        output[idx] = sum;
    }
}

// Host 端包裝：計算 shared_mem_size、設定 opt-in、正確啟動 kernel
std::vector<torch::Tensor> custom_conv_shared_cuda_forward(
    torch::Tensor input,
    torch::Tensor weight,
    int stride,
    int padding,
    int groups) {

    const auto batch_size   = input.size(0);
    const auto in_channels  = input.size(1);
    const auto length    = input.size(2);
    const auto out_channels = weight.size(0);
    const auto kernel_size  = weight.size(2);
    const int output_length = (length + 2 * padding - kernel_size) / stride + 1;

    auto output = torch::zeros({batch_size, out_channels, output_length}, input.options());

    // 設定 threads、blocks（可依總輸出元素數調整）
    const int threads = 256;
    const int blocks  = 256;

    const int channels_per_group = in_channels / groups;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(),"custom_conv_forward_cuda",([&] {
        using scalar_t_ = scalar_t;

        // 動態 shared memory 需求（bytes）
        const size_t shared_mem_size = static_cast<size_t>(out_channels) * static_cast<size_t>(channels_per_group) 
        * static_cast<size_t>(kernel_size) * sizeof(scalar_t_);

        // 如需超過 48KB，在 Ampere cc 8.6 需 opt-in 到 ~100KB，建議先查詢可用上限再設置
        cudaFuncSetAttribute(custom_conv_shared_forward_kernel<scalar_t_>,cudaFuncAttributeMaxDynamicSharedMemorySize,49152);  // 48 KB（RTX 3080, cc 8.6）
        // cudaFuncSetAttribute(custom_conv_shared_forward_kernel<scalar_t_>,cudaFuncAttributeMaxDynamicSharedMemorySize,10240);  // 10 KB（RTX 3080, cc 8.6）

        // 啟動（第三參數為動態 shared memory bytes；第四參數可傳 stream）
        custom_conv_shared_forward_kernel<scalar_t_><<<blocks, threads, shared_mem_size>>>(
            input.data_ptr<scalar_t_>(),
            weight.data_ptr<scalar_t_>(),
            output.data_ptr<scalar_t_>(),
            batch_size,
            in_channels,
            out_channels,
            length,
            kernel_size,
            stride,
            padding,
            groups);
    }));

    return {output};
}