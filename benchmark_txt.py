import torch
import torch.nn as nn
import time
import json
import os
from custom_conv_shared import CustomConvShared  # 改為導入 custom_conv/CustomConv or custom_conv_shared/CustomConvShared

def get_memory_stats():
    return {
        'allocated': torch.cuda.memory_allocated(),
        'reserved': torch.cuda.max_memory_reserved(),
        'cached': torch.cuda.memory_reserved()
    }

def benchmark_custom_conv(
    batch_size=1,
    in_channels=1,
    out_channels=1,
    input_size=1,
    kernel_size=1,
    stride=1,
    padding=1,
    groups=1,
    num_warmup=30, # 設定預熱次數
    num_tests=300 # 設定測試次數
):
    output_lines = []
    output_lines.append("預熱 GPU 中...\n")

    output_length = (input_size - kernel_size + 2 * padding) // stride + 1
    output_size = batch_size * out_channels * output_length

    input_tensor = torch.randn(batch_size, in_channels, input_size).cuda()
    pytorch_conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                             stride=stride, padding=padding, groups=groups).cuda()
    custom_conv = CustomConvShared(in_channels, out_channels, kernel_size, 
                              stride=stride, padding=padding, groups=groups).cuda() #需要改

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    for _ in range(num_warmup):
        _ = pytorch_conv(input_tensor)
        _ = custom_conv(input_tensor)
    torch.cuda.synchronize()

    initial_memory = get_memory_stats()

    start_event.record()
    for _ in range(num_tests):
        output_pytorch = pytorch_conv(input_tensor)
        torch.cuda.synchronize()
    end_event.record()
    end_event.synchronize()
    pytorch_time = start_event.elapsed_time(end_event) / num_tests

    start_event.record()
    for _ in range(num_tests):
        output_custom = custom_conv(input_tensor)
        torch.cuda.synchronize()
    end_event.record()
    end_event.synchronize()
    custom_time = start_event.elapsed_time(end_event) / num_tests
    
    if groups == 1:
        total_ops = (2 * kernel_size * in_channels * out_channels * output_length)
    else:
        channels_per_group = in_channels // groups
        out_channels_per_group = out_channels // groups
        total_ops = (2 * kernel_size * channels_per_group * out_channels_per_group * output_length * groups)

    pytorch_gflops = (total_ops * batch_size) / (pytorch_time * 1e9)
    custom_gflops = (total_ops * batch_size) / (custom_time * 1e9)

    final_memory = get_memory_stats()
    memory_increase = {
        'allocated': final_memory['allocated'] - initial_memory['allocated'],
        'reserved': final_memory['reserved'] - initial_memory['reserved']
    }

    with torch.no_grad():
        output_diff = torch.abs(output_pytorch - output_custom)
        max_diff = output_diff.max().item()
        mean_diff = output_diff.mean().item()
        std_diff = output_diff.std().item()
        relative_error = torch.norm(output_diff) / torch.norm(output_pytorch)

    output_lines.append("------------------------- 性能/精度比較 結果 -------------------------\n")
    output_lines.append(f"{'最大誤差:':3} {max_diff:.6f}\t{'平均誤差:':3} {mean_diff:.6f}\t{'相對誤差:':3} {relative_error.item():.6f}\n")
    
    output_lines.append("PyTorch 原始一維卷積層:")
    output_lines.append(f"{'- 平均時間:':3} {pytorch_time:.3f} ms\t{'GFLOPS:':5} {pytorch_gflops:.2f}")
    
    output_lines.append("\nCustom Conv 1D (固定配置):")
    output_lines.append(f"{'- 平均時間:':3} {custom_time:.3f} ms\t{'GFLOPS:':5} {custom_gflops:.2f}    {'速度提升:':3} {((pytorch_time - custom_time) / pytorch_time * 100):.2f}%")

    for line in output_lines:
        print(line)

    results = {
        'configuration': {
            'batch_size': batch_size,
            'in_channels': in_channels,
            'out_channels': out_channels,
            'input_size': input_size,
            'kernel_size': kernel_size,
            'stride': stride,
            'padding': padding,
            'groups': groups
        },
        'performance': {
            'pytorch_time_ms': pytorch_time,
            'custom_time_ms': custom_time,
            'speedup_percent': ((pytorch_time - custom_time) / pytorch_time * 100),
            'pytorch_gflops': pytorch_gflops,
            'custom_gflops': custom_gflops
        },
        'accuracy': {
            'max_diff': max_diff,
            'mean_diff': mean_diff,
            'std_diff': std_diff,
            'relative_error': relative_error.item()
        },
        'memory': memory_increase,
        'formatted_output': output_lines
    }
    return results

if __name__ == "__main__":
    configs = [
        {'batch_size': 128, 'in_channels': 512, 'out_channels': 1024, 'input_size': 448, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'groups': 1},
        {'batch_size': 128, 'in_channels': 512, 'out_channels': 1024, 'input_size': 448, 'kernel_size': 5, 'stride': 1, 'padding': 1, 'groups': 1},
        {'batch_size': 128, 'in_channels': 512, 'out_channels': 1024, 'input_size': 448, 'kernel_size': 7, 'stride': 1, 'padding': 1, 'groups': 1},
        {'batch_size': 128, 'in_channels': 512, 'out_channels': 1024, 'input_size': 448, 'kernel_size': 9, 'stride': 1, 'padding': 1, 'groups': 1},
        
        {'batch_size': 128, 'in_channels': 512, 'out_channels': 1024, 'input_size': 448, 'kernel_size': 3, 'stride': 2, 'padding': 1, 'groups': 1},
        {'batch_size': 128, 'in_channels': 512, 'out_channels': 1024, 'input_size': 448, 'kernel_size': 3, 'stride': 4, 'padding': 1, 'groups': 1},
        {'batch_size': 128, 'in_channels': 512, 'out_channels': 1024, 'input_size': 448, 'kernel_size': 3, 'stride': 8, 'padding': 1, 'groups': 1},
        {'batch_size': 128, 'in_channels': 512, 'out_channels': 1024, 'input_size': 448, 'kernel_size': 3, 'stride': 16, 'padding': 1, 'groups': 1},
        
        {'batch_size': 128, 'in_channels': 512, 'out_channels': 1024, 'input_size': 448, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'groups': 2},
        {'batch_size': 128, 'in_channels': 512, 'out_channels': 1024, 'input_size': 448, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'groups': 4},
        {'batch_size': 128, 'in_channels': 512, 'out_channels': 1024, 'input_size': 448, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'groups': 8},
        {'batch_size': 128, 'in_channels': 512, 'out_channels': 1024, 'input_size': 448, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'groups': 16},
        
        {'batch_size': 128, 'in_channels':  32, 'out_channels':  64, 'input_size': 448, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'groups': 1},
        {'batch_size': 128, 'in_channels':  64, 'out_channels': 128, 'input_size': 448, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'groups': 1},
        {'batch_size': 128, 'in_channels': 128, 'out_channels': 256, 'input_size': 448, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'groups': 1},
        {'batch_size': 128, 'in_channels': 256, 'out_channels': 512, 'input_size': 448, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'groups': 1},
    
    ]

    all_results = {}
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    txt_file = f'benchmark_conv1d_fixed_config_results_{timestamp}.txt'

    header = []
    TOTAL_THREADS = 256 * 256
    header.append("Custom Conv 1D (固定 Block/Thread 配置) 基準測試")
    header.append("CUDA 配置 (已在 custom_conv_cuda.cu 中固定):")
    header.append(f"{'- Threads per Block:':25} 256")
    header.append(f"{'- Number of Blocks:':25} 256")
    header.append(f"{'- Total Threads:':25} {TOTAL_THREADS:,}")

    try:
        with open(txt_file, 'w', encoding='utf-8') as f_txt:
            
            for line in header:
                f_txt.write(line + '\n')
            f_txt.write("\n")
            
            for i, config in enumerate(configs):
                print("=" * 130)
                print(f"運行測試 {i+1}/{len(configs)}")
                print(f"配置: {config}\n")

                f_txt.write("=" * 130 + "\n")
                f_txt.write(f"運行測試 {i+1}/{len(configs)}\n")
                f_txt.write(f"配置: {config}\n\n")

                torch.cuda.empty_cache()
                key = f"test_{i+1}_b{config['batch_size']}_i{config['input_size']}_c{config['in_channels']}_s{config['stride']}_g{config['groups']}"
                result = benchmark_custom_conv(**config)
                all_results[key] = result

                for line in result['formatted_output']:
                    f_txt.write(line + "\n")

        print(f"\n結果已保存： {txt_file}")

    except Exception as e:
        print(f"錯誤：{e}")
        import traceback
        traceback.print_exc()
