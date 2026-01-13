# Custom Convolution with CUDA Shared Memory

高效能卷積層的 CUDA 實作，使用共享記憶體優化，支援 PyTorch 整合。

## 特點

- ✅ 使用 CUDA 共享記憶體優化卷積
- ✅ 完整支援 PyTorch `nn.Module` 介面
- ✅ 支援 stride、padding 和 groups 參數
- ✅ 包含完整的效能基準測試工具
- ✅ 與 PyTorch 原生實作結果精度驗證

## 系統需求

- Python 3.7+
- PyTorch 1.8+
- CUDA 11.0+ (測試於 sm_86 架構)
- C++ 編譯器 (支援 C++14)

## 安裝
```bash
# 編譯 CUDA 擴展
python setup.py install
```

## 快速使用
```python
import torch
from custom_conv_shared import CustomConvShared

# 創建卷積層
conv = CustomConvShared(
    in_channels=512,
    out_channels=1024,
    kernel_size=3,
    stride=1,
    padding=1,
    groups=1
).cuda()

# 前向傳播
input = torch.randn(128, 512, 448).cuda()
output = conv(input)  # [128, 1024, 448]
```

## 性能測試
```bash
python benchmark_txt.py
```

測試配置包含：
- 不同 kernel size (3, 5, 7, 9)
- 不同 stride (1, 2, 4, 8, 16)
- 不同 groups (1, 2, 4, 8, 16)
- 不同 channel 數量 (32→64, 64→128, 128→256, 256→512, 512→1024)

輸出範例：
```
------------------------- 性能/精度比較 結果 -------------------------
最大誤差: 0.000123    平均誤差: 0.000012    相對誤差: 0.000001

PyTorch 原始一維卷積層:
- 平均時間: 2.456 ms    GFLOPS: 125.43

Custom Conv (固定配置):
- 平均時間: 2.234 ms    GFLOPS: 137.89    速度提升: 9.04%
```

## CUDA 配置

- Threads per Block: 256
- Number of Blocks: 256
- Total Threads: 65,536
- 架構: sm_86 (可在 setup.py 中修改)

## 技術細節

### 共享記憶體使用
- 利用 shared memory 減少全局記憶體訪問
- 支援 groups convolution 的記憶體分配策略

## 客製化修改

修改 CUDA 架構:
```python
# setup.py
extra_compile_args={
    'nvcc': ['-arch=sm_XX']  # 改為你的 GPU 架構
}
```

### 修改測試配置參數
```python
# benchmark_txt.py
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
```

調整 block/thread 配置:
```cpp
// custom_conv_shared_cuda.cu
dim3 blocks(256);    // 修改 block 數量
dim3 threads(256);   // 修改每個 block 的 thread 數量
```

## 代辦事項 (TODO)

- [ ] 增加最基本的 Kernel Size 支援
- [ ] 使用 NVIDIA Nsight System 檢查每個 Shared Memory 使用率和整體效能使用率
