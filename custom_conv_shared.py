import torch
import torch.nn as nn
import custom_conv_shared_cuda

class CustomConvShared(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1):
        super(CustomConvShared, self).__init__()
        
        # 基本參數設置
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size  # 一維卷積，不需要元組
        self.stride = stride            # 一維卷積，不需要元組
        self.padding = padding          # 一維卷積，不需要元組
        self.groups = groups
        
        # 初始化權重和偏置 - 一維卷積
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, self.kernel_size)
        )
        self.bias = nn.Parameter(torch.zeros(out_channels))
        
        # 使用 PyTorch 默認的初始化方法
        nn.init.kaiming_uniform_(self.weight, a=1.0)
        nn.init.uniform_(self.bias, -0.1, 0.1)
    
    def forward(self, input):
        # 計算輸出尺寸 - 一維卷積
        batch_size, _, length = input.shape
        out_length = (length - self.kernel_size + 2 * self.padding) // self.stride + 1
        
        # 使用共享記憶體 CUDA 核心進行一維卷積運算
        output = custom_conv_shared_cuda.forward(
            input, 
            self.weight, 
            self.stride,
            self.padding,
            self.groups
        )[0]
        
        # 添加 bias
        if self.bias is not None:
            output += self.bias.view(1, -1, 1)
        
        return output