---
layout: post
title: Triton| .cubin文件调用研究
categories: [Triton]
description: .cubin文件的两种调用方式，适用于triton和cuda分别生成的.cubin文件
keywords: Triton, CUDA
mermaid: false
sequence: false
flow: false
mathjax: true
mindmap: true
mindmap2: false
---

本文记录了.cubin文件的两种调用方式，适用于triton和cuda分别生成的.cubin文件

## 思路一：直接调用缓存中保存的.cubin文件

##### CUDA调用

- .cubin文件与.ptx文件保存在~/.triton/cache下，调用方式如下：

```c
#include <iostream>
#include <cuda.h>
int main() {
    CUdevice device;
    CUcontext context;
    CUmodule module;
    CUfunction kernel;
    CUresult err;
    // 初始化CUDA驱动API
    err = cuInit(0);
    // 获取第一个设备
    err = cuDeviceGet(&device, 0);
    // 创建上下文
    err = cuCtxCreate(&context, 0, device);
    // 加载模块
    err = cuModuleLoad(&module, "matmul_cache_kernel.cubin");
    // 获取内核函数
    err = cuModuleGetFunction(&kernel, module, "matmul_cache_kernel_0d1d2d");
    // 定义输入和输出数据
    float h_A[512][512];float h_B[512][512];float h_C[512][512];
    float* d_A=nullptr; float* d_B=nullptr; float* d_C=nullptr;
    int M=512;int N=512;int K=512;
    // 初始化
    for (int i = 0; i < 512; i++) {
        for(int j = 0;j < 512; j++)
            h_A[i][j] = 0.1;       
    }
    for (int i = 0; i < 512; i++) {
        for(int j = 0;j < 512; j++)
            h_B[i][j] = 0.2;       
    }
    // 分配
    err = cuMemAlloc((CUdeviceptr*)&d_A, M*K * sizeof(float));
    err = cuMemAlloc((CUdeviceptr*)&d_B, K*N * sizeof(float));
    err = cuMemAlloc((CUdeviceptr*)&d_C, M*N * sizeof(float));
    // 复制
    err = cuMemcpyHtoD((CUdeviceptr)d_A, h_A, M*K * sizeof(float));
    err = cuMemcpyHtoD((CUdeviceptr)d_B, h_B, K*N * sizeof(float));
    void *args[] = {(d_A), (d_B), (d_C)};
    void *d_args=nullptr;
    err = cuMemAlloc((CUdeviceptr*)&d_args, 3* sizeof(void*));
    err = cuMemcpyHtoD((CUdeviceptr)d_args, args,3* sizeof(void*));
    // 设置内核参数
    CUstream hstream;
    err = cuStreamCreate(&hstream, CU_STREAM_DEFAULT);  
    // 启动
    dim3 gridDim(1, 1, 1);
    dim3 blockDim(256, 1, 1);
    unsigned int  sharedMemBytes=49152;
    err = cuLaunchKernel(kernel,
                         gridDim.x, gridDim.y, gridDim.z,
                         blockDim.x, blockDim.y, blockDim.z,
                         sharedMemBytes, // 共享内存字节数
                         hstream, // 流
                         &d_args, // 参数
                         NULL); // 额外参数
    // 复制
    err = cuMemcpyDtoH(h_C, (CUdeviceptr)d_C, M*N * sizeof(float));
    // 打印
    for (int i = 0; i < 10; i++) { 
        std::cout << "C[" << i <<","<<i<<" ] = " << h_C[i][i] << std::endl;
    }
    // 释放
    err = cuMemFree((CUdeviceptr)d_A);
    err = cuMemFree((CUdeviceptr)d_B);
    err = cuMemFree((CUdeviceptr)d_C);
    err = cuCtxDestroy(context);
    return 0;
}
```

- 附注：Curesults检查模板

```c
#define checkCudaErrors(err) {\
    if(err != CUDA_SUCCESS) {\
    const char *errorName;\
    const char *errorString;\
    cuGetErrorName(err, &errorName);\
    cuGetErrorString(err, &errorString);\
    std::cerr << "CUDA Error: " << errorName << ": " << errorString << std::endl;\
    exit(EXIT_FAILURE);}\
}
//检查：
    checkCudaErrors(err);
```

##### CPM_KERNELS调用

```python
import ctypes
import os
from typing import List, Any, Tuple
from cpm_kernels.library import cuda, cudart
from cpm_kernels.device import Device
from cpm_kernels.base import Kernel,Kernelfunc
import pkg_resources
DevicePointer = int
CUDAStream = cudart.cudaStream_t

RESOURCE_PACKAGE_NAME = __name__

import torch
import numpy as np
arg1 = torch.randn((512,512), device='cuda', dtype=torch.float16)
arg2 = torch.randn((512,512), device='cuda', dtype=torch.float16)
arg3 = torch.empty((512,512), device='cuda', dtype=torch.float32)

# arg1 = torch.randn((512,512), device='cpu', dtype=torch.float16)
# arg2 = torch.randn((512,512), device='cpu', dtype=torch.float16)
# arg3 = torch.empty((512,512), device='cpu', dtype=torch.float32)

# arg1 = np.random.randn(512, 512).astype(np.float16)
# arg2 = np.random.randn(512, 512).astype(np.float16)
# arg3 = np.empty((512, 512), dtype=np.float32)

kernelfunc = Kernel(filename='matmul_cache_kernel',function_names=['matmul_cache_kernel_0d1d2d'])
kernelfunc.matmul_cache_kernel_0d1d2d(gridDim=(8, 8, 1),blockDim=(32, 32, 1),sharedMemBytes=16384,stream=0,params=[arg1, arg2, arg3])
```

- 以上两种调用方式本质上一致，都是输入params给CUDA内置的API：CuLaunchKernel来启动
- 遇到的问题：以上方法对CUDA自己编译生成的.cubin文件非常有效但对于Triton编译出的.cubin仍会报错
  - 我在使用简化参数和代码结构后仍无法解决

![](/images/triton/1.png)

![](/images/triton/2.png)

## 思路二：使用Triton自带的compile.py

##### Triton版本

```shell
pip install -U --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly
```

##### 以一个简单的Triton核为例:

```python
#(文件名:test.py)
import triton
import triton.language as tl
import triton.compiler as tc
# triton kernel
@triton.jit
def kernel(X, stride_xm,  #
           Z, stride_zn,  #
           BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    off_m = tl.arange(0, BLOCK_M)
    off_n = tl.arange(0, BLOCK_N)
    Xs = X + off_m[:, None] * stride_xm + off_n[None, :] * 1
    Zs = Z + off_m[:, None] * 1 + off_n[None, :] * stride_zn
    tl.store(Zs, tl.load(Xs))
```

- 找到triton/python/triton/tools文件夹下的几个文件：
  - compile.py:运行其可以得到kernel.c/kernel.h
  - compile.c/compile.h:模板文件
  - link.py:封装文件

```Shell
python compile.py --kernel-name kernel --signature "*fp32:16, i32:16, *fp32:16, i32, 256, 128" --out-name kernel --out-path $PWD/example $PWD/test.py --grid 1024,1,1
```

- 附编译选项：	

```python
parser = ArgumentParser(description=desc)
parser.add_argument("path",help="Path to Python source containing desired kernel in its scope. File will be executed.")
parser.add_argument("--kernel-name", "-n", type=str, default="", help="Name of the kernel to compile",required=True)
parser.add_argument("--num-warps", "-w", type=int, default=1, help="Number of warps to launch the kernel")
parser.add_argument("--num-stages", "-ns", type=int, default=3,help="Number of stages (meta-parameter of the kernel)")
parser.add_argument("--out-name", "-on", type=str, default=None, help="Out name for the compiled kernel")
parser.add_argument("--out-path", "-o", type=Path, default=None, help="Out filename")
parser.add_argument("--signature", "-s", type=str, help="Signature of the kernel", required=True)
parser.add_argument("--grid", "-g", type=str, help="Launch grid of the kernel", required=True)
```

- 封装：

```shell
python link.py ./*.h -o kernel_name
```

- 调用程序：

```c
#include <iostream>
#include <cuda.h>
extern "C" {
    #include "kernel.785f2513_0d1d2d3.h"
}

int main() {
    CUdevice device;
    CUcontext context;
    CUresult err;
    err = cuInit(0);
    err = cuDeviceGet(&device, 0);
    err = cuCtxCreate(&context, 0, device);
    float h_A[512][512];float h_C[512][512];
    float* d_A=nullptr; float* d_C=nullptr;
    int M=512;int N=512;int K=512;

    for (int i = 0; i < 512; i++){
        for(int j = 0;j < 512; j++)
            h_A[i][j] = 0.1;
  }
    err = cuMemAlloc((CUdeviceptr*)&d_A, M*K * sizeof(float));
    err = cuMemAlloc((CUdeviceptr*)&d_C, M*N * sizeof(float));
    err = cuMemcpyHtoD((CUdeviceptr)d_A, h_A, M*K * sizeof(float));
    CUstream hstream;
    err = cuStreamCreate(&hstream, CU_STREAM_DEFAULT);
    kernel_785f2513_0d1d2d3(hstream, (CUdeviceptr)d_A, (int32_t)512, (CUdeviceptr)d_C, (int32_t)512);
    err = cuMemcpyDtoH(h_C, (CUdeviceptr)d_C, M*N * sizeof(float));
    for (int i = 0; i < 10; i++) {
        std::cout << "C[" << i <<","<<i<<" ] = " << h_C[i][i] << std::endl;
    }
    err = cuMemFree((CUdeviceptr)d_A);
    err = cuMemFree((CUdeviceptr)d_C);
    err = cuCtxDestroy(context);
    return 0;
}
```

- 编译

```shell
nvcc -std=c++17 -I $PWD test.cu kernel_name.c  kernel.785f2513_0d1d2d3.c --gpu-architecture=sm_89 -o test -lcuda

#也可以编译成so（可选）
nvcc -shared -o libexam.so kernel.785f2513_0d1d2d3.c -Xcompiler -fPIC
#编译链接so的测试程序（可选）
gcc -std=c++17 -I -L --gpu-architecture=sm_89 -lcuda -lexam test.cu -o test
```

- 测试结果表明：这样可以成功调用，这样的方法将cubin作为数组存了起来：
  -  ELF 二进制文件的十六进制表示：它包含了一个 CUDA 内核编译后二进制代码

![](/images/triton/3.png)

##### 封装矩阵乘法

- 矩阵乘法的Triton核：

```python
@triton.jit
def matmul_cache_kernel(
        a_ptr, b_ptr, c_ptr,
        M,N,K,
        stride_am,stride_ak,
        stride_bk,stride_bn,
        stride_cm,stride_cn,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
):
```

- 编译：

```shell
python3 compile.py --kernel-name matmul_cache_kernel --signature "*fp32:16, *fp32:16, *fp32:16,i32,i32,i32,i32,i32,i32,i32,i32,i32,32,32,32,4" --out-name kernel --out-path $PWD/example/kernel $PWD/cache.py --grid 1024,1,1
python link.py ./*.h -o kernel_name
```

```shell
nvcc -std=c++17 -I $PWD test.cu kernel_name.c  kernel.da01dcef_0d1d2d34567891011.c --gpu-architecture=sm_89 -o test -lcuda
```

##### 实现自动化

-  Kernel sig_hash的计算：

```python
signature = ['*fp32:16', '*fp32:16', '*fp32:16', 'i32', 'i32', 'i32', 'i32', 'i32', 'i32', 'i32', 'i32', 'i32']
signature.append(f"{numbers[5]}")
signature.append(f"{numbers[6]}")
signature.append(f"{numbers[7]}")
signature.append(f"{numbers[8]}")

args = type("Args", (), {"num_warps": numbers[9], "num_stages": numbers[11]})()
meta_sig = f"warps{args.num_warps}xstages{args.num_stages}"
def hash_signature(signature: List[str]):
    m = hashlib.sha256()
    m.update(" ".join(signature).encode())
    return m.hexdigest()[:8]
sig_hash = hash_signature(signature + [meta_sig])
kernel_name="kernel_"+sig_hash+"_0d1d2d34567891011"
numbers.append(kernel_name)
```

![](/images/triton/4.png)

- 执行config计算后，我们将其最有配置对应的hash值也一起存下来，当调用的时候可以直接查询得到其需要的函数"kernel_xxxxxxxx_0d1d2d34567891011"调用即可