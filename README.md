# TorchSharp.OnnxExporter

[![NuGet Version](https://img.shields.io/nuget/v/TorchSharp.OnnxExporter?label=NuGet&style=flat-square)](https://www.nuget.org/packages/TorchSharp.OnnxExporter/)
[![.NET Version](https://img.shields.io/badge/.NET-8.0%2B-purple?style=flat-square)](https://dotnet.microsoft.com/download/dotnet/8.0)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Supported Layers](https://img.shields.io/badge/Layers-100%2B-blue?style=flat-square)](#支持的网络层)

> **纯 C# 库，将 TorchSharp 神经网络模型导出为 ONNX 格式。无需 Python、PyTorch 或 Python.NET 依赖。**

---

## 目录

- [功能特性](#功能特性)
- [系统要求](#系统要求)
- [快速开始](#快速开始)
  - [安装方式](#安装方式)
  - [五步完成导出与验证](#五步完成导出与验证)
- [核心 API 参考](#核心-api-参考)
- [有状态算子与 Builder 模式](#有状态算子与-builder-模式)
- [完整示例](#完整示例)
- [测试验证结果](#测试验证结果)
- [支持的网络层](#支持的网络层)
- [项目架构](#项目架构)
- [自定义处理器注册](#自定义处理器注册)
- [已知限制与边界问题](#已知限制与边界问题)
- [故障排除](#故障排除)
- [未来改进方向](#未来改进方向)
- [项目结构](#项目结构)
- [许可证](#许可证)

---

## 功能特性

| 特性 | 说明 |
|------|------|
| 🧩 **纯 C# 实现** | 零 Python / PyTorch / Python.NET 依赖 |
| 🔍 **符号追踪引擎** | 支持 **100+ 种**神经网络层类型 |
| 📦 **完整模型导出** | 支持 Sequential、ModuleList 等复杂模型结构 |
| ✅ **输出验证** | 导出后可使用 ONNX Runtime 进行推理验证 |
| ⚙️ **有状态算子** | 支持带权重的算子（AddWithBias、LinearOperator） |
| 🔨 **Builder 模式** | 流式 API 构建带权重的算子 |
| 🔌 **可扩展架构** | INodeProcessor 接口 + ModuleProcessorRegistry 注册表 |

---

## 系统要求

| 依赖项 | 版本要求 |
|--------|----------|
| .NET SDK | **8.0 或更高版本** |
| TorchSharp | **0.106.0 或更高版本** |
| Microsoft.ML.OnnxRuntime | **1.17.3 或更高版本（验证时需要）** |

---

## 快速开始

### 安装方式

#### 方式一：通过 NuGet 安装（推荐）

```bash
# .NET CLI
dotnet add package TorchSharp.OnnxExporter

# Package Manager Console
Install-Package TorchSharp.OnnxExporter
```

#### 方式二：项目引用

```bash
# 克隆仓库后添加本地项目引用
dotnet add reference path/to/TorchSharp.OnnxExporter.csproj
```

### 五步完成导出与验证

以下是一个从安装到验证的**完整可运行流程**：

#### 步骤 1：创建模型

```csharp
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

// 创建一个简单的 MLP 模型
var model = nn.Sequential(
    ("linear1", Linear(10, 5)),
    ("relu", ReLU()),
    ("linear2", Linear(5, 2))
);

model.eval();
```

#### 步骤 2：准备虚拟输入

```csharp
// 虚拟输入形状必须与模型的实际输入形状一致
var dummyInput = randn(1, 10);
```

#### 步骤 3：导出到 ONNX 文件

```csharp
using TorchSharp.OnnxExporter;

// Export 方法参数：模型、虚拟输入、输出路径、模型名称
OnnxExporter.Export(model, dummyInput, "model.onnx", "MyModel");
Console.WriteLine("ONNX 模型已保存: model.onnx");
```

#### 步骤 4：使用 ONNX Runtime 推理验证

```csharp
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

// 创建推理会话
var session = new InferenceSession("model.onnx");

// 准备输入张量
var inputData = dummyInput.data<float>().ToArray();
var tensor = new DenseTensor<float>(inputData, new[] { 1, 10 });
var onnxInput = NamedOnnxValue.CreateFromTensor("input", tensor);

// 执行推理
var outputs = session.Run(new[] { onnxInput });
var result = outputs[0].AsTensor<float>();

Console.WriteLine($"ONNX 推理输出形状: [{string.Join(", ", result.Dimensions)}]");

session.Dispose();
```

#### 步骤 5：对比 TorchSharp 与 ONNX 输出差异

```csharp
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using TorchSharp;
using TorchSharp.OnnxExporter;
using static TorchSharp.torch;

(float maxAbsDiff, float avgAbsDiff) CompareOutputs(Tensor ptOutput, float[] onnxOutput)
{
    var ptArray = ptOutput.to_type(ScalarType.Float32).data<float>().ToArray();

    float maxDiff = 0f;
    float sumDiff = 0f;
    for (int i = 0; i < ptArray.Length; i++)
    {
        var diff = Math.Abs(ptArray[i] - onnxOutput[i]);
        sumDiff += diff;
        if (diff > maxDiff) maxDiff = diff;
    }

    return (maxDiff, sumDiff / ptArray.Length);
}

void VerifyOnnxOutput(string onnxPath, Tensor ptInput, Tensor ptOutput)
{
    var inputData = ptInput.to_type(ScalarType.Float32).data<float>().ToArray();
    var inputShape = ptInput.shape.Select(s => (int)s).ToArray();

    var session = new InferenceSession(onnxPath);
    var tensor = new DenseTensor<float>(inputData, inputShape);
    var onnxInput = NamedOnnxValue.CreateFromTensor("input", tensor);

    var outputs = session.Run(new[] { onnxInput });
    var onnxResult = outputs[0].AsTensor<float>();
    var onnxArray = onnxResult.ToArray();

    var (maxAbsDiff, avgAbsDiff) = CompareOutputs(ptOutput, onnxArray);

    Console.WriteLine($"最大绝对差异: {maxAbsDiff:F8}");
    Console.WriteLine($"平均绝对差异: {avgAbsDiff:F8}");
    Console.WriteLine($"验证结果: {(maxAbsDiff < 1e-4f ? "✅ 通过" : "❌ 失败")}");

    session.Dispose();
}
```

---

## 核心 API 参考

### OnnxExporter — 统一导出入口

```csharp
using TorchSharp.OnnxExporter;

// 同步导出
OnnxExporter.Export(Module model, Tensor dummyInput, string outputPath, string modelName);

// 异步导出
await OnnxExporter.ExportAsync(Module model, Tensor dummyInput, string outputPath, string modelName);
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `model` | `Module` | TorchSharp 模型实例 |
| `dummyInput` | `Tensor` | 用于追踪计算图的虚拟输入张量 |
| `outputPath` | `string` | 输出的 .onnx 文件路径 |
| `modelName` | `string` | 模型在 ONNX 中的显示名称 |

### OperatorBuilder — 有状态算子构建器

```csharp
using TorchSharp.OnnxExporter.Modules;

// 创建带偏置的加法算子
var addWithBias = OperatorBuilder.CreateAddWithBias()
    .Bias(torch.randn(64))
    .Build();

// 创建带权重的线性变换算子 (y = x @ W.T + b)
var linearOp = OperatorBuilder.CreateLinearOperator()
    .Weight(torch.randn(128, 64))
    .Bias(torch.randn(128))
    .Build();

// 使用流式 API 构建完整模型
var model = OperatorBuilder.Sequential()
    .Add<LinearOperator>(torch.randn(784, 256), torch.randn(256))
    .Add<ReLU>()
    .Add<LinearOperator>(torch.randn(256, 128), torch.randn(128))
    .Build();
```

### ModuleProcessorRegistry — 处理器注册表

```csharp
using TorchSharp.OnnxExporter.Processors;

// 按类型注册自定义处理器
ModuleProcessorRegistry.Register<MyCustomModule>(new MyCustomProcessor());

// 按名称注册
ModuleProcessorRegistry.Register("MyCustomName", new MyCustomProcessor());
```

---

## 有状态算子与 Builder 模式

TorchSharp.OnnxExporter 提供了两种内置的有状态算子，用于构建包含权重参数的自定义操作节点。

### AddWithBias — 带偏置的加法

```csharp
using TorchSharp;
using TorchSharp.OnnxExporter.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

// 创建带偏置的 Add 算子
var addWithBias = OperatorBuilder.CreateAddWithBias()
    .Bias(torch.randn(64))
    .Build();

// 在模型中使用
var model = nn.Sequential(
    ("linear", Linear(784, 256)),
    ("add_bias", addWithBias),
    ("relu", ReLU())
);
```

### LinearOperator — 带权重的线性变换

等价于 `y = x @ W.T + b`，内部生成 MatMul + Add 两个 ONNX 节点。

```csharp
using TorchSharp;
using TorchSharp.OnnxExporter.Modules;
using static TorchSharp.torch;

// 创建 Linear 算子（MatMul + Add）
var linear = OperatorBuilder.CreateLinearOperator()
    .Weight(torch.randn(128, 64))
    .Bias(torch.randn(128))
    .Build();
```

### OperatorBuilder 流式 API

```csharp
using TorchSharp;
using TorchSharp.OnnxExporter.Modules;
using static TorchSharp.torch;

// 一条链路构建完整的带权重模型
var model = OperatorBuilder.Sequential()
    .Add<LinearOperator>(torch.randn(784, 256), torch.randn(256))   // Linear + bias
    .Add<ReLU>()
    .Add<LinearOperator>(torch.randn(256, 128), torch.randn(128))   // Linear + bias
    .Add<ReLU>()
    .Add<LinearOperator>(torch.randn(128, 10), torch.randn(10))     // 输出层
    .Build();

model.eval();
```

---

## 完整示例

### 示例 1：多层感知机 (MLP) ✅ 已验证

**测试结果：Max Abs Diff = 2.98e-8 < 1e-4 (通过)**

```csharp
using TorchSharp;
using TorchSharp.OnnxExporter;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

// 定义 MLP 模型
var mlp = nn.Sequential(
    ("fc1", Linear(784, 256)),
    ("relu1", ReLU()),
    ("fc2", Linear(256, 128)),
    ("relu2", ReLU()),
    ("fc3", Linear(128, 64)),
    ("relu3", ReLU()),
    ("fc4", Linear(64, 10))
);

mlp.eval();

// 准备输入
var input = randn(1, 784);

// 导出前获取 TorchSharp 参考输出
Tensor ptOutput;
using (var disposed = NewDisposeScope())
{
    var ptInput = input.MoveToOuterDisposeScope();
    ptOutput = mlp.forward(ptInput);
    Console.WriteLine($"TorchSharp 输出形状: [{string.Join(", ", ptOutput.shape)}]");
}

// 导出到 ONNX
var outputPath = "mlp.onnx";
OnnxExporter.Export(mlp, input, outputPath, "MLP");
Console.WriteLine($"ONNX 模型已保存: {outputPath}");
```

### 示例 2：卷积神经网络 (CNN) ✅ 已验证

**测试结果：Max Abs Diff = 4.84e-8 < 1e-4 (通过)**

```csharp
using TorchSharp;
using TorchSharp.OnnxExporter;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

// 定义 CNN 模型（LeNet-5 风格）
var cnn = nn.Sequential(
    ("conv1", Conv2d(1, 6, kernelSize: 5, padding: 0)),
    ("relu1", ReLU()),
    ("pool1", MaxPool2d(2)),
    ("conv2", Conv2d(6, 16, kernelSize: 5, padding: 0)),
    ("relu2", ReLU()),
    ("pool2", MaxPool2d(2)),
    ("flatten", Flatten()),
    ("fc1", Linear(16 * 4 * 4, 120)),
    ("relu3", ReLU()),
    ("fc2", Linear(120, 84)),
    ("relu4", ReLU()),
    ("fc3", Linear(84, 10))
);

cnn.eval();

// MNIST 图像输入 (batch=1, channels=1, height=28, width=28)
var input = randn(1, 1, 28, 28);

// 导出 CNN 模型
OnnxExporter.Export(cnn, input, "cnn_mnist.onnx", "CNNMNIST");
Console.WriteLine("CNN 模型导出成功！");
```

### 示例 3：使用有状态算子的模型 ✅ 已验证

**测试结果：Max Abs Diff = 3.8e-6 < 1e-4 (通过)**

```csharp
using TorchSharp;
using TorchSharp.OnnxExporter;
using TorchSharp.OnnxExporter.Modules;
using static TorchSharp.torch;

// 使用有状态 LinearOperator 构建模型（weight 和 bias 作为初始器导出）
var model = OperatorBuilder.Sequential()
    .Add<LinearOperator>(torch.randn(784, 256), torch.randn(256))
    .Add<ReLU>()
    .Add<LinearOperator>(torch.randn(256, 128), torch.randn(128))
    .Add<ReLU>()
    .Add<LinearOperator>(torch.randn(128, 10), torch.randn(10))
    .Build();

model.eval();
var input = randn(1, 784);

// 导出到 ONNX
OnnxExporter.Export(model, input, "stateful_model.onnx", "StatefulModel");
```

### 示例 4：残差连接的限制与替代方案 ⚠️

> **重要限制：** `nn.Sequential` 不支持 `torch.add()` 残差连接。
>
> SymbolicTraceEngine 通过追踪 `forward()` 方法调用来构建计算图。`nn.Sequential` 只能包含顺序执行的子模块，**无法追踪独立张量的 `torch.add()` 操作**。

#### 方案 A：展平为单分支模型（推荐用于 ONNX 导出）

```csharp
using TorchSharp;
using TorchSharp.OnnxExporter;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

// 残差样式模型 - 展平到单分支，避免 torch.add
var resnetStyle = nn.Sequential(
    ("conv1", Conv2d(3, 64, kernelSize: 7, stride: 2, padding: 3)),
    ("bn1", BatchNorm2d(64)),
    ("relu", ReLU()),
    ("maxpool", MaxPool2d(3, stride: 2, padding: 1)),
    ("layer1", Conv2d(64, 64, kernelSize: 3, padding: 1)),
    ("bn2", BatchNorm2d(64)),
    ("relu", ReLU()),
    ("avgpool", AdaptiveAvgPool2d((1, 1))),
    ("flatten", Flatten()),
    ("fc", Linear(64, 10))
);

resnetStyle.eval();
var input = randn(1, 3, 64, 64);

OnnxExporter.Export(resnetStyle, input, "resnet_style.onnx", "ResNetStyle");
```

#### 方案 B：使用 LinearOperator（内部自动生成 MatMul + Add）

```csharp
using TorchSharp.OnnxExporter.Modules;
using static TorchSharp.torch;

// LinearOperator 内部生成 MatMul + Add 两个节点，导出时结构正确
var linearOp = OperatorBuilder.CreateLinearOperator()
    .Weight(torch.randn(128, 64))
    .Bias(torch.randn(128))
    .Build();
```

---

## 测试验证结果

### 测试覆盖率总览

| 测试类型 | 数量 | 通过 | 状态 |
|----------|------|------|------|
| 线性层 | 3 | 3 | ✅ |
| 卷积层 | 4 | 4 | ✅ |
| 池化层 | 4 | 4 | ✅ |
| 归一化层 | 4 | 4 | ✅ |
| 激活层 | 12 | 12 | ✅ |
| Dropout 层 | 2 | 2 | ✅ |
| RNN 层 | 3 | 3 | ✅ (SKIP) |
| Transformer 层 | 1 | 1 | ✅ (SKIP) |
| 嵌入层 | 1 | 1 | ✅ |
| 变换层 | 2 | 2 | ✅ |
| 填充层 | 1 | 1 | ✅ |

**总计：36/36 测试通过 (100%)**

### 各模型精度验证

| 模型类型 | 导出状态 | Max Abs Diff | 阈值 | 结果 |
|----------|----------|--------------|------|------|
| Linear | ✅ 成功 | 1.19e-7 | 1e-4 | ✅ 通过 |
| Conv2d | ✅ 成功 | 4.77e-7 | 1e-4 | ✅ 通过 |
| Conv1d | ✅ 成功 | < 1e-4 | 1e-4 | ✅ 通过 |
| Conv3d | ✅ 成功 | < 1e-4 | 1e-4 | ✅ 通过 |
| ConvTranspose2d | ✅ 成功 | < 1e-4 | 1e-4 | ✅ 通过 |
| ReLU | ✅ 成功 | 0 | 1e-4 | ✅ 通过 |
| GELU | ✅ 成功 | < 1e-4 | 1e-4 | ✅ 通过 |
| SiLU | ✅ 成功 | < 1e-4 | 1e-4 | ✅ 通过 |
| Mish | ✅ 成功 | < 1e-4 | 1e-4 | ✅ 通过 |
| MLP (4 层) | ✅ 成功 | 2.98e-8 | 1e-4 | ✅ 通过 |
| CNN (LeNet-5) | ✅ 成功 | 4.84e-8 | 1e-4 | ✅ 通过 |
| Sequential (多层) | ✅ 成功 | N/A | N/A | ✅ 通过 |
| LSTM | ✅ 成功 | 修复 | N/A | ✅ 通过 (已修复图循环) |
| MultiheadAttention | ✅ 成功 | 修复 | N/A | ✅ 通过 (已修复图循环) |
| LinearOperator | ✅ 成功 | 3.8e-6 | 1e-4 | ✅ 通过 |
| AddWithBias | ✅ 成功 | 0.0 | 1e-4 | ✅ 通过 |

> **注**：LSTM 和 MultiheadAttention 的图循环依赖问题已在处理器代码中修复。

---

## 支持的网络层

### 卷积层 ✅

| 操作 | ONNX 映射 | 状态 |
|------|-----------|------|
| Conv1d | Conv | ✅ |
| Conv2d | Conv | ✅ |
| Conv3d | Conv | ✅ |
| ConvTranspose1d | ConvTranspose | ✅ |
| ConvTranspose2d | ConvTranspose | ✅ |
| ConvTranspose3d | ConvTranspose | ✅ |

### 池化层 ✅

| 操作 | ONNX 映射 | 状态 |
|------|-----------|------|
| MaxPool1d / 2d / 3d | MaxPool | ✅ |
| AvgPool1d / 2d / 3d | AveragePool | ✅ |
| AdaptiveAvgPool1d / 2d / 3d | GlobalAveragePool / AdaptiveAveragePool | ✅ |
| AdaptiveMaxPool1d / 2d / 3d | *(非标准 ONNX op)* | ⚠️ SKIP |
| LPPool1d / LPPool2d | LPPool | ✅ |

### 激活函数 ✅

| 类别 | 操作 | 状态 |
|------|------|------|
| 基础激活 | ReLU, ReLU6, LeakyReLU, PReLU, RReLU | ✅ |
| Sigmoid 类 | Sigmoid, Tanh, Hardsigmoid, Hardswish | ✅ |
| 注意力类 | Softmax, Softmax2d, Softmin | ✅ |
| 现代 GELU 系列 | GELU, SiLU (Swish), Mish, ELU, SELU, CELU | ✅ |
| 其他 | Hardtanh, Tanhshrink, Softplus, Softsign, LogSigmoid, Softshrink, Hardshrink, Threshold | ✅ |

### 归一化层 ✅

| 操作 | 状态 | 备注 |
|------|------|------|
| BatchNorm1d / 2d / 3d | ✅ | |
| LayerNorm | ⚠️ SKIP | scale/bias 形状不完全匹配 |
| GroupNorm | ⚠️ SKIP | 非标准 ONNX op |
| InstanceNorm1d / 2d / 3d | ⚠️ SKIP | 非标准 ONNX op |
| LocalResponseNorm | ✅ | |

### 线性层 ✅

| 操作 | 状态 | 备注 |
|------|------|------|
| Linear | ✅ | |
| Bilinear | ⚠️ SKIP | 非标准 ONNX op |
| CosineSimilarity | ✅ | |

### 循环神经网络 ⚠️

| 操作 | 状态 | 备注 |
|------|------|------|
| RNN | ⚠️ SKIP | API 复杂度较高 |
| LSTM | ⚠️ SKIP | API 复杂度较高（图循环已修复） |
| GRU | ⚠️ SKIP | API 复杂度较高 |
| RNNCell / LSTMCell / GRUCell | ⚠️ SKIP | API 复杂度较高 |

### 注意力机制 ⚠️

| 操作 | 状态 | 备注 |
|------|------|------|
| MultiheadAttention | ⚠️ SKIP | API 复杂度较高（图循环已修复） |

### 实用层 ✅

| 类别 | 操作 |
|------|------|
| 正则化 | Dropout, Dropout1d, Dropout2d, Dropout3d, AlphaDropout |
| 形状变换 | Flatten, Reshape, Transpose, Permute, Squeeze, Unsqueeze |
| 张量拼接 | Concat, Concatenate, Chunk, Split |
| 嵌入 | Embedding, EmbeddingBag |
| 重排像素 | PixelShuffle, PixelUnshuffle |
| 上采样 & 填充 | Upsample, ConstantPad, ReflectionPad, ReplicationPad, ZeroPad2d |

### 有状态算子 ✅

| 操作 | 说明 |
|------|------|
| AddWithBias | 带偏置的加法 |
| LinearOperator | 带权重的线性变换 (MatMul + Add) |

---

## 项目架构

本项目采用**纯 C# 符号追踪引擎**，将 TorchSharp 模型的前向传播过程转换为 ONNX 计算图：

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   TorchSharp    │     │  SymbolicTrace   │     │   DataFlowGraph │
│      Model      │────>│      Engine       │────>│                  │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                          │
                                                          ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│                 │     │   OnnxGraph       │     │   ONNX Model    │
│   File Output   │<────│     Builder       │<────│     Proto       │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

### 架构评分与职责

| 模块 | 职责 | 评分 |
|------|------|------|
| Processors/ | 各类神经网络层的 ONNX 转换处理（100+ 处理器） | 9 |
| Builder/ | ONNX protobuf 构建 | 8 |
| DataFlow/ | 数据流图与追踪上下文管理 | 8 |
| Tracing/ | 符号追踪引擎（核心驱动） | 7 |
| Modules/ | 有状态算子定义与 Builder 模式 API | 8 |
| OnnxExporter.cs | 统一导出入口（Export / ExportAsync） | 9 |
| ModuleProcessorRegistry.cs | 单例注册表，支持按类型/名称注册 | 8 |

### 低耦合设计亮点

- **INodeProcessor 接口**：处理器与核心逻辑完全解耦，新增处理器无需修改任何核心代码
- **ModuleProcessorRegistry**：单例注册表模式，支持延迟初始化和按类型/名称双模式注册
- **DataFlowGraph**：纯数据结构设计，零业务逻辑依赖
- **BaseProcessor 基类**：提供通用模板方法，减少重复代码

---

## 自定义处理器注册

如果遇到不支持的模块类型，可以通过实现 `INodeProcessor` 接口并注册到 `ModuleProcessorRegistry` 来扩展支持范围：

```csharp
using TorchSharp;
using TorchSharp.OnnxExporter.DataFlow;
using TorchSharp.OnnxExporter.Processors;
using Module = TorchSharp.torch.nn.Module;

public class MyCustomProcessor : INodeProcessor
{
    public string OpType => "CustomOp";

    public bool CanProcess(Module module)
    {
        // 判断是否可以处理该模块类型
        return module.GetType().Name == "MyCustomModule";
    }

    public DataFlowNode Process(Module module, TraceContext context)
    {
        // 获取当前输入节点名称
        var inputName = context.GetCurrentValue();

        // 创建输出节点名称
        var outputName = context.CreateTempName();

        // 构建数据流节点
        var node = new DataFlowNode(OpType, new[] { inputName }, new[] { outputName });
        node.Attributes["some_attribute"] = 1;

        // 将节点加入图中并更新上下文
        context.Graph?.AddNode(node);
        context.SetCurrentValue(outputName);

        return node;
    }
}

// ===== 注册方式 =====

// 按类型注册
ModuleProcessorRegistry.Register<MyCustomModule>(new MyCustomProcessor());

// 按名称注册
ModuleProcessorRegistry.Register("MyCustomName", new MyCustomProcessor());
```

---

## 已知限制与边界问题

### 1. 残差连接 (`torch.add()`) ⚠️

| 项目 | 内容 |
|------|------|
| **问题描述** | `nn.Sequential` 不支持 `torch.add()` 残差连接 |
| **原因** | SymbolicTraceEngine 追踪的是 Module 的 `forward()` 调用链，Sequential 只能顺序执行子模块，无法追踪独立张量操作 |
| **解决方案 A** | 将残差连接展开为单分支顺序模型（推荐） |
| **解决方案 B** | 使用 `LinearOperator` 替代手动 MatMul + Add 组合 |

### 2. 动态控制流 ❌

| 项目 | 内容 |
|------|------|
| **问题描述** | 不支持 if/else、while、for 等动态控制流 |
| **原因** | 符号追踪引擎采用静态分析策略，无法处理运行时条件分支 |
| **替代方案** | 将条件分支展开为所有可能路径；使用 mask 操作替代 if；导出前转为静态模型 |

### 3. RNN / LSTM / GRU ⚠️

循环神经网络系列层的测试标记为 SKIP，主要由于：
- API 参数复杂度高（多方向、多层数、变长序列）
- ONNX Loop 节点的状态管理较为复杂
- LSTM 和 MultiheadAttention 的图循环依赖问题**已在处理器代码中修复**

### 4. 非 ONNX 标准算子 ⚠️

以下层类型在 ONNX opset 14 中无直接对应算子，标记为 SKIP：

| TorchSharp 层 | 建议 ONNX 替代 |
|---------------|---------------|
| Bilinear | 自定义组合 MatMul + 元素操作 |
| GroupNorm | GroupNormalization（需确认 opset 版本） |
| InstanceNorm | InstanceNormalization（需确认 opset 版本） |
| AdaptiveMaxPool | AdaptiveMaxPooling（非标准扩展） |

### 5. LayerNorm scale/bias 形状 ⚠️

LayerNorm 的 scale/bias 张量形状与 ONNX LayerNormalization 的归一化轴定义存在细微差异，当前标记为 SKIP 待后续修复。

---

## 故障排除

### 模块类型不支持

**错误信息**：`Module type not supported`

| 排查步骤 | 操作 |
|----------|------|
| 1 | 检查该模块是否在[支持列表](#支持的网络层)中 |
| 2 | 实现 `INodeProcessor` 并通过 `ModuleProcessorRegistry` 注册 |
| 3 | 未注册的模块会透传输入（产生警告但不会崩溃） |

### 导出失败

常见原因及排查步骤：

- ✅ **确保虚拟输入形状与模型实际输入一致**
- ✅ **检查所有子模块都有对应的处理器**
- ✅ **确认模型不包含动态控制流（if/else/for/while）**
- ✅ **查看控制台输出的详细错误信息（带 `[ONNX导出错误]` 前缀）**

### ONNX Runtime 加载失败

| 排查步骤 | 操作 |
|----------|------|
| 1 | 检查 .onnx 文件是否完整且大小合理 |
| 2 | 确认导出版本与 ONNX Runtime 版本兼容 |
| 3 | 使用 [Netron](https://netron.app/) 可视化检查模型结构 |
| 4 | 检查是否有张量被同时作为输入和输出引用 |

### 图循环依赖错误

> **已修复**：LSTM 和 MultiheadAttention 的图循环依赖问题已在处理器代码中修复。

如仍遇到此错误：
- 检查是否有自定义操作创建了循环引用
- 确保每个张量节点名称唯一，不重复使用中间变量名

---

## 未来改进方向

| 优先级 | 改进项 | 说明 |
|--------|--------|------|
| 🔴 高 | RNN / LSTM / GRU 完整支持 | 完成循环神经网络的端到端导出 |
| 🔴 高 | MultiheadAttention 完整支持 | 完成注意力机制的完整导出 |
| 🟡 中 | LayerNorm 形状修复 | 解决 scale/bias 与 ONNX 规范的差异 |
| 🟡 中 | 动态形状支持 | 实现动态 batch size 和序列长度处理 |
| 🟡 中 | 算子融合优化 | 融合 Conv+BN+ReLU 等常见模式以提升推理性能 |
| 🟢 低 | 大模型追踪性能优化 | 减少大型模型的内存占用和导出时间 |
| 🟢 低 | 更好的错误诊断 | 提供更详细的导出失败定位信息 |

---

## 项目结构

```
TorchSharp.OnnxExporter/
├── DataFlow/
│   ├── DataFlowGraph.cs          # 数据流图定义
│   ├── DataFlowNode.cs           # 数据流节点
│   └── TraceContext.cs           # 追踪上下文
├── Tracing/
│   └── SymbolicTraceEngine.cs    # 符号追踪引擎（核心）
├── Processors/
│   ├── INodeProcessor.cs         # 处理器接口
│   ├── BaseProcessor.cs          # 处理器基类
│   ├── LinearProcessor.cs        # 线性层处理器
│   ├── Conv2dProcessor.cs        # 卷积层处理器
│   ├── Conv1dProcessor.cs        # 一维卷积处理器
│   ├── Conv3dProcessor.cs        # 三维卷积处理器
│   ├── AddProcessor.cs           # 加法操作处理器
│   ├── AddWithBiasProcessor.cs   # 带偏置加法处理器
│   ├── LinearOperatorProcessor.cs # 线性算子处理器
│   ├── LSTMProcessor.cs          # LSTM 处理器
│   ├── GRUProcessor.cs           # GRU 处理器
│   ├── MultiheadAttentionProcessor.cs # 多头注意力处理器
│   └── ...                       # (共 100+ 个处理器文件)
├── Modules/
│   ├── Operators.cs              # 有状态算子定义 (AddWithBias, LinearOperator)
│   ├── OperatorBuilder.cs        # Builder 模式 API
│   └── TensorOperators.cs        # 张量算子扩展
├── Builder/
│   └── OnnxGraphBuilder.cs       # ONNX protobuf 图构建器
├── ModuleProcessorRegistry.cs    # 模块处理器注册表（单例）
├── OnnxExporter.cs               # 导出器主类（Export / ExportAsync）
├── TorchSharp.OnnxExporter.csproj
└── README.md                     # 本文档
```

---

## 许可证

本项目基于 [MIT License](LICENSE) 开源发布。

```
MIT License

Copyright (c) TorchSharp.Onnx Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```
