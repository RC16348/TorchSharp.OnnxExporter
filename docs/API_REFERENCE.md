# TorchSharp.OnnxExporter API 参考文档

> **版本**: 1.0.0 | **命名空间根**: `TorchSharp.OnnxExporter` | **ONNX IR 版本**: 7 | **Opset 版本**: 14

---

## 目录

- [1. OnnxExporter 类](#1-onnxexporter-类)
- [2. OperatorBuilder 类](#2-operatorbuilder-类)
- [3. ModuleProcessorRegistry 类](#3-moduleprocessorregistry-类)
- [4. INodeProcessor 接口与 BaseProcessor 抽象类](#4-inodeprocessor-接口与-baseprocessor-抽象类)
- [5. 有状态算子类（Operator 基类体系）](#5-有状态算子类operator-基类体系)
- [6. 张量操作算子类（TensorOperators）](#6-张量操作算子类tensoroperators)
- [7. 数据流相关类](#7-数据流相关类)
- [8. OnnxGraphBuilder 类](#8-onnxgraphbuilder-类)
- [9. SymbolicTraceEngine 类](#9-symbolictraceengine-类)

---

## 1. OnnxExporter 类

| 项目 | 说明 |
|------|------|
| **命名空间** | `TorchSharp.OnnxExporter` |
| **类型** | `static class` |
| **文件** | [OnnxExporter.cs](../OnnxExporter.cs) |
| **简要描述** | ONNX 导出器的统一入口，提供同步/异步两种方式将 TorchSharp 模型导出为 ONNX 格式文件。内部使用符号跟踪引擎 (`SymbolicTraceEngine`) 构建数据流图，再通过 `OnnxGraphBuilder` 生成 ONNX protobuf 并写入文件。 |

### 静态方法

#### Export(Module, Tensor, string, string)

```csharp
public static void Export(
    Module model,
    Tensor dummyInput,
    string outputPath,
    string modelName = "model"
)
```

**功能**: 同步导出模型到 ONNX 格式文件。

| 参数 | 类型 | 说明 |
|------|------|------|
| `model` | `Module` | 要导出的 TorchSharp 模型（不可为 null） |
| `dummyInput` | `Tensor` | 虚拟输入张量，用于形状推断和前向跟踪（不可为 null） |
| `outputPath` | `string` | 输出 ONNX 文件的完整路径（不可为空） |
| `modelName` | `string` | 模型名称，默认值为 `"model"` |

**异常**:

| 异常类型 | 触发条件 |
|----------|----------|
| `ArgumentNullException` | `model` 为 null、`dummyInput` 为 null、`outputPath` 为空或 null |
| `InvalidOperationException` | 符号跟踪失败、图构建失败、处理器注册失败 |
| 封装 `IOException` | 文件写入失败（路径不存在、权限不足等） |

**执行流程**:
1. 调用 `RegisterDefaultProcessors()` 注册全部内置处理器
2. 创建 `SymbolicTraceEngine` 并执行 `Trace()` 构建数据流图
3. 创建 `OnnxGraphBuilder` 并执行 `Build()` 生成 `ModelProto`
4. 使用 `CodedOutputStream` 将 protobuf 写入文件

---

#### ExportAsync(Module, Tensor, string, string)

```csharp
public static async Task ExportAsync(
    Module model,
    Tensor dummyInput,
    string outputPath,
    string modelName = "model"
)
```

**功能**: 异步导出模型到 ONNX 格式文件。内部逻辑与 `Export()` 一致，仅在最后写入文件时使用 `Task.Run()` 进行异步包装。

**参数**: 与 `Export()` 完全相同。

**异常**: 与 `Export()` 完全相同。

### 使用示例

```csharp
using TorchSharp;
using TorchSharp.OnnxExporter;
using static TorchSharp.torch.nn;

// 1. 定义模型
var model = Sequential(
    Linear(784, 256),
    ReLU(),
    Dropout(0.2),
    Linear(256, 10)
);

// 2. 创建虚拟输入
var dummyInput = torch.randn(1, 784);

// 3. 同步导出
OnnxExporter.Export(model, dummyInput, "my_model.onnx", "MNISTClassifier");

// 4. 或异步导出
await OnnxExporter.ExportAsync(model, dummyInput, "my_model_async.onnx", "MNISTClassifier");
```

---

## 2. OperatorBuilder 类

| 项目 | 说明 |
|------|------|
| **命名空间** | `TorchSharp.OnnxExporter.Modules` |
| **类型** | `static class` |
| **文件** | [OperatorBuilder.cs](../Modules/OperatorBuilder.cs) |
| **简要描述** | 提供建造者模式 (Builder Pattern) 的静态工厂方法，用于创建带参数的有状态算子模块（`AddWithBias`、`LinearOperator`）。 |

### 静态工厂方法

#### CreateAddWithBias()

```csharp
public static AddWithBiasBuilder CreateAddWithBias()
```

**返回值**: `AddWithBiasBuilder` — 一个建造者实例，链式配置后调用 `Build()` 生成 `AddWithBias` 算子。

#### CreateLinearOperator()

```csharp
public static LinearOperatorBuilder CreateLinearOperator()
```

**返回值**: `LinearOperatorBuilder` — 一个建造者实例，链式配置后调用 `Build()` 生成 `LinearOperator` 算子。

---

### AddWithBiasBuilder 内部类

位于 `OperatorBuilder` 类内部。

#### Bias(Tensor)

```csharp
public AddWithBiasBuilder Bias(Tensor bias)
```

**功能**: 设置偏置张量。**必须调用**，否则 `Build()` 将抛出异常。

| 参数 | 类型 | 说明 |
|------|------|------|
| `bias` | `Tensor` | 偏置张量 |

**返回值**: `AddWithBiasBuilder`（自身，支持链式调用）

#### Build()

```csharp
public AddWithBias Build()
```

**返回值**: `AddWithBias` — 配置完成的加偏置算子模块。

**异常**: `InvalidOperationException` — 当未调用 `Bias()` 设置偏置时抛出，消息为 `"Bias is required for AddWithBias"`。

---

### LinearOperatorBuilder 内部类

位于 `OperatorBuilder` 类内部。

#### Weight(Tensor)

```csharp
public LinearOperatorBuilder Weight(Tensor weight)
```

**功能**: 设置权重张量。**必须调用**，否则 `Build()` 将抛出异常。

| 参数 | 类型 | 说明 |
|------|------|------|
| `weight` | `Tensor` | 权重张量 |

**返回值**: `LinearOperatorBuilder`（自身，支持链式调用）

#### Bias(Tensor)

```csharp
public LinearOperatorBuilder Bias(Tensor bias)
```

**功能**: 设置可选的偏置张量。可不调用（此时等价于无偏置线性变换）。

| 参数 | 类型 | 说明 |
|------|------|------|
| `bias` | `Tensor` | 偏置张量（可选） |

**返回值**: `LinearOperatorBuilder`（自身，支持链式调用）

#### Build()

```csharp
public LinearOperator Build()
```

**返回值**: `LinearOperator` — 配置完成的线性算子模块。

**异常**: `InvalidOperationException` — 当未调用 `Weight()` 设置权重时抛出，消息为 `"Weight is required for LinearOperator"`。

### 使用示例

```csharp
using TorchSharp;
using TorchSharp.OnnxExporter.Modules;
using static TorchSharp.torch;

// 创建 AddWithBias 算子
var bias = torch.randn(10);
var addWithBias = OperatorBuilder.CreateAddWithBias()
    .Bias(bias)
    .Build();

// 创建 LinearOperator 算子
var weight = torch.randn(784, 256);
var linearOp = OperatorBuilder.CreateLinearOperator()
    .Weight(weight)
    .Bias(torch.randn(256))   // 偏置可选
    .Build();
```

---

## 3. ModuleProcessorRegistry 类

| 项目 | 说明 |
|------|------|
| **命名空间** | `TorchSharp.OnnxExporter` |
| **类型** | `sealed class`（单例） |
| **文件** | [ModuleProcessorRegistry.cs](../ModuleProcessorRegistry.cs) |
| **简要描述** | 模块处理器的全局注册表，采用单例模式管理 Module 类型到 `INodeProcessor` 的映射关系。支持按类型精确匹配和按名称模糊匹配两种查找策略。首次访问时自动初始化并注册所有内置处理器（100+ 个）。 |

### 静态属性

#### Instance

```csharp
public static ModuleProcessorRegistry Instance { get; }
```

**说明**: 获取全局唯一单例实例。线程安全（`readonly static` 初始化）。

---

### 静态方法

#### Register\<TModule\>(INodeProcessor)

```csharp
public static void Register<TModule>(INodeProcessor processor)
    where TModule : TorchSharp.torch.nn.Module
```

**功能**: 按模块类型注册自定义处理器。

| 参数 | 类型 | 说明 |
|------|------|------|
| `processor` | `INodeProcessor` | 处理器实现 |

**类型参数** `TModule`: 目标模块的类型（需继承 `Module`）。

---

#### RegisterByName(string, INodeProcessor)

```csharp
public static void RegisterByName(string moduleName, INodeProcessor processor)
```

**功能**: 按模块名称字符串注册处理器（适用于无法通过类型直接匹配的场景）。

| 参数 | 类型 | 说明 |
|------|------|------|
| `moduleName` | `string` | 模块名称（如 `"Concat"`、`"Reshape"`） |
| `processor` | `INodeProcessor` | 处理器实现 |

---

#### RegisterDefaultProcessors()

```csharp
public static void RegisterDefaultProcessors()
```

**功能**: 注册所有内置处理器。包括：

| 类别 | 已注册的处理器 |
|------|---------------|
| 卷积层 | Conv1d, Conv2d, Conv3d, ConvTranspose1d, ConvTranspose2d, ConvTranspose3d |
| 全连接层 | Linear, LinearOperator, Bilinear |
| 激活函数 | ReLU, ReLU6, LeakyReLU, PReLU, RReLU, ELU, CELU, SELU, Mish, SiLU, GELU, Sigmoid, Tanh, Softmax, Softmax2d, Softmin, LogSoftmax, Softplus, Hardswish, Hardsigmoid, Hardtanh, Hardshrink, Softshrink, Softsign, LogSigmoid, Tanhshrink, Threshold |
| 归一化 | BatchNorm1d/2d/3d, LayerNorm, GroupNorm, InstanceNorm1d/2d/3d, LocalResponseNorm |
| 池化层 | MaxPool1d/2d/3d, AvgPool1d/2d/3d, AdaptiveAvgPool1d/2d/3d, AdaptiveMaxPool1d/2d/3d, FractionalMaxPool2d/3d, MaxUnpool1d/2d/3d, LPPool1d/2d |
| Dropout | Dropout, Dropout1d, Dropout2d, Dropout3d, AlphaDropout |
| 填充 | ConstantPad1d/2d/3d, ReflectionPad1d/2d/3d, ReplicationPad1d/2d/3d, ZeroPad2d |
| 形状变换 | Flatten, Reshape, Transpose, Squeeze, Unsqueeze, Concat, Stack, Chunk, Split |
| 张量操作 | Fold, Unfold, PixelShuffle, PixelUnuffle, Upsample, ChannelShuffle |
| 嵌入 | Embedding, EmbeddingBag |
| RNN 系列 | RNN, RNNCell, LSTM, LSTMCell, GRU, GRUCell |
| 注意力 | MultiheadAttention |
| 自定义算子 | Add, Sub, Mul, Div, MatMul, Pow, Sqrt, Exp, Log, AddWithBias, Concat, Stack, ReshapeOp, TransposeOp, SqueezeOp, UnsqueezeOp, ClampOp, WhereOp, SumOp, MeanOp |
| 其他 | Identity, OneHot, PairwiseDistance, CosineSimilarity |

---

### 实例方法

#### GetProcessor(Module)

```csharp
public INodeProcessor? GetProcessor(TorchSharp.torch.nn.Module module)
```

**功能**: 根据模块实例查找对应的处理器。查找顺序：
1. 按 `module.GetType()` 精确匹配 `_processors` 字典
2. 若未命中，尝试按基类型匹配（`IsAssignableFrom`）
3. 若仍未命中，按 `module.GetType().Name` 在 `_nameProcessors` 字典中查找

| 参数 | 类型 | 说明 |
|------|------|------|
| `module` | `Module` | 模块实例 |

**返回值**: `INodeProcessor?` — 找到的处理器，若无匹配则返回 `null`。

---

#### GetProcessor(Type)

```csharp
public INodeProcessor? GetProcessor(Type moduleType)
```

**功能**: 根据模块类型查找处理器。查找逻辑同上，但不经过名称匹配步骤。

| 参数 | 类型 | 说明 |
|------|------|------|
| `moduleType` | `Type` | 模块的 `Type` 对象 |

**返回值**: `INodeProcessor?` — 找到的处理器，若无匹配则返回 `null`。

---

#### CanProcess(Module)

```csharp
public bool CanProcess(TorchSharp.torch.nn.Module module)
```

**功能**: 判断指定模块是否有可用的处理器。

| 参数 | 类型 | 说明 |
|------|------|------|
| `module` | `Module` | 模块实例 |

**返回值**: `bool` — 若存在对应处理器返回 `true`，否则返回 `false`。

---

### 静态便捷方法

#### GetProcessorStatic(Module)

```csharp
public static INodeProcessor? GetProcessorStatic(TorchSharp.torch.nn.Module module)
```

**说明**: `Instance.GetProcessor(module)` 的静态包装。

#### CanProcessStatic(Module)

```csharp
public static bool CanProcessStatic(TorchSharp.torch.nn.Module module)
```

**说明**: `Instance.CanProcess(module)` 的静态包装。

### 使用示例

```csharp
using TorchSharp;
using TorchSharp.OnnxExporter;
using TorchSharp.OnnxExporter.Processors;
using static TorchSharp.torch.nn;

// 查询已有模块是否被支持
var linear = Linear(10, 20);
bool supported = ModuleProcessorRegistry.CanProcessStatic(linear); // true

// 注册自定义处理器
ModuleProcessorRegistry.Register<MyCustomModule>(new MyCustomProcessor());

// 按名称注册
ModuleProcessorRegistry.RegisterByName("MyCustomOp", new MyCustomProcessor());
```

---

## 4. INodeProcessor 接口与 BaseProcessor 抽象类

### 4.1 INodeProcessor 接口

| 项目 | 说明 |
|------|------|
| **命名空间** | `TorchSharp.OnnxExporter.Processors` |
| **类型** | `interface` |
| **文件** | [INodeProcessor.cs](../Processors/INodeProcessor.cs) |
| **简要描述** | 节点处理器接口，定义了将 TorchSharp `Module` 转换为 ONNX `DataFlowNode` 的契约。所有模块处理器均需实现此接口。 |

#### 属性

##### OpType

```csharp
string OpType { get; }
```

**说明**: 该处理器生成的 ONNX 操作类型名称（如 `"Conv"`、`"Relu"`、`"MatMul"` 等），对应 ONNX 规范中的 op_type 字段。

---

#### 方法

##### CanProcess(Module)

```csharp
bool CanProcess(Module module)
```

**功能**: 判断此处理器是否能处理给定的模块实例。

| 参数 | 类型 | 说明 |
|------|------|------|
| `module` | `Module` | 待判断的模块实例 |

**返回值**: `bool` — 能处理返回 `true`，否则返回 `false`。

---

##### Process(Module, TraceContext)

```csharp
DataFlowNode Process(Module module, TraceContext context)
```

**功能**: 处理模块，生成对应的数据流节点并将其添加到上下文的图中。

| 参数 | 类型 | 说明 |
|------|------|------|
| `module` | `Module` | 待处理的模块实例 |
| `context` | `TraceContext` | 跟踪上下文，提供当前输入输出名、图形引用、形状信息等 |

**返回值**: `DataFlowNode` — 生成的数据流节点。

---

### 4.2 BaseProcessor\<TModule\> 抽象类

| 项目 | 说明 |
|------|------|
| **命名空间** | `TorchSharp.OnnxExporter.Processors` |
| **类型** | `abstract class` : `INodeProcessor` |
| **文件** | [BaseProcessor.cs](../Processors/BaseProcessor.cs) |
| **简要描述** | 泛型抽象基类，简化处理器实现。自动完成类型检查和转换，子类只需关注特定模块类型的处理逻辑。 |

```csharp
public abstract class BaseProcessor<TModule> : INodeProcessor
    where TModule : Module
```

**已实现的成员**:

| 成员 | 实现 |
|------|------|
| `CanProcess(Module)` | 返回 `module is TModule` |
| `Process(Module, TraceContext)` | 将 `module` 强转为 `TModule` 后调用泛型版本 |

**需要子类实现的抽象成员**:

| 成员 | 签名 |
|------|------|
| `OpType` | `string OpType { get; }` |
| `Process` | `DataFlowNode Process(TModule module, TraceContext context)` |

---

### 4.3 ElementWiseProcessor 类

| 项目 | 说明 |
|------|------|
| **命名空间** | `TorchSharp.OnnxExporter.Processors` |
| **类型** | `class` : `INodeProcessor` |
| **文件** | [ElementWiseProcessor.cs](../Processors/ElementWiseProcessor.cs) |
| **简要描述** | 通用逐元素操作处理器，通过构造函数传入 `opType` 即可用于多种单输入逐元素算子（Sqrt、Exp、Log、Clamp、Where、ReduceSum、ReduceMean 等）。 |

#### 构造函数

```csharp
public ElementWiseProcessor(string opType)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `opType` | `string` | ONNX 操作类型名称 |

#### Process 行为

从 `TraceContext` 获取当前值作为输入，创建临时输出名称，构造 `DataFlowNode` 加入图，并将当前值更新为输出名称。

### 使用示例

```csharp
using TorchSharp.OnnxExporter.Processors;
using TorchSharp.OnnxExporter.DataFlow;

// 自定义处理器实现（推荐使用 BaseProcessor）
public class MyConvProcessor : BaseProcessor<Conv2d>
{
    public override string OpType => "Conv";

    public override DataFlowNode Process(Conv2d module, TraceContext context)
    {
        var input = context.GetCurrentValue();
        var output = context.CreateTempName();

        var node = new DataFlowNode(OpType, new[] { input }, new[] { output });
        node.Attributes["kernel_shape"] = new int[] { module.kernel_size[0], module.kernel_size[1] };
        // ... 更多属性设置

        context.Graph.AddNode(node);
        context.SetCurrentValue(output);
        return node;
    }
}

// 使用 ElementWiseProcessor 处理通用逐元素操作
var sqrtProcessor = new ElementWiseProcessor("Sqrt");
var expProcessor = new ElementWiseProcessor("Exp");
```

---

## 5. 有状态算子类（Operator 基类体系）

| 项目 | 说明 |
|------|------|
| **命名空间** | `TorchSharp.OnnxExporter.Modules` |
| **文件** | [Operators.cs](../Modules/Operators.cs) |
| **简要描述** | 提供一组可在 TorchSharp 模型中使用的可追踪算子模块。这些算子继承自 `Module`，可像标准 TorchSharp 层一样嵌入 `Sequential` 或自定义模块中，并能被 ONNX 导出器正确识别和转换。 |

### 5.1 IOperator 接口

```csharp
public interface IOperator
{
    Tensor forward(params Tensor[] inputs);
}
```

**说明**: 算子的核心接口，定义前向计算签名。所有具体算子均实现此接口。

| 成员 | 签名 | 说明 |
|------|------|------|
| `forward` | `Tensor forward(params Tensor[] inputs)` | 前向计算方法，接收可变数量的张量输入，返回计算结果张量 |

---

### 5. Operator 抽象类

```csharp
public abstract class Operator : Module
```

**继承关系**: `Operator` → `Module`

**说明**: 所有有状态算子的抽象基类。继承 `Module` 使其可作为子模块嵌入 TorchSharp 模型；同时要求子类实现 `IOperator.forward()` 方法。

#### 构造函数

```csharp
protected Operator(string name)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `name` | `string` | 算子名称（传递给基类 `Module`） |

#### 抽象方法

```csharp
public abstract Tensor forward(params Tensor[] inputs);
```

---

### 具体算子类一览

以下所有算子类均位于 `TorchSharp.OnnxExporter.Modules` 命名空间，继承自 `Operator`。

#### Add — 逐元素加法

| 项目 | 值 |
|------|-----|
| **构造函数** | `public Add()` |
| **OpType** | `"Add"` |
| **forward 逻辑** | 对 `inputs[0..N]` 依次执行 `torch.add()`，支持两个及以上输入 |
| **输入要求** | 至少 2 个 Tensor |
| **异常** | `ArgumentException` — 输入少于 2 个时抛出 |

```csharp
var add = new Add();
var result = add.forward(a, b, c); // a + b + c
```

---

#### Sub — 逐元素减法

| 项目 | 值 |
|------|-----|
| **构造函数** | `public Sub()` |
| **OpType** | `"Sub"` |
| **forward 逻辑** | 对 `inputs[0..N]` 依次执行 `torch.sub()` |
| **输入要求** | 至少 2 个 Tensor |
| **异常** | `ArgumentException` — 输入少于 2 个时抛出 |

---

#### Mul — 逐元素乘法

| 项目 | 值 |
|------|-----|
| **构造函数** | `public Mul()` |
| **OpType** | `"Mul"` |
| **forward 逻辑** | 对 `inputs[0..N]` 依次执行 `torch.mul()` |
| **输入要求** | 至少 2 个 Tensor |
| **异常** | `ArgumentException` — 输入少于 2 个时抛出 |

---

#### Div — 逐元素除法

| 项目 | 值 |
|------|-----|
| **构造函数** | `public Div()` |
| **OpType** | `"Div"` |
| **forward 逻辑** | 对 `inputs[0..N]` 依次执行 `torch.div()` |
| **输入要求** | 至少 2 个 Tensor |
| **异常** | `ArgumentException` — 输入少于 2 个时抛出 |

---

#### MatMul — 矩阵乘法

| 项目 | 值 |
|------|-----|
| **构造函数** | `public MatMul()` |
| **OpType** | `"MatMul"` |
| **forward 逻辑** | 对 `inputs[0..N]` 依次执行 `torch.matmul()` |
| **输入要求** | 至少 2 个 Tensor |
| **异常** | `ArgumentException` — 输入少于 2 个时抛出 |

---

#### Pow — 幂运算

| 项目 | 值 |
|------|-----|
| **构造函数** | `public Pow()` |
| **OpType** | `"Pow"` |
| **forward 逻辑** | `torch.pow(inputs[0], inputs[1])`，仅使用前两个输入 |
| **输入要求** | 至少 2 个 Tensor（底数和指数） |
| **异常** | `ArgumentException` — 输入少于 2 个时抛出 |

---

#### Sqrt — 平方根

| 项目 | 值 |
|------|-----|
| **构造函数** | `public Sqrt()` |
| **OpType** | `"Sqrt"` |
| **forward 逻辑** | `torch.sqrt(inputs[0])`，仅使用第一个输入 |
| **输入要求** | 至少 1 个 Tensor |
| **异常** | `ArgumentException` — 无输入时抛出 |

---

#### Exp — 指数运算

| 项目 | 值 |
|------|-----|
| **构造函数** | `public Exp()` |
| **OpType** | `"Exp"` |
| **forward 逻辑** | `torch.exp(inputs[0])` |
| **输入要求** | 至少 1 个 Tensor |
| **异常** | `ArgumentException` — 无输入时抛出 |

---

#### Log — 对数运算

| 项目 | 值 |
|------|-----|
| **构造函数** | `public Log()` |
| **OpType** | `"Log"` |
| **forward 逻辑** | `torch.log(inputs[0])` |
| **输入要求** | 至少 1 个 Tensor |
| **异常** | `ArgumentException` — 无输入时抛出 |

---

#### AddWithBias — 加偏置

| 项目 | 值 |
|------|-----|
| **构造函数** | `public AddWithBias(Tensor bias)` |
| **OpType** | — |
| **字段** | `public readonly Tensor bias` |
| **forward 逻辑** | `torch.add(inputs[0], bias)` |
| **输入要求** | 至少 1 个 Tensor |
| **异常** | `ArgumentException` — 无输入时抛出 |

```csharp
var addWithBias = new AddWithBias(torch.randn(10));
var result = addWithBias.forward(x); // x + bias
```

---

#### LinearOperator — 线性变换算子

| 项目 | 值 |
|------|-----|
| **构造函数** | `public LinearOperator(Tensor weight, Tensor? bias = null)` |
| **字段** | `public readonly Tensor weight`, `public readonly Tensor? bias` |
| **forward 逻辑** | `matmul(inputs[0], weight)` + 可选 `add(output, bias)` |
| **输入要求** | 至少 1 个 Tensor |
| **异常** | `ArgumentException` — 无输入时抛出 |

```csharp
var linearOp = new LinearOperator(
    weight: torch.randn(784, 256),
    bias: torch.randn(256)     // 可选，传 null 则无偏置
);
var result = linearOp.forward(x); // x @ weight + bias
```

### 使用示例 — 在 Sequential 中组合算子

```csharp
using TorchSharp;
using TorchSharp.OnnxExporter.Modules;
using static TorchSharp.torch.nn;

var model = Sequential(
    Linear(784, 128),
    new ReLU(),
    new Mul(),           // 自定义算子：需要多分支输入时配合残差连接使用
    new AddWithBias(torch.randn(128)),
    Linear(128, 10)
);

var input = torch.randn(1, 784);
OnnxExporter.Export(model, input, "custom_model.onnx", "CustomModel");
```

---

## 6. 张量操作算子类（TensorOperators）

| 项目 | 说明 |
|------|------|
| **命名空间** | `TorchSharp.OnnxExporter.Modules` |
| **文件** | [TensorOperators.cs](../Modules/TensorOperators.cs) |
| **简要描述** | 提供张量形状变换和归约操作的算子模块。每个算子在构造时接收操作参数（如维度、形状等），在 `forward` 中对输入张量执行对应的 TorchSharp 操作。所有类均继承自 `Operator`。 |

### Concat — 张量拼接

| 项目 | 值 |
|------|-----|
| **构造函数** | `public Concat(int dim = 0)` |
| **公开字段** | `public int dim` |
| **OpType** | `"Concat"` |
| **forward 逻辑** | `torch.cat(inputs, dim)` |
| **输入要求** | 至少 1 个 Tensor |
| **异常** | `ArgumentException` — 无输入时抛出 |

```csharp
var concat = new Concat(dim: 1);
var result = concat.forward(tensorA, tensorB, tensorC); // 沿 dim=1 拼接
```

---

### Stack — 张量堆叠

| 项目 | 值 |
|------|-----|
| **构造函数** | `public Stack(int dim = 0)` |
| **公开字段** | `public int dim` |
| **OpType** | — |
| **forward 逻辑** | `torch.stack(inputs, dim)` |
| **输入要求** | 至少 1 个 Tensor |
| **异常** | `ArgumentException` — 无输入时抛出 |

---

### ReshapeOp — 形状重塑

| 项目 | 值 |
|------|-----|
| **构造函数** | `public ReshapeOp(params long[] shape)` |
| **公开字段** | `public long[] shape` |
| **OpType** | `"Reshape"` |
| **forward 逻辑** | `torch.reshape(inputs[0], shape)` |
| **输入要求** | 至少 1 个 Tensor |
| **异常** | `ArgumentException` — 无输入时抛出 |

```csharp
var reshape = new ReshapeOp(-1, 256);  // 展平后变为 (N, 256)
var result = reshape.forward(x);
```

---

### TransposeOp — 维度转置

| 项目 | 值 |
|------|-----|
| **构造函数** | `public TransposeOp(int dim0 = 0, int dim1 = 1)` |
| **公开字段** | `public int dim0`, `public int dim1` |
| **OpType** | `"Transpose"` |
| **forward 逻辑** | `torch.transpose(inputs[0], dim0, dim1)` |
| **输入要求** | 至少 1 个 Tensor |
| **异常** | `ArgumentException` — 无输入时抛出 |

```csharp
var transpose = new TransposeOp(0, 2);  // 交换第0维和第2维
var result = transpose.forward(x);
```

---

### SqueezeOp — 压缩维度

| 项目 | 值 |
|------|-----|
| **构造函数** | `public SqueezeOp(int dim = -1)` |
| **公开字段** | `public int dim` |
| **OpType** | `"Squeeze"` |
| **forward 逻辑** | `dim >= 0` 时调用 `torch.squeeze(inputs[0], dim)`；否则调用 `torch.squeeze(inputs[0])` 压缩所有长度为1的维度 |
| **输入要求** | 至少 1 个 Tensor |
| **异常** | `ArgumentException` — 无输入时抛出 |

---

### UnsqueezeOp — 插入维度

| 项目 | 值 |
|------|-----|
| **构造函数** | `public UnsqueezeOp(int dim = 0)` |
| **公开字段** | `public int dim` |
| **OpType** | `"Unsqueeze"` |
| **forward 逻辑** | `torch.unsqueeze(inputs[0], dim)` |
| **输入要求** | 至少 1 个 Tensor |
| **异常** | `ArgumentException` — 无输入时抛出 |

---

### ClampOp — 值域裁剪

| 项目 | 值 |
|------|-----|
| **构造函数** | `public ClampOp(double? min = null, double? max = null)` |
| **公开字段** | `public double? min`, `public double? max` |
| **OpType** | `"Clamp"` |
| **forward 逻辑** | 根据 min/max 的有无分别调用不同签名的 `torch.clamp()`；两者都为 null 时原样返回输入 |
| **输入要求** | 至少 1 个 Tensor |
| **异常** | `ArgumentException` — 无输入时抛出 |

```csharp
var clamp = new ClampOp(min: 0.0, max: 1.0);  // 裁剪到 [0, 1]
var result = clamp.forward(x);
```

---

### WhereOp — 条件选择

| 项目 | 值 |
|------|-----|
| **构造函数** | `public WhereOp()` |
| **OpType** | `"Where"` |
| **forward 逻辑** | `torch.where(inputs[0], inputs[1], inputs[2])` |
| **输入要求** | **恰好 3 个** Tensor（条件、x、y） |
| **异常** | `ArgumentException` — 输入数量不为 3 时抛出 |

```csharp
var whereOp = new WhereOp();
var result = whereOp.forward(condition, x, y);  // 条件为真选x，否则选y
```

---

### SumOp — 求和归约

| 项目 | 值 |
|------|-----|
| **构造函数** | `public SumOp(int? dim = null, bool keepdim = false)` |
| **公开字段** | `public int? dim`, `public bool keepdim` |
| **OpType** | `"ReduceSum"` |
| **forward 逻辑** | 有 dim 时 `torch.sum(inputs[0], dim, keepdim)`；否则 `torch.sum(inputs[0])` 全局求和 |
| **输入要求** | 至少 1 个 Tensor |
| **异常** | `ArgumentException` — 无输入时抛出 |

---

### MeanOp — 均值归约

| 项目 | 值 |
|------|-----|
| **构造函数** | `public MeanOp(int? dim = null, bool keepdim = false)` |
| **公开字段** | `public int? dim`, `public bool keepdim` |
| **OpType** | `"ReduceMean"` |
| **forward 逻辑** | 有 dim 时 `torch.mean(inputs[0], new long[] { dim }, keepdim)`；否则 `torch.mean(inputs[0])` 全局均值 |
| **输入要求** | 至少 1 个 Tensor |
| **异常** | `ArgumentException` — 无输入时抛出 |

### 使用示例

```csharp
using TorchSharp.OnnxExporter.Modules;

// 形状变换流水线
var pipeline = new Modules.ReshapeOp(-1, 768)
    .forward(
        new Modules.Concat(dim: 0).forward(part1, part2),
        new Modules.SqueezeOp(dim: 0).forward(extra)
    );

// 归约操作
var sumResult = new Modules.SumOp(dim: 1, keepdim: true).forward(features);
var meanResult = new Modules.MeanOp().forward(features);  // 全局均值
```

---

## 7. 数据流相关类

### 7.1 DataFlowGraph 类

| 项目 | 说明 |
|------|------|
| **命名空间** | `TorchSharp.OnnxExporter.DataFlow` |
| **类型** | `class` |
| **文件** | [DataFlowGraph.cs](../DataFlow/DataFlowGraph.cs) |
| **简要描述** | 数据流图的内存表示，在符号跟踪过程中逐步构建。包含节点列表、输入/输出列表、初始器（权重）、以及各阶段的形状信息。最终可通过 `ToGraphProto()` 转换为 ONNX `GraphProto`。 |

#### 属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `Nodes` | `List<DataFlowNode>` | 图中所有数据流节点的有序列表 |
| `Inputs` | `List<string>` | 图输入张量名称列表 |
| `Outputs` | `List<string>` | 图输出张量名称列表 |
| `Initializers` | `List<TensorProto>` | 模型权重/常量的 ONNX TensorProto 列表 |
| `InputShapes` | `Dictionary<string, List<long>>` | 输入张量名称 → 形状 的映射 |
| `OutputShapes` | `Dictionary<string, List<long>>` | 输出张量名称 → 形状 的映射 |
| `IntermediateShapes` | `Dictionary<string, List<long>>` | 中间张量名称 → 形状 的映射 |

---

#### 方法

##### AddNode(DataFlowNode)

```csharp
public DataFlowNode AddNode(DataFlowNode node)
```

**功能**: 向图中添加一个数据流节点。

| 参数 | 类型 | 说明 |
|------|------|------|
| `node` | `DataFlowNode` | 待添加的节点 |

**返回值**: `DataFlowNode` — 返回传入的节点引用（方便链式调用）。

---

##### AddInitializer(string, Tensor)

```csharp
public void AddInitializer(string name, Tensor tensor)
```

**功能**: 从 TorchSharp 张量创建 ONNX 初始化器（权重）。自动推断数据类型（Float32/Double）和形状。同名初始器不会重复添加。

| 参数 | 类型 | 说明 |
|------|------|------|
| `name` | `string` | 初始器名称 |
| `tensor` | `Tensor` | TorchSharp 张量（可为 null，此时不执行任何操作） |

**数据类型映射**:
- `ScalarType.Float32` → `TensorProto.Types.DataType.Float`
- 其他类型 → `TensorProto.Types.DataType.Double`

---

##### AddInitializer(string, float[], long[])

```csharp
public void AddInitializer(string name, float[] data, long[] dims)
```

**功能**: 从浮点数组和维度数组创建 ONNX 初始化器。

| 参数 | 类型 | 说明 |
|------|------|------|
| `name` | `string` | 初始器名称 |
| `data` | `float[]` | 浮点数据（可为 null） |
| `dims` | `long[]` | 各维度大小 |

---

##### AddInitializer(string, long[])

```csharp
public void AddInitializer(string name, long[] data)
```

**功能**: 从整数数组创建 Int64 类型的 ONNX 初始化器（常用于存储形状信息）。

| 参数 | 类型 | 说明 |
|------|------|------|
| `name` | `string` | 初始器名称 |
| `data` | `long[]` | 整数数据（可为 null） |

---

##### AddInput(string, Tensor)

```csharp
public void AddInput(string name, Tensor tensor)
```

**功能**: 注册图输入，同时记录其形状信息。

| 参数 | 类型 | 说明 |
|------|------|------|
| `name` | `string` | 输入名称 |
| `tensor` | `Tensor` | 输入张量（用于获取形状，可为 null） |

---

##### AddOutput(string, Tensor)

```csharp
public void AddOutput(string name, Tensor tensor)
```

**功能**: 注册图输出，同时记录其形状信息。

| 参数 | 类型 | 说明 |
|------|------|------|
| `name` | `string` | 输出名称 |
| `tensor` | `Tensor` | 输出张量（用于获取形状，可为 null） |

---

##### AddIntermediateShape(string, List\<long\>)

```csharp
public void AddIntermediateShape(string name, List<long> shape)
```

**功能**: 记录中间张量的形状信息（用于生成 ValueInfo）。

| 参数 | 类型 | 说明 |
|------|------|------|
| `name` | `string` | 中间张量名称 |
| `shape` | `List<long>` | 形状列表（可为 null） |

---

##### ToGraphProto()

```csharp
public GraphProto ToGraphProto()
```

**功能**: 将整个数据流图转换为 ONNX `GraphProto`。处理内容包括：
- 构建 Input ValueInfoProto（含形状和动态维度标记）
- 将所有 DataFlowNode 转换为 NodeProto
- 构建 Output ValueInfoProto（含形状信息）
- 添加 Initializer 列表

**返回值**: `GraphProto` — ONNX 图协议缓冲区对象。

**维度处理规则**:
- `dim > 0`: 使用 `DimValue` 设为固定值
- `dim <= 0`: 使用 `DimParam` 设为 `"dynamic"`

---

### 7.2 DataFlowNode 类

| 项目 | 说明 |
|------|------|
| **命名空间** | `TorchSharp.OnnxExporter.DataFlow` |
| **类型** | `class` |
| **文件** | [DataFlowNode.cs](../DataFlow/DataFlowNode.cs) |
| **简要描述** | 数据流图中单个节点的表示，对应 ONNX 规范中的一个计算节点 (NodeProto)。包含操作类型、输入输出名称、属性字典等信息。 |

#### 属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `Name` | `string` | 节点唯一名称，格式为 `{OpType}_{GUID}`，可读写 |
| `OpType` | `string` | ONNX 操作类型（如 `"Conv"`, `"Relu"`），只读 |
| `Inputs` | `IReadOnlyList<string>` | 输入张量名称列表，只读 |
| `Outputs` | `IReadOnlyList<string>` | 输出张量名称列表，只读 |
| `Attributes` | `Dictionary<string, object>` | 节点属性字典，可读写 |

---

#### 构造函数

```csharp
public DataFlowNode(
    string opType,
    IEnumerable<string> inputs,
    IEnumerable<string> outputs
)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `opType` | `string` | 操作类型（不可为 null） |
| `inputs` | `IEnumerable<string>` | 输入名称集合（可为 null，此时为空列表） |
| `outputs` | `IEnumerable<string>` | 输出名称集合（可为 null，此时为空列表） |

**异常**: `ArgumentNullException` — `opType` 为 null 时抛出。

**自动行为**: 构造时自动生成 `{OpType}_{GUID}` 格式的唯一名称，初始化空的 Attributes 字典。

---

#### 方法

##### ToNodeProto()

```csharp
public NodeProto ToNodeProto()
```

**功能**: 将此数据流节点转换为 ONNX `NodeProto`。

**返回值**: `NodeProto` — 包含 OpType、Name、Inputs、Outputs 和所有属性的 ONNX 节点。

**支持的属性类型转换**:

| C# 类型 | ONNX AttributeType |
|---------|-------------------|
| `int` / `long` | Int |
| `float` / `double` | Float |
| `string` | String (UTF-8 ByteString) |
| `bool` | Int (1/0) |
| `IEnumerable<int\|long>` | Ints |
| `IEnumerable<float\|double>` | Floats |
| `IEnumerable<string>` | Strings |
| `TensorProto` | Tensor |
| `GraphProto` | Graph |
| 不支持的类型 | 跳过（输出警告日志） |

---

### 7.3 TraceContext 类

| 项目 | 说明 |
|------|------|
| **命名空间** | `TorchSharp.OnnxExporter.DataFlow` |
| **类型** | `class` |
| **文件** | [TraceContext.cs](../DataFlow/TraceContext.cs) |
| **简要描述** | 符号跟踪过程的上下文对象，贯穿整个跟踪生命周期。维护值名称映射、形状信息、作用域栈、当前值指针等状态，供处理器在 `Process()` 方法中查询和更新。 |

#### 属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `Graph` | `DataFlowGraph?` | 关联的数据流图引用（可在外部设置） |
| `Values` | `Dictionary<string, string>` | 用户变量名 → ONNX 临时名的映射 |
| `NamedValues` | `Dictionary<string, string>` | 命名值存储（如 scope、current_input 等） |
| `ValueShapes` | `Dictionary<string, List<long>>` | ONNX 临时名 → 形状的映射 |

---

#### 构造函数

```csharp
public TraceContext()
```
创建空的跟踪上下文。

```csharp
public TraceContext(DataFlowGraph graph)
```
创建并与指定的数据流图关联。

---

#### 方法

##### AddValue(string, Tensor)

```csharp
public string AddValue(string name, Tensor value)
```

**功能**: 注册一个值，分配唯一的 ONNX 临时名称（`tmp_N` 格式），并记录形状。

| 参数 | 类型 | 说明 |
|------|------|------|
| `name` | `string` | 用户变量名 |
| `value` | `Tensor` | 张量值（用于获取形状） |

**返回值**: `string` — 分配的 ONNX 临时名称。

---

##### GetValue(string)

```csharp
public string GetValue(string name)
```

**功能**: 查找用户变量名对应的 ONNX 名称。若未找到则返回原始名称本身。

**返回值**: `string` — ONNX 名称或原始名称。

---

##### SetNamedValue(string, string) / GetNamedValue(string)

```csharp
public void SetNamedValue(string key, string value)
public string GetNamedValue(string key)
```

**功能**: 存取命名键值对（用于存储 scope 等元信息）。`GetNamedValue` 未找到时返回 key 本身。

---

##### CreateTempName()

```csharp
public string CreateTempName()
```

**功能**: 生成新的唯一临时名称（格式 `tmp_0`, `tmp_1`, ...，全局递增计数器）。

**返回值**: `string` — 新的临时名称。

---

##### GetCurrentValue() / SetCurrentValue(string)

```csharp
public string GetCurrentValue()
public void SetCurrentValue(string value)
```

**功能**: 管理"当前值"指针——即上一个操作输出的 ONNX 名称。处理器通常读取 `GetCurrentValue()` 作为输入，处理后通过 `SetCurrentValue()` 更新。
`GetCurrentValue()` 在未设置过时返回最近一次创建的临时名。

---

##### GetNextValue()

```csharp
public string GetNextValue()
```

**功能**: 预览下一个将被分配的临时名称（不消耗计数器）。可用于提前规划输出名称。

**返回值**: `string` — 下一个临时名称（如 `tmp_5`）。

---

##### SetOriginalInput(string) / GetOriginalInput()

```csharp
public void SetOriginalInput(string value)
public string? GetOriginalInput()
```

**功能**: 记录和获取整个跟踪过程的原始输入名称。用于残差连接等场景中回溯到最初输入。

---

##### PushScope(string) / PopScope()

```csharp
public void PushScope(string scopeName)
public void PopScope()
```

**功能**: 作用域栈管理。`PushScope` 将当前 scope 名存入 NamedValues（key=`"current_scope"`）；`PopScope` 将其清空。

---

##### SetShape(string, List\<long\>) / TryGetShape(string, out List\<long\>)

```csharp
public void SetShape(string name, List<long> shape)
public bool TryGetShape(string name, out List<long> shape)
```

**功能**: 存取张量形状信息。`TryGetShape` 返回是否找到。

### 使用示例

```csharp
using TorchSharp.OnnxExporter.DataFlow;

var graph = new DataFlowGraph();
var ctx = new TraceContext(graph);

// 注册输入
ctx.SetNamedValue("input", "input");
ctx.SetShape("input", new List<long> { 1, 3, 224, 224 });

// 模拟处理器的工作流程
ctx.SetCurrentValue("input");                    // 设置当前输入
var outputName = ctx.CreateTempName();            // 预分配输出名 tmp_0
var node = new DataFlowNode("Conv", new[] { ctx.GetCurrentValue() }, new[] { outputName });
graph.AddNode(node);
ctx.SetCurrentValue(outputName);                  // 推进当前值
ctx.SetShape(outputName, new List<long> { 1, 64, 112, 112 });

// 作用域管理
ctx.PushScope("block_1");
// ... 子操作 ...
ctx.PopScope();
```

---

## 8. OnnxGraphBuilder 类

| 项目 | 说明 |
|------|------|
| **命名空间** | `TorchSharp.OnnxExporter.Builder` |
| **类型** | `sealed class` |
| **文件** | [OnnxGraphBuilder.cs](../Builder/OnnxGraphBuilder.cs) |
| **简要描述** | ONNX 图构建器，负责将 `DataFlowGraph`（内存中的数据流图表示）转换为标准的 ONNX `ModelProto`（protobuf 序列化格式）。是导出管线中从内部表示到外部格式的桥梁。 |

### 构造函数

```csharp
public OnnxGraphBuilder()
```

**初始化内容**:

| 字段 | 初始值 |
|------|--------|
| `_model.IrVersion` | `7` |
| `_model.ProducerName` | `"TorchSharp.OnnxExporter"` |
| `_model.ProducerVersion` | `"1.0.0"` |
| Opset Import (ai.onnx) | Version = 14 |
| Opset Import (空域) | Version = 14 |

---

### 公开方法

#### AddInput(ValueInfoProto)

```csharp
public void AddInput(ValueInfoProto input)
```

**功能**: 手动添加输入张量信息。

| 参数 | 类型 | 说明 |
|------|------|------|
| `input` | `ValueInfoProto` | ONNX 输入值信息 |

**异常**: `ArgumentNullException` — `input` 为 null 时抛出。

> **注意**: 在实际导出流程中，输入主要通过 `Build()` 方法从 `DataFlowGraph.Inputs` 自动填充，此方法主要用于高级自定义场景。

---

#### AddInitializer(TensorProto)

```csharp
public void AddInitializer(TensorProto initializer)
```

**功能**: 手动添加初始化器（模型权重）。

| 参数 | 类型 | 说明 |
|------|------|------|
| `initializer` | `TensorProto` | ONNX 张量原型 |

**异常**: `ArgumentNullException` — `initializer` 为 null 时抛出。

---

#### AddOutput(ValueInfoProto)

```csharp
public void AddOutput(ValueInfoProto output)
```

**功能**: 手动添加输出张量信息。

| 参数 | 类型 | 说明 |
|------|------|------|
| `output` | `ValueInfoProto` | ONNX 输出值信息 |

**异常**: `ArgumentNullException` — `output` 为 null 时抛出。

---

#### Build(DataFlowGraph, string)

```csharp
public ModelProto Build(DataFlowGraph dataFlow, string modelName)
```

**功能**: 核心方法——将数据流图完整转换为 ONNX ModelProto。

| 参数 | 类型 | 说明 |
|------|------|------|
| `dataFlow` | `DataFlowGraph` | 待转换的数据流图 |
| `modelName` | `string` | 模型名称（为空时默认 `"TorchSharpModel"`） |

**返回值**: `ModelProto` — 完整的 ONNX 模型协议缓冲区。

**异常**:

| 异常类型 | 触发条件 |
|----------|----------|
| `ArgumentNullException` | `dataFlow` 为 null |
| `InvalidOperationException` | 图构建过程中发生错误（含详细错误信息和排查建议） |

**内部处理流程**:

```
1. 处理输入 (Inputs)
   ├─ 跳过空名称
   ├─ 构建 ValueInfoProto（含类型 Float 和形状信息）
   └─ 动态维度标记 (dim <= 0 → "dynamic")

2. 处理初始器 (Initializers)
   └─ 直接复制到 _initializers 列表

3. 处理节点 (Nodes)
   ├─ 跳过 null 节点
   └─ 调用每个 DataFlowNode.ToNodeProto() 转换

4. 处理输出 (Outputs)
   ├─ 跳过空名称
   └─ 构建 ValueInfoProto（含形状信息）

5. 组装 GraphProto
   ├─ graph.Input ← 所有输入
   ├─ graph.Node ← 所有节点
   ├─ graph.Initializer ← 所有初始器
   ├─ graph.Output ← 所有输出
   └─ graph.ValueInfo ← 中间张量形状信息

6. 组装 ModelProto
   ├─ model.Graph ← graph
   └─ model.DocString ← modelName
```

---

### 内部方法（供扩展参考）

#### ConvertToTensorProto(DataFlowNode, string)

```csharp
internal TensorProto ConvertToTensorProto(DataFlowNode node, string name)
```

**功能**: 将数据流节点转换为 TensorProto（辅助方法）。

#### InferShape(Tensor)

```csharp
internal TensorShapeProto InferShape(Tensor tensor)
```

**功能**: 从 TorchSharp 张量推断 ONNX 形状。正数维度用 `DimValue`，非正数维度用 `DimParam = "batch_size"`。

### 使用示例

```csharp
using TorchSharp.OnnxExporter.Builder;
using TorchSharp.OnnxExporter.DataFlow;

// 方式一：通过 OnnxExporter.Export() 间接使用（推荐）
OnnxExporter.Export(model, dummyInput, "output.onnx", "MyModel");

// 方式二：手动构建
var dataFlowGraph = new DataFlowGraph();
// ... 通过 SymbolicTraceEngine 或手动填充 dataFlowGraph ...

var builder = new OnnxGraphBuilder();
var modelProto = builder.Build(dataFlowGraph, "ManualModel");

// 序列化为文件
using var fs = File.Create("manual_output.onnx");
using var coded = new Google.Protobuf.CodedOutputStream(fs);
modelProto.WriteTo(coded);
```

---

## 9. SymbolicTraceEngine 类

| 项目 | 说明 |
|------|------|
| **命名空间** | `TorchSharp.OnnxExporter.Tracing` |
| **类型** | `class` |
| **文件** | [SymbolicTraceEngine.cs](../Tracing/SymbolicTraceEngine.cs) |
| **简要描述** | 符号跟踪引擎——导出管线的核心组件。负责模拟模型的 `forward` 执行过程，在不进行实际数值计算的前提下，记录每个操作的输入/输出依赖关系，构建完整的数据流图 (`DataFlowGraph`)。支持 `Sequential`、`ModuleList`、标准模块递归、以及反射发现的算子字段四种跟踪策略。 |

### 构造函数

```csharp
public SymbolicTraceEngine(int maxRecursionDepth = 100)
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `maxRecursionDepth` | `int` | `100` | 最大递归深度，防止无限递归 |

---

### 公开方法

#### Trace(Module, Tensor)

```csharp
public DataFlowGraph Trace(Module model, Tensor dummyInput)
```

**功能**: 跟踪模型并构建完整的数据流图。这是导出流程中最关键的步骤之一。

| 参数 | 类型 | 说明 |
|------|------|------|
| `model` | `Module` | 要跟踪的 TorchSharp 模型 |
| `dummyInput` | `Tensor` | 虚拟输入张量（用于形状推断） |

**返回值**: `DataFlowGraph` — 构建完成的数据流图。

**异常**:

| 异常类型 | 触发条件 |
|----------|----------|
| `ArgumentNullException` | `model` 或 `dummyInput` 为 null |
| `InvalidOperationException` | 跟踪过程中发生错误（含模型类型、输入形状、详细建议） |

**执行流程**:

```
1. 创建 DataFlowGraph 和 TraceContext
2. 注册输入 ("input") 及其形状
3. 调用 TraceModule(model, "input") 开始跟踪
4. 注册最终输出
5. 推断中间张量形状 (InferIntermediateShapesFromTrace)
6. 返回 DataFlowGraph
```

---

#### TraceModule(Module, string)

```csharp
public string TraceModule(Module module, string inputName)
```

**功能**: 跟踪单个模块，自动分发到适当的跟踪策略。

| 参数 | 类型 | 说明 |
|------|------|------|
| `module` | `Module` | 要跟踪的模块 |
| `inputName` | `string` | 输入张量的 ONNX 名称 |

**返回值**: `string` — 输出张量的 ONNX 名称。

**分发策略**:

| 模块类型 | 跟踪方法 |
|----------|----------|
| `Sequential` | `TraceSequential()` — 顺序跟踪所有子模块 |
| `ModuleList*` | `TraceModuleList()` — 顺序跟踪所有子模块 |
| 其他 | `TraceStandardModule()` — 标准模块处理 |

---

#### TraceSequential(Module, string)

```csharp
public string TraceSequential(Module module, string inputName)
```

**功能**: 跟踪 `Sequential` 模块。遍历所有子模块，将前一子模块的输出作为后一子模块的输入，串联执行。

**返回值**: `string` — 最后一个子模块的输出名称。

---

#### TraceModuleList(Module, string)

```csharp
public string TraceModuleList(Module moduleList, string inputName)
```

**功能**: 跟踪 `ModuleList` 模块。逻辑同 `TraceSequential`。

**返回值**: `string` — 最后一个子模块的输出名称。

### 内部跟踪策略 (TraceStandardModule)

`TraceStandardModule()` 是最核心的跟踪方法，按优先级依次尝试以下策略：

```
优先级 1: 查找注册的 INodeProcessor
         ↓ 找到 → ProcessWithProcessor() 生成 DataFlowNode
         ↓ 未找到 ↓

优先级 2: 检查是否为 Sequential / ModuleList
         ↓ 是 → 递归调用 TraceSequential / TraceModuleList
         ↓ 否 ↓

优先级 3: 检查是否有子模块 (children())
         ↓ 有 → 递归跟踪每个子模块
         ↓ 否 ↓

优先级 4: 反射发现 Operator 字段 (DiscoverOperatorFields)
         ↓ 找到 → 导出张量参数 + 递归跟踪每个算子
         ↓ 未找到 ↓

优先级 5: 透传（返回 inputName，输出警告日志）
```

### 使用示例

```csharp
using TorchSharp.OnnxExporter.Tracing;

var engine = new SymbolicTraceEngine(maxRecursionDepth: 50);
var dataFlowGraph = engine.Trace(myModel, torch.randn(1, 3, 224, 224));

// dataFlowGraph 现在包含了完整的 ONNX 图结构
Console.WriteLine($"节点数: {dataFlowGraph.Nodes.Count}");
Console.WriteLine($"输入: {string.Join(", ", dataFlowGraph.Inputs)}");
Console.WriteLine($"输出: {string.Join(", ", dataFlowGraph.Outputs)}");
```

---

## 附录 A：架构总览

```
┌──────────────────────────────────────────────────────────────┐
│                     用户调用入口                               │
│              OnnxExporter.Export() / ExportAsync()            │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│              Step 1: RegisterDefaultProcessors()              │
│            ModuleProcessorRegistry (单例注册表)                │
│  ┌─────────────────────────────────────────────────────┐     │
│  │ Type → INodeProcessor 映射 (100+ 内置处理器)          │     │
│  │ Name → INodeProcessor 映射 (名称模糊匹配)             │     │
│  └─────────────────────────────────────────────────────┘     │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│           Step 2: SymbolicTraceEngine.Trace()                 │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐  │
│  │ TraceContext │←→│DataFlowGraph │  │ ModuleProcessor    │  │
│  │ (值/形状管理) │  │ (图结构)     │  │ Registry (查找)    │  │
│  └──────────────┘  └──────────────┘  └────────────────────┘  │
│       ↓ Process()                                            │
│  ┌─────────────────────────────────────────────────────┐     │
│  │ INodeProcessor.Process() → DataFlowNode             │     │
│  └─────────────────────────────────────────────────────┘     │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│            Step 3: OnnxGraphBuilder.Build()                   │
│  DataFlowGraph ──转换──→ ModelProto (ONNX protobuf)          │
│       │                      │                                │
│       ▼                      ▼                                │
│  GraphProto              CodedOutputStream                   │
│  (NodeProto × N)         → .onnx 文件                        │
│  (TensorProto 权重)                                        │
└──────────────────────────────────────────────────────────────┘
```

## 附录 B：命名空间索引

| 命名空间 | 主要类型 |
|----------|----------|
| `TorchSharp.OnnxExporter` | `OnnxExporter`, `ModuleProcessorRegistry` |
| `TorchSharp.OnnxExporter.Modules` | `Operator`, `IOperator`, `Add`, `Sub`, `Mul`, `Div`, `MatMul`, `Pow`, `Sqrt`, `Exp`, `Log`, `AddWithBias`, `LinearOperator`, `Concat`, `Stack`, `ReshapeOp`, `TransposeOp`, `SqueezeOp`, `UnsqueezeOp`, `ClampOp`, `WhereOp`, `SumOp`, `MeanOp`, `OperatorBuilder` |
| `TorchSharp.OnnxExporter.Processors` | `INodeProcessor`, `BaseProcessor<T>`, `ElementWiseProcessor`, 以及 100+ 具体 Processor 类 |
| `TorchSharp.OnnxExporter.DataFlow` | `DataFlowGraph`, `DataFlowNode`, `TraceContext` |
| `TorchSharp.OnnxExporter.Builder` | `OnnxGraphBuilder` |
| `TorchSharp.OnnxExporter.Tracing` | `SymbolicTraceEngine` |

---

*文档生成时间: 2026-04-18 | 基于 TorchSharp.OnnxExporter v1.0.0 源码*
