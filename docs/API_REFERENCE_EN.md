# TorchSharp.OnnxExporter API Reference

> **Version**: 1.0.0 | **Namespace Root**: `TorchSharp.OnnxExporter` | **ONNX IR Version**: 7 | **Opset Version**: 14

---

## Table of Contents

- [1. OnnxExporter Class](#1-onnxexporter-class)
- [2. OperatorBuilder Class](#2-operatorbuilder-class)
- [3. ModuleProcessorRegistry Class](#3-moduleprocessorregistry-class)
- [4. INodeProcessor Interface and BaseProcessor Abstract Class](#4-inodeprocessor-interface-and-baseprocessor-abstract-class)
- [5. Stateful Operator Classes (Operator Class Hierarchy)](#5-stateful-operator-classes-operator-class-hierarchy)
- [6. Tensor Operation Operators (TensorOperators)](#6-tensor-operation-operators-tensoroperators)
- [7. Data Flow Classes](#7-data-flow-classes)
- [8. OnnxGraphBuilder Class](#8-onnxgraphbuilder-class)
- [9. SymbolicTraceEngine Class](#9-symbolictraceengine-class)

---

## 1. OnnxExporter Class

| Item | Description |
|------|-------------|
| **Namespace** | `TorchSharp.OnnxExporter` |
| **Type** | `static class` |
| **File** | [OnnxExporter.cs](../OnnxExporter.cs) |
| **Summary** | The unified entry point for the ONNX exporter, providing both synchronous and asynchronous methods to export TorchSharp models to ONNX format files. Internally uses a symbolic tracing engine (`SymbolicTraceEngine`) to build a data flow graph, then generates ONNX protobuf via `OnnxGraphBuilder` and writes it to file. |

### Static Methods

#### Export(Module, Tensor, string, string)

```csharp
public static void Export(
    Module model,
    Tensor dummyInput,
    string outputPath,
    string modelName = "model"
)
```

**Purpose**: Synchronously exports a model to an ONNX format file.

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | `Module` | The TorchSharp model to export (must not be null) |
| `dummyInput` | `Tensor` | Dummy input tensor used for shape inference and forward tracing (must not be null) |
| `outputPath` | `string` | Full path for the output ONNX file (must not be empty) |
| `modelName` | `string` | Model name; defaults to `"model"` |

**Exceptions**:

| Exception Type | Condition |
|----------------|-----------|
| `ArgumentNullException` | `model` is null, `dummyInput` is null, or `outputPath` is empty or null |
| `InvalidOperationException` | Symbolic tracing fails, graph construction fails, or processor registration fails |
| Wrapped `IOException` | File write failure (path does not exist, insufficient permissions, etc.) |

**Execution Flow**:
1. Calls `RegisterDefaultProcessors()` to register all built-in processors
2. Creates a `SymbolicTraceEngine` and executes `Trace()` to build the data flow graph
3. Creates an `OnnxGraphBuilder` and executes `Build()` to generate `ModelProto`
4. Uses `CodedOutputStream` to write the protobuf to file

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

**Purpose**: Asynchronously exports a model to an ONNX format file. Internal logic is identical to `Export()`, except that the final file write is wrapped with `Task.Run()` for asynchronous execution.

**Parameters**: Identical to `Export()`.

**Exceptions**: Identical to `Export()`.

### Usage Examples

```csharp
using TorchSharp;
using TorchSharp.OnnxExporter;
using static TorchSharp.torch.nn;

// 1. Define the model
var model = Sequential(
    Linear(784, 256),
    ReLU(),
    Dropout(0.2),
    Linear(256, 10)
);

// 2. Create dummy input
var dummyInput = torch.randn(1, 784);

// 3. Synchronous export
OnnxExporter.Export(model, dummyInput, "my_model.onnx", "MNISTClassifier");

// 4. Or asynchronous export
await OnnxExporter.ExportAsync(model, dummyInput, "my_model_async.onnx", "MNISTClassifier");
```

---

## 2. OperatorBuilder Class

| Item | Description |
|------|-------------|
| **Namespace** | `TorchSharp.OnnxExporter.Modules` |
| **Type** | `static class` |
| **File** | [OperatorBuilder.cs](../Modules/OperatorBuilder.cs) |
| **Summary** | Provides static factory methods following the Builder Pattern for creating parameterized stateful operator modules (e.g., `AddWithBias`, `LinearOperator`). |

### Static Factory Methods

#### CreateAddWithBias()

```csharp
public static AddWithBiasBuilder CreateAddWithBias()
```

**Returns**: `AddWithBiasBuilder` — A builder instance; after chained configuration, call `Build()` to produce an `AddWithBias` operator.

#### CreateLinearOperator()

```csharp
public static LinearOperatorBuilder CreateLinearOperator()
```

**Returns**: `LinearOperatorBuilder` — A builder instance; after chained configuration, call `Build()` to produce a `LinearOperator` operator.

---

### AddWithBiasBuilder Inner Class

Nested inside the `OperatorBuilder` class.

#### Bias(Tensor)

```csharp
public AddWithBiasBuilder Bias(Tensor bias)
```

**Purpose**: Sets the bias tensor. **Must be called**, otherwise `Build()` will throw an exception.

| Parameter | Type | Description |
|-----------|------|-------------|
| `bias` | `Tensor` | The bias tensor |

**Returns**: `AddWithBiasBuilder` (itself, supports method chaining)

#### Build()

```csharp
public AddWithBias Build()
```

**Returns**: `AddWithBias` — The fully configured add-bias operator module.

**Exception**: `InvalidOperationException` — Thrown when `Bias()` has not been called to set the bias, with message `"Bias is required for AddWithBias"`.

---

### LinearOperatorBuilder Inner Class

Nested inside the `OperatorBuilder` class.

#### Weight(Tensor)

```csharp
public LinearOperatorBuilder Weight(Tensor weight)
```

**Purpose**: Sets the weight tensor. **Must be called**, otherwise `Build()` will throw an exception.

| Parameter | Type | Description |
|-----------|------|-------------|
| `weight` | `Tensor` | The weight tensor |

**Returns**: `LinearOperatorBuilder` (itself, supports method chaining)

#### Bias(Tensor)

```csharp
public LinearOperatorBuilder Bias(Tensor bias)
```

**Purpose**: Sets an optional bias tensor. May be omitted (equivalent to a linear transformation without bias).

| Parameter | Type | Description |
|-----------|------|-------------|
| `bias` | `Tensor` | The bias tensor (optional) |

**Returns**: `LinearOperatorBuilder` (itself, supports method chaining)

#### Build()

```csharp
public LinearOperator Build()
```

**Returns**: `LinearOperator` — The fully configured linear operator module.

**Exception**: `InvalidOperationException` — Thrown when `Weight()` has not been called to set the weight, with message `"Weight is required for LinearOperator"`.

### Usage Examples

```csharp
using TorchSharp;
using TorchSharp.OnnxExporter.Modules;
using static TorchSharp.torch;

// Create an AddWithBias operator
var bias = torch.randn(10);
var addWithBias = OperatorBuilder.CreateAddWithBias()
    .Bias(bias)
    .Build();

// Create a LinearOperator operator
var weight = torch.randn(784, 256);
var linearOp = OperatorBuilder.CreateLinearOperator()
    .Weight(weight)
    .Bias(torch.randn(256))   // bias is optional
    .Build();
```

---

## 3. ModuleProcessorRegistry Class

| Item | Description |
|------|-------------|
| **Namespace** | `TorchSharp.OnnxExporter` |
| **Type** | `sealed class` (Singleton) |
| **File** | [ModuleProcessorRegistry.cs](../ModuleProcessorRegistry.cs) |
| **Summary** | Global registry for module processors, using the Singleton pattern to manage mappings from `Module` types to `INodeProcessor`. Supports both exact type matching and fuzzy name-based lookup strategies. Automatically initializes and registers all built-in processors (100+) on first access. |

### Static Properties

#### Instance

```csharp
public static ModuleProcessorRegistry Instance { get; }
```

**Description**: Retrieves the globally unique singleton instance. Thread-safe (`readonly static` initialization).

---

### Static Methods

#### Register\<TModule\>(INodeProcessor)

```csharp
public static void Register<TModule>(INodeProcessor processor)
    where TModule : TorchSharp.torch.nn.Module
```

**Purpose**: Registers a custom processor by module type.

| Parameter | Type | Description |
|-----------|------|-------------|
| `processor` | `INodeProcessor` | The processor implementation |

**Type Parameter** `TModule`: The target module type (must inherit from `Module`).

---

#### RegisterByName(string, INodeProcessor)

```csharp
public static void RegisterByName(string moduleName, INodeProcessor processor)
```

**Purpose**: Registers a processor by module name string (for scenarios where direct type matching is not possible).

| Parameter | Type | Description |
|-----------|------|-------------|
| `moduleName` | `string` | Module name (e.g., `"Concat"`, `"Reshape"`) |
| `processor` | `INodeProcessor` | The processor implementation |

---

#### RegisterDefaultProcessors()

```csharp
public static void RegisterDefaultProcessors()
```

**Purpose**: Registers all built-in processors, including:

| Category | Registered Processors |
|----------|----------------------|
| Convolution Layers | Conv1d, Conv2d, Conv3d, ConvTranspose1d, ConvTranspose2d, ConvTranspose3d |
| Fully Connected Layers | Linear, LinearOperator, Bilinear |
| Activation Functions | ReLU, ReLU6, LeakyReLU, PReLU, RReLU, ELU, CELU, SELU, Mish, SiLU, GELU, Sigmoid, Tanh, Softmax, Softmax2d, Softmin, LogSoftmax, Softplus, Hardswish, Hardsigmoid, Hardtanh, Hardshrink, Softshrink, Softsign, LogSigmoid, Tanhshrink, Threshold |
| Normalization | BatchNorm1d/2d/3d, LayerNorm, GroupNorm, InstanceNorm1d/2d/3d, LocalResponseNorm |
| Pooling Layers | MaxPool1d/2d/3d, AvgPool1d/2d/3d, AdaptiveAvgPool1d/2d/3d, AdaptiveMaxPool1d/2d/3d, FractionalMaxPool2d/3d, MaxUnpool1d/2d/3d, LPPool1d/2d |
| Dropout | Dropout, Dropout1d, Dropout2d, Dropout3d, AlphaDropout |
| Padding | ConstantPad1d/2d/3d, ReflectionPad1d/2d/3d, ReplicationPad1d/2d/3d, ZeroPad2d |
| Shape Transformation | Flatten, Reshape, Transpose, Squeeze, Unsqueeze, Concat, Stack, Chunk, Split |
| Tensor Operations | Fold, Unfold, PixelShuffle, PixelUnuffle, Upsample, ChannelShuffle |
| Embedding | Embedding, EmbeddingBag |
| RNN Family | RNN, RNNCell, LSTM, LSTMCell, GRU, GRUCell |
| Attention | MultiheadAttention |
| Custom Operators | Add, Sub, Mul, Div, MatMul, Pow, Sqrt, Exp, Log, AddWithBias, Concat, Stack, ReshapeOp, TransposeOp, SqueezeOp, UnsqueezeOp, ClampOp, WhereOp, SumOp, MeanOp |
| Other | Identity, OneHot, PairwiseDistance, CosineSimilarity |

---

### Instance Methods

#### GetProcessor(Module)

```csharp
public INodeProcessor? GetProcessor(TorchSharp.torch.nn.Module module)
```

**Purpose**: Looks up the corresponding processor for a given module instance. Lookup order:
1. Exact match against `_processors` dictionary by `module.GetType()`
2. If no match, attempt base-type match via `IsAssignableFrom`
3. If still no match, look up in `_nameProcessors` dictionary by `module.GetType().Name`

| Parameter | Type | Description |
|-----------|------|-------------|
| `module` | `Module` | The module instance |

**Returns**: `INodeProcessor?` — The found processor, or `null` if no match exists.

---

#### GetProcessor(Type)

```csharp
public INodeProcessor? GetProcessor(Type moduleType)
```

**Purpose**: Looks up a processor by module type. Lookup logic is the same as above, but skips the name-matching step.

| Parameter | Type | Description |
|-----------|------|-------------|
| `moduleType` | `Type` | The `Type` object of the module |

**Returns**: `INodeProcessor?` — The found processor, or `null` if no match exists.

---

#### CanProcess(Module)

```csharp
public bool CanProcess(TorchSharp.torch.nn.Module module)
```

**Purpose**: Determines whether a processor is available for the specified module.

| Parameter | Type | Description |
|-----------|------|-------------|
| `module` | `Module` | The module instance |

**Returns**: `bool` — Returns `true` if a corresponding processor exists, otherwise `false`.

---

### Static Convenience Methods

#### GetProcessorStatic(Module)

```csharp
public static INodeProcessor? GetProcessorStatic(TorchSharp.torch.nn.Module module)
```

**Description**: Static wrapper around `Instance.GetProcessor(module)`.

#### CanProcessStatic(Module)

```csharp
public static bool CanProcessStatic(TorchSharp.torch.nn.Module module)
```

**Description**: Static wrapper around `Instance.CanProcess(module)`.

### Usage Examples

```csharp
using TorchSharp;
using TorchSharp.OnnxExporter;
using TorchSharp.OnnxExporter.Processors;
using static TorchSharp.torch.nn;

// Query whether an existing module is supported
var linear = Linear(10, 20);
bool supported = ModuleProcessorRegistry.CanProcessStatic(linear); // true

// Register a custom processor
ModuleProcessorRegistry.Register<MyCustomModule>(new MyCustomProcessor());

// Register by name
ModuleProcessorRegistry.RegisterByName("MyCustomOp", new MyCustomProcessor());
```

---

## 4. INodeProcessor Interface and BaseProcessor Abstract Class

### 4.1 INodeProcessor Interface

| Item | Description |
|------|-------------|
| **Namespace** | `TorchSharp.OnnxExporter.Processors` |
| **Type** | `interface` |
| **File** | [INodeProcessor.cs](../Processors/INodeProcessor.cs) |
| **Summary** | Node processor interface that defines the contract for converting a TorchSharp `Module` into an ONNX `DataFlowNode`. All module processors must implement this interface. |

#### Properties

##### OpType

```csharp
string OpType { get; }
```

**Description**: The ONNX operation type name generated by this processor (e.g., `"Conv"`, `"Relu"`, `"MatMul"`), corresponding to the `op_type` field in the ONNX specification.

---

#### Methods

##### CanProcess(Module)

```csharp
bool CanProcess(Module module)
```

**Purpose**: Determines whether this processor can handle the given module instance.

| Parameter | Type | Description |
|-----------|------|-------------|
| `module` | `Module` | The module instance to evaluate |

**Returns**: `bool` — Returns `true` if it can process, otherwise `false`.

---

##### Process(Module, TraceContext)

```csharp
DataFlowNode Process(Module module, TraceContext context)
```

**Purpose**: Processes the module, generates the corresponding data flow node, and adds it to the context's graph.

| Parameter | Type | Description |
|-----------|------|-------------|
| `module` | `Module` | The module instance to process |
| `context` | `TraceContext` | Tracing context providing current input/output names, graph reference, shape information, etc. |

**Returns**: `DataFlowNode` — The generated data flow node.

---

### 4.2 BaseProcessor\<TModule\> Abstract Class

| Item | Description |
|------|-------------|
| **Namespace** | `TorchSharp.OnnxExporter.Processors` |
| **Type** | `abstract class` : `INodeProcessor` |
| **File** | [BaseProcessor.cs](../Processors/BaseProcessor.cs) |
| **Summary** | Generic abstract base class that simplifies processor implementation. Automatically handles type checking and casting; subclasses only need to focus on processing logic specific to their target module type. |

```csharp
public abstract class BaseProcessor<TModule> : INodeProcessor
    where TModule : Module
```

**Implemented Members**:

| Member | Implementation |
|--------|---------------|
| `CanProcess(Module)` | Returns `module is TModule` |
| `Process(Module, TraceContext)` | Casts `module` to `TModule` then calls the generic version |

**Abstract Members Subclasses Must Implement**:

| Member | Signature |
|--------|----------|
| `OpType` | `string OpType { get; }` |
| `Process` | `DataFlowNode Process(TModule module, TraceContext context)` |

---

### 4.3 ElementWiseProcessor Class

| Item | Description |
|------|-------------|
| **Namespace** | `TorchSharp.OnnxExporter.Processors` |
| **Type** | `class` : `INodeProcessor` |
| **File** | [ElementWiseProcessor.cs](../Processors/ElementWiseProcessor.cs) |
| **Summary** | Generic element-wise operation processor that accepts an `opType` via its constructor and can be used for various single-input element-wise operators (e.g., Sqrt, Exp, Log, Clamp, Where, ReduceSum, ReduceMean). |

#### Constructor

```csharp
public ElementWiseProcessor(string opType)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `opType` | `string` | The ONNX operation type name |

#### Process Behavior

Retrieves the current value from `TraceContext` as input, creates a temporary output name, constructs a `DataFlowNode` and adds it to the graph, then updates the current value to the output name.

### Usage Examples

```csharp
using TorchSharp.OnnxExporter.Processors;
using TorchSharp.OnnxExporter.DataFlow;

// Custom processor implementation (recommended: use BaseProcessor)
public class MyConvProcessor : BaseProcessor<Conv2d>
{
    public override string OpType => "Conv";

    public override DataFlowNode Process(Conv2d module, TraceContext context)
    {
        var input = context.GetCurrentValue();
        var output = context.CreateTempName();

        var node = new DataFlowNode(OpType, new[] { input }, new[] { output });
        node.Attributes["kernel_shape"] = new int[] { module.kernel_size[0], module.kernel_size[1] };
        // ... more attribute settings ...

        context.Graph.AddNode(node);
        context.SetCurrentValue(output);
        return node;
    }
}

// Use ElementWiseProcessor for generic element-wise operations
var sqrtProcessor = new ElementWiseProcessor("Sqrt");
var expProcessor = new ElementWiseProcessor("Exp");
```

---

## 5. Stateful Operator Classes (Operator Class Hierarchy)

| Item | Description |
|------|-------------|
| **Namespace** | `TorchSharp.OnnxExporter.Modules` |
| **File** | [Operators.cs](../Modules/Operators.cs) |
| **Summary** | Provides a set of traceable operator modules that can be used within TorchSharp models. These operators inherit from `Module` and can be embedded in `Sequential` or custom modules just like standard TorchSharp layers, and are correctly recognized and converted by the ONNX exporter. |

### 5.1 IOperator Interface

```csharp
public interface IOperator
{
    Tensor forward(params Tensor[] inputs);
}
```

**Description**: Core interface for operators, defining the forward computation signature. All concrete operators implement this interface.

| Member | Signature | Description |
|--------|-----------|-------------|
| `forward` | `Tensor forward(params Tensor[] inputs)` | Forward computation method accepting a variable number of tensor inputs and returning the result tensor |

---

### 5. Operator Abstract Class

```csharp
public abstract class Operator : Module
```

**Inheritance**: `Operator` → `Module`

**Description**: Abstract base class for all stateful operators. Inherits from `Module` so it can be embedded as a submodule in TorchSharp models; also requires subclasses to implement the `IOperator.forward()` method.

#### Constructor

```csharp
protected Operator(string name)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `string` | Operator name (passed to base class `Module`) |

#### Abstract Method

```csharp
public abstract Tensor forward(params Tensor[] inputs);
```

---

### Concrete Operator Reference

All operator classes below are located in the `TorchSharp.OnnxExporter.Modules` namespace and inherit from `Operator`.

#### Add — Element-wise Addition

| Item | Value |
|------|-------|
| **Constructor** | `public Add()` |
| **OpType** | `"Add"` |
| **forward Logic** | Sequentially applies `torch.add()` on `inputs[0..N]`; supports two or more inputs |
| **Input Requirement** | At least 2 Tensors |
| **Exception** | `ArgumentException` — Thrown when fewer than 2 inputs are provided |

```csharp
var add = new Add();
var result = add.forward(a, b, c); // a + b + c
```

---

#### Sub — Element-wise Subtraction

| Item | Value |
|------|-------|
| **Constructor** | `public Sub()` |
| **OpType** | `"Sub"` |
| **forward Logic** | Sequentially applies `torch.sub()` on `inputs[0..N]` |
| **Input Requirement** | At least 2 Tensors |
| **Exception** | `ArgumentException` — Thrown when fewer than 2 inputs are provided |

---

#### Mul — Element-wise Multiplication

| Item | Value |
|------|-------|
| **Constructor** | `public Mul()` |
| **OpType** | `"Mul"` |
| **forward Logic** | Sequentially applies `torch.mul()` on `inputs[0..N]` |
| **Input Requirement** | At least 2 Tensors |
| **Exception** | `ArgumentException` — Thrown when fewer than 2 inputs are provided |

---

#### Div — Element-wise Division

| Item | Value |
|------|-------|
| **Constructor** | `public Div()` |
| **OpType** | `"Div"` |
| **forward Logic** | Sequentially applies `torch.div()` on `inputs[0..N]` |
| **Input Requirement** | At least 2 Tensors |
| **Exception** | `ArgumentException` — Thrown when fewer than 2 inputs are provided |

---

#### MatMul — Matrix Multiplication

| Item | Value |
|------|-------|
| **Constructor** | `public MatMul()` |
| **OpType** | `"MatMul"` |
| **forward Logic** | Sequentially applies `torch.matmul()` on `inputs[0..N]` |
| **Input Requirement** | At least 2 Tensors |
| **Exception** | `ArgumentException` — Thrown when fewer than 2 inputs are provided |

---

#### Pow — Power Operation

| Item | Value |
|------|-------|
| **Constructor** | `public Pow()` |
| **OpType** | `"Pow"` |
| **forward Logic** | `torch.pow(inputs[0], inputs[1])`; only uses the first two inputs |
| **Input Requirement** | At least 2 Tensors (base and exponent) |
| **Exception** | `ArgumentException` — Thrown when fewer than 2 inputs are provided |

---

#### Sqrt — Square Root

| Item | Value |
|------|-------|
| **Constructor** | `public Sqrt()` |
| **OpType** | `"Sqrt"` |
| **forward Logic** | `torch.sqrt(inputs[0])`; only uses the first input |
| **Input Requirement** | At least 1 Tensor |
| **Exception** | `ArgumentException` — Thrown when no input is provided |

---

#### Exponential Operation

| Item | Value |
|------|-------|
| **Constructor** | `public Exp()` |
| **OpType** | `"Exp"` |
| **forward Logic** | `torch.exp(inputs[0])` |
| **Input Requirement** | At least 1 Tensor |
| **Exception** | `ArgumentException` — Thrown when no input is provided |

---

#### Log — Logarithm Operation

| Item | Value |
|------|-------|
| **Constructor** | `public Log()` |
| **OpType** | `"Log"` |
| **forward Logic** | `torch.log(inputs[0])` |
| **Input Requirement** | At least 1 Tensor |
| **Exception** | `ArgumentException` — Thrown when no input is provided |

---

#### AddWithBias — Addition with Bias

| Item | Value |
|------|-------|
| **Constructor** | `public AddWithBias(Tensor bias)` |
| **OpType** | — |
| **Field** | `public readonly Tensor bias` |
| **forward Logic** | `torch.add(inputs[0], bias)` |
| **Input Requirement** | At least 1 Tensor |
| **Exception** | `ArgumentException` — Thrown when no input is provided |

```csharp
var addWithBias = new AddWithBias(torch.randn(10));
var result = addWithBias.forward(x); // x + bias
```

---

#### LinearOperator — Linear Transformation Operator

| Item | Value |
|------|-------|
| **Constructor** | `public LinearOperator(Tensor weight, Tensor? bias = null)` |
| **Fields** | `public readonly Tensor weight`, `public readonly Tensor? bias` |
| **forward Logic** | `matmul(inputs[0], weight)` + optional `add(output, bias)` |
| **Input Requirement** | At least 1 Tensor |
| **Exception** | `ArgumentException` — Thrown when no input is provided |

```csharp
var linearOp = new LinearOperator(
    weight: torch.randn(784, 256),
    bias: torch.randn(256)     // Optional; pass null for no bias
);
var result = linearOp.forward(x); // x @ weight + bias
```

### Usage Example — Composing Operators in Sequential

```csharp
using TorchSharp;
using TorchSharp.OnnxExporter.Modules;
using static TorchSharp.torch.nn;

var model = Sequential(
    Linear(784, 128),
    new ReLU(),
    new Mul(),           // Custom operator: use with residual connections for multi-branch input
    new AddWithBias(torch.randn(128)),
    Linear(128, 10)
);

var input = torch.randn(1, 784);
OnnxExporter.Export(model, input, "custom_model.onnx", "CustomModel");
```

---

## 6. Tensor Operation Operators (TensorOperators)

| Item | Description |
|------|-------------|
| **Namespace** | `TorchSharp.OnnxExporter.Modules` |
| **File** | [TensorOperators.cs](../Modules/TensorOperators.cs) |
| **Summary** | Provides operator modules for tensor shape transformation and reduction operations. Each operator receives operation parameters at construction time (e.g., dimensions, shapes) and performs the corresponding TorchSharp operation on input tensors during `forward`. All classes inherit from `Operator`. |

### Concat — Tensor Concatenation

| Item | Value |
|------|-------|
| **Constructor** | `public Concat(int dim = 0)` |
| **Public Field** | `public int dim` |
| **OpType** | `"Concat"` |
| **forward Logic** | `torch.cat(inputs, dim)` |
| **Input Requirement** | At least 1 Tensor |
| **Exception** | `ArgumentException` — Thrown when no input is provided |

```csharp
var concat = new Concat(dim: 1);
var result = concat.forward(tensorA, tensorB, tensorC); // concatenate along dim=1
```

---

### Stack — Tensor Stacking

| Item | Value |
|------|-------|
| **Constructor** | `public Stack(int dim = 0)` |
| **Public Field** | `public int dim` |
| **OpType** | — |
| **forward Logic** | `torch.stack(inputs, dim)` |
| **Input Requirement** | At least 1 Tensor |
| **Exception** | `ArgumentException` — Thrown when no input is provided |

---

### ReshapeOp — Shape Reshaping

| Item | Value |
|------|-------|
| **Constructor** | `public ReshapeOp(params long[] shape)` |
| **Public Field** | `public long[] shape` |
| **OpType** | `"Reshape"` |
| **forward Logic** | `torch.reshape(inputs[0], shape)` |
| **Input Requirement** | At least 1 Tensor |
| **Exception** | `ArgumentException` — Thrown when no input is provided |

```csharp
var reshape = new ReshapeOp(-1, 256);  // flatten then reshape to (N, 256)
var result = reshape.forward(x);
```

---

### TransposeOp — Dimension Transposition

| Item | Value |
|------|-------|
| **Constructor** | `public TransposeOp(int dim0 = 0, int dim1 = 1)` |
| **Public Fields** | `public int dim0`, `public int dim1` |
| **OpType** | `"Transpose"` |
| **forward Logic** | `torch.transpose(inputs[0], dim0, dim1)` |
| **Input Requirement** | At least 1 Tensor |
| **Exception** | `ArgumentException` — Thrown when no input is provided |

```csharp
var transpose = new TransposeOp(0, 2);  // swap dimension 0 and dimension 2
var result = transpose.forward(x);
```

---

### SqueezeOp — Dimension Squeezing

| Item | Value |
|------|-------|
| **Constructor** | `public SqueezeOp(int dim = -1)` |
| **Public Field** | `public int dim` |
| **OpType** | `"Squeeze"` |
| **forward Logic** | When `dim >= 0`: calls `torch.squeeze(inputs[0], dim)`; otherwise calls `torch.squeeze(inputs[0])` to squeeze all dimensions of size 1 |
| **Input Requirement** | At least 1 Tensor |
| **Exception** | `ArgumentException` — Thrown when no input is provided |

---

### UnsqueezeOp — Dimension Insertion

| Item | Value |
|------|-------|
| **Constructor** | `public UnsqueezeOp(int dim = 0)` |
| **Public Field** | `public int dim` |
| **OpType** | `"Unsqueeze"` |
| **forward Logic** | `torch.unsqueeze(inputs[0], dim)` |
| **Input Requirement** | At least 1 Tensor |
| **Exception** | `ArgumentException` — Thrown when no input is provided |

---

### ClampOp — Value Clipping

| Item | Value |
|------|-------|
| **Constructor** | `public ClampOp(double? min = null, double? max = null)` |
| **Public Fields** | `public double? min`, `public double? max` |
| **OpType** | `"Clamp"` |
| **forward Logic** | Calls different overloads of `torch.clamp()` depending on whether min/max are provided; returns input unchanged if both are null |
| **Input Requirement** | At least 1 Tensor |
| **Exception** | `ArgumentException` — Thrown when no input is provided |

```csharp
var clamp = new ClampOp(min: 0.0, max: 1.0);  // clip to [0, 1]
var result = clamp.forward(x);
```

---

### WhereOp — Conditional Selection

| Item | Value |
|------|-------|
| **Constructor** | `public WhereOp()` |
| **OpType** | `"Where"` |
| **forward Logic** | `torch.where(inputs[0], inputs[1], inputs[2])` |
| **Input Requirement** | **Exactly 3** Tensors (condition, x, y) |
| **Exception** | `ArgumentException` — Thrown when input count is not 3 |

```csharp
var whereOp = new WhereOp();
var result = whereOp.forward(condition, x, y);  // select x where condition is true, else y
```

---

### SumOp — Sum Reduction

| Item | Value |
|------|-------|
| **Constructor** | `public SumOp(int? dim = null, bool keepdim = false)` |
| **Public Fields** | `public int? dim`, `public bool keepdim` |
| **OpType** | `"ReduceSum"` |
| **forward Logic** | When `dim` is provided: `torch.sum(inputs[0], dim, keepdim)`; otherwise: `torch.sum(inputs[0])` for global sum |
| **Input Requirement** | At least 1 Tensor |
| **Exception** | `ArgumentException` — Thrown when no input is provided |

---

### MeanOp — Mean Reduction

| Item | Value |
|------|-------|
| **Constructor** | `public MeanOp(int? dim = null, bool keepdim = false)` |
| **Public Fields** | `public int? dim`, `public bool keepdim` |
| **OpType** | `"ReduceMean"` |
| **forward Logic** | When `dim` is provided: `torch.mean(inputs[0], new long[] { dim }, keepdim)`; otherwise: `torch.mean(inputs[0])` for global mean |
| **Input Requirement** | At least 1 Tensor |
| **Exception** | `ArgumentException` — Thrown when no input is provided |

### Usage Examples

```csharp
using TorchSharp.OnnxExporter.Modules;

// Shape transformation pipeline
var pipeline = new Modules.ReshapeOp(-1, 768)
    .forward(
        new Modules.Concat(dim: 0).forward(part1, part2),
        new Modules.SqueezeOp(dim: 0).forward(extra)
    );

// Reduction operations
var sumResult = new Modules.SumOp(dim: 1, keepdim: true).forward(features);
var meanResult = new Modules.MeanOp().forward(features);  // global mean
```

---

## 7. Data Flow Classes

### 7.1 DataFlowGraph Class

| Item | Description |
|------|-------------|
| **Namespace** | `TorchSharp.OnnxExporter.DataFlow` |
| **Type** | `class` |
| **File** | [DataFlowGraph.cs](../DataFlow/DataFlowGraph.cs) |
| **Summary** | In-memory representation of a data flow graph, built incrementally during symbolic tracing. Contains node lists, input/output lists, initializers (weights), and shape information at each stage. Can ultimately be converted to an ONNX `GraphProto` via `ToGraphProto()`. |

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `Nodes` | `List<DataFlowNode>` | Ordered list of all data flow nodes in the graph |
| `Inputs` | `List<string>` | List of graph input tensor names |
| `Outputs` | `List<string>` | List of graph output tensor names |
| `Initializers` | `List<TensorProto>` | List of ONNX TensorProto objects for model weights/constants |
| `InputShapes` | `Dictionary<string, List<long>>` | Mapping from input tensor name to shape |
| `OutputShapes` | `Dictionary<string, List<long>>` | Mapping from output tensor name to shape |
| `IntermediateShapes` | `Dictionary<string, List<long>>` | Mapping from intermediate tensor name to shape |

---

#### Methods

##### AddNode(DataFlowNode)

```csharp
public DataFlowNode AddNode(DataFlowNode node)
```

**Purpose**: Adds a data flow node to the graph.

| Parameter | Type | Description |
|-----------|------|-------------|
| `node` | `DataFlowNode` | The node to add |

**Returns**: `DataFlowNode` — Returns the passed-in node reference (convenient for chaining).

---

##### AddInitializer(string, Tensor)

```csharp
public void AddInitializer(string name, Tensor tensor)
```

**Purpose**: Creates an ONNX initializer (weight) from a TorchSharp tensor. Automatically infers data type (Float32/Double) and shape. Duplicate initializers with the same name will not be added.

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `string` | Initializer name |
| `tensor` | `Tensor` | TorchSharp tensor (may be null, in which case no action is taken) |

**Data Type Mapping**:
- `ScalarType.Float32` → `TensorProto.Types.DataType.Float`
- Other types → `TensorProto.Types.DataType.Double`

---

##### AddInitializer(string, float[], long[])

```csharp
public void AddInitializer(string name, float[] data, long[] dims)
```

**Purpose**: Creates an ONNX initializer from a float array and dimension array.

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `string` | Initializer name |
| `data` | `float[]` | Float data (may be null) |
| `dims` | `long[]` | Dimension sizes |

---

##### AddInitializer(string, long[])

```csharp
public void AddInitializer(string name, long[] data)
```

**Purpose**: Creates an Int64-typed ONNX initializer from an integer array (commonly used for storing shape information).

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `string` | Initializer name |
| `data` | `long[]` | Integer data (may be null) |

---

##### AddInput(string, Tensor)

```csharp
public void AddInput(string name, Tensor tensor)
```

**Purpose**: Registers a graph input while recording its shape information.

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `string` | Input name |
| `tensor` | `Tensor` | Input tensor (used to obtain shape; may be null) |

---

##### AddOutput(string, Tensor)

```csharp
public void AddOutput(string name, Tensor tensor)
```

**Purpose**: Registers a graph output while recording its shape information.

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `string` | Output name |
| `tensor` | `Tensor` | Output tensor (used to obtain shape; may be null) |

---

##### AddIntermediateShape(string, List\<long\>)

```csharp
public void AddIntermediateShape(string name, List<long> shape)
```

**Purpose**: Records shape information for intermediate tensors (used to generate ValueInfo).

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `string` | Intermediate tensor name |
| `shape` | `List<long>` | Shape list (may be null) |

---

##### ToGraphProto()

```csharp
public GraphProto ToGraphProto()
```

**Purpose**: Converts the entire data flow graph to an ONNX `GraphProto`. Processing includes:
- Building Input ValueInfoProto (including shape and dynamic dimension markers)
- Converting all `DataFlowNode` instances to NodeProto
- Building Output ValueInfoProto (including shape information)
- Adding Initializer list

**Returns**: `GraphProto` — ONNX graph protocol buffer object.

**Dimension Handling Rules**:
- `dim > 0`: Use `DimValue` set to fixed value
- `dim <= 0`: Use `DimParam` set to `"dynamic"`

---

### 7.2 DataFlowNode Class

| Item | Description |
|------|-------------|
| **Namespace** | `TorchSharp.OnnxExporter.DataFlow` |
| **Type** | `class` |
| **File** | [DataFlowNode.cs](../DataFlow/DataFlowNode.cs) |
| **Summary** | Representation of a single node in the data flow graph, corresponding to one computation node (NodeProto) in the ONNX specification. Contains operation type, input/output names, attribute dictionary, etc. |

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `Name` | `string` | Unique node name in format `{OpType}_{GUID}`; read-write |
| `OpType` | `string` | ONNX operation type (e.g., `"Conv"`, `"Relu"`); read-only |
| `Inputs` | `IReadOnlyList<string>` | List of input tensor names; read-only |
| `Outputs` | `IReadOnlyList<string>` | List of output tensor names; read-only |
| `Attributes` | `Dictionary<string, object>` | Node attribute dictionary; read-write |

---

#### Constructor

```csharp
public DataFlowNode(
    string opType,
    IEnumerable<string> inputs,
    IEnumerable<string> outputs
)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `opType` | `string` | Operation type (must not be null) |
| `inputs` | `IEnumerable<string>` | Collection of input names (may be null, treated as empty list) |
| `outputs` | `IEnumerable<string>` | Collection of output names (may be null, treated as empty list) |

**Exception**: `ArgumentNullException` — Thrown when `opType` is null.

**Automatic Behavior**: Upon construction, automatically generates a unique name in `{OpType}_{GUID}` format and initializes an empty Attributes dictionary.

---

#### Methods

##### ToNodeProto()

```csharp
public NodeProto ToNodeProto()
```

**Purpose**: Converts this data flow node to an ONNX `NodeProto`.

**Returns**: `NodeProto` — An ONNX node containing OpType, Name, Inputs, Outputs, and all attributes.

**Supported Attribute Type Conversions**:

| C# Type | ONNX AttributeType |
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
| Unsupported types | Skipped (warning log emitted) |

---

### 7.3 TraceContext Class

| Item | Description |
|------|-------------|
| **Namespace** | `TorchSharp.OnnxExporter.DataFlow` |
| **Type** | `class` |
| **File** | [TraceContext.cs](../DataFlow/TraceContext.cs) |
| **Summary** | Context object for the symbolic tracing process, persisting throughout the entire tracing lifecycle. Maintains value name mapping, shape information, scope stack, current value pointer, and other state for processors to query and update within their `Process()` methods. |

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `Graph` | `DataFlowGraph?` | Reference to the associated data flow graph (can be set externally) |
| `Values` | `Dictionary<string, string>` | Mapping from user variable names to ONNX temporary names |
| `NamedValues` | `Dictionary<string, string>` | Named value storage (e.g., scope, current_input, etc.) |
| `ValueShapes` | `Dictionary<string, List<long>>` | Mapping from ONNX temporary names to shapes |

---

#### Constructors

```csharp
public TraceContext()
```
Creates an empty tracing context.

```csharp
public TraceContext(DataFlowGraph graph)
```
Creates a tracing context and associates it with the specified data flow graph.

---

#### Methods

##### AddValue(string, Tensor)

```csharp
public string AddValue(string name, Tensor value)
```

**Purpose**: Registers a value, assigning a unique ONNX temporary name (format `tmp_N`) and recording its shape.

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `string` | User variable name |
| `value` | `Tensor` | Tensor value (used to obtain shape) |

**Returns**: `string` — The assigned ONNX temporary name.

---

##### GetValue(string)

```csharp
public string GetValue(string name)
```

**Purpose**: Looks up the ONNX name corresponding to a user variable name. Returns the original name itself if not found.

**Returns**: `string` — The ONNX name or the original name.

---

##### SetNamedValue(string, string) / GetNamedValue(string)

```csharp
public void SetNamedValue(string key, string value)
public string GetNamedValue(string key)
```

**Purpose**: Stores/retrieves named key-value pairs (used for storing metadata such as scope). `GetNamedValue` returns the key itself when not found.

---

##### CreateTempName()

```csharp
public string CreateTempName()
```

**Purpose**: Generates a new unique temporary name (format `tmp_0`, `tmp_1`, ..., global incrementing counter).

**Returns**: `string` — The new temporary name.

---

##### GetCurrentValue() / SetCurrentValue(string)

```csharp
public string GetCurrentValue()
public void SetCurrentValue(string value)
```

**Purpose**: Manages the "current value" pointer — i.e., the ONNX name of the previous operation's output. Processors typically read `GetCurrentValue()` as input and update via `SetCurrentValue()` after processing.
`GetCurrentValue()` returns the most recently created temporary name if none has been explicitly set.

---

##### GetNextValue()

```csharp
public string GetNextValue()
```

**Purpose**: Previews the next temporary name to be assigned (without consuming the counter). Useful for planning output names in advance.

**Returns**: `string` — The next temporary name (e.g., `tmp_5`).

---

##### SetOriginalInput(string) / GetOriginalInput()

```csharp
public void SetOriginalInput(string value)
public string? GetOriginalInput()
```

**Purpose**: Records and retrieves the original input name for the entire tracing process. Used for backtracking to the initial input in scenarios such as residual connections.

---

##### PushScope(string) / PopScope()

```csharp
public void PushScope(string scopeName)
public void PopScope()
```

**Purpose**: Scope stack management. `PushScope` stores the current scope name in NamedValues (key=`"current_scope"`); `PopScope` clears it.

---

##### SetShape(string, List\<long\>) / TryGetShape(string, out List\<long\>)

```csharp
public void SetShape(string name, List<long> shape)
public bool TryGetShape(string name, out List<long> shape)
```

**Purpose**: Stores/retrieves tensor shape information. `TryGetShape` indicates whether a match was found.

### Usage Examples

```csharp
using TorchSharp.OnnxExporter.DataFlow;

var graph = new DataFlowGraph();
var ctx = new TraceContext(graph);

// Register input
ctx.SetNamedValue("input", "input");
ctx.SetShape("input", new List<long> { 1, 3, 224, 224 });

// Simulate a processor workflow
ctx.SetCurrentValue("input");                    // Set current input
var outputName = ctx.CreateTempName();            // Pre-allocate output name tmp_0
var node = new DataFlowNode("Conv", new[] { ctx.GetCurrentValue() }, new[] { outputName });
graph.AddNode(node);
ctx.SetCurrentValue(outputName);                  // Advance current value
ctx.SetShape(outputName, new List<long> { 1, 64, 112, 112 });

// Scope management
ctx.PushScope("block_1");
// ... sub-operations ...
ctx.PopScope();
```

---

## 8. OnnxGraphBuilder Class

| Item | Description |
|------|-------------|
| **Namespace** | `TorchSharp.OnnxExporter.Builder` |
| **Type** | `sealed class` |
| **File** | [OnnxGraphBuilder.cs](../Builder/OnnxGraphBuilder.cs) |
| **Summary** | ONNX graph builder responsible for converting a `DataFlowGraph` (in-memory data flow graph representation) into a standard ONNX `ModelProto` (protobuf serialized format). Acts as the bridge between internal representation and external format in the export pipeline. |

### Constructor

```csharp
public OnnxGraphBuilder()
```

**Initialization Contents**:

| Field | Initial Value |
|-------|---------------|
| `_model.IrVersion` | `7` |
| `_model.ProducerName` | `"TorchSharp.OnnxExporter"` |
| `_model.ProducerVersion` | `"1.0.0"` |
| Opset Import (ai.onnx) | Version = 14 |
| Opset Import (empty domain) | Version = 14 |

---

### Public Methods

#### AddInput(ValueInfoProto)

```csharp
public void AddInput(ValueInfoProto input)
```

**Purpose**: Manually adds input tensor information.

| Parameter | Type | Description |
|-----------|------|-------------|
| `input` | `ValueInfoProto` | ONNX input value info |

**Exception**: `ArgumentNullException` — Thrown when `input` is null.

> **Note**: In the actual export workflow, inputs are primarily auto-populated from `DataFlowGraph.Inputs` via the `Build()` method. This method is intended mainly for advanced customization scenarios.

---

#### AddInitializer(TensorProto)

```csharp
public void AddInitializer(TensorProto initializer)
```

**Purpose**: Manually adds an initializer (model weight).

| Parameter | Type | Description |
|-----------|------|-------------|
| `initializer` | `TensorProto` | ONNX tensor prototype |

**Exception**: `ArgumentNullException` — Thrown when `initializer` is null.

---

#### AddOutput(ValueInfoProto)

```csharp
public void AddOutput(ValueInfoProto output)
```

**Purpose**: Manually adds output tensor information.

| Parameter | Type | Description |
|-----------|------|-------------|
| `output` | `ValueInfoProto` | ONNX output value info |

**Exception**: `ArgumentNullException` — Thrown when `output` is null.

---

#### Build(DataFlowGraph, string)

```csharp
public ModelProto Build(DataFlowGraph dataFlow, string modelName)
```

**Purpose**: Core method — fully converts a data flow graph to an ONNX ModelProto.

| Parameter | Type | Description |
|-----------|------|-------------|
| `dataFlow` | `DataFlowGraph` | The data flow graph to convert |
| `modelName` | `string` | Model name (defaults to `"TorchSharpModel"` if empty) |

**Returns**: `ModelProto` — Complete ONNX model protocol buffer.

**Exceptions**:

| Exception Type | Condition |
|----------------|-----------|
| `ArgumentNullException` | `dataFlow` is null |
| `InvalidOperationException` | Error occurs during graph construction (includes detailed error information and troubleshooting suggestions) |

**Internal Processing Flow**:

```
1. Process Inputs
   ├─ Skip empty names
   ├─ Build ValueInfoProto (with Float type and shape info)
   └─ Mark dynamic dimensions (dim <= 0 → "dynamic")

2. Process Initializers
   └─ Direct copy to _initializers list

3. Process Nodes
   ├─ Skip null nodes
   └─ Call each DataFlowNode.ToNodeProto() for conversion

4. Process Outputs
   ├─ Skip empty names
   └─ Build ValueInfoProto (with shape info)

5. Assemble GraphProto
   ├─ graph.Input ← all inputs
   ├─ graph.Node ← all nodes
   ├─ graph.Initializer ← all initializers
   ├─ graph.Output ← all outputs
   └─ graph.ValueInfo ← intermediate tensor shape info

6. Assemble ModelProto
   ├─ model.Graph ← graph
   └─ model.DocString ← modelName
```

---

### Internal Methods (for Extension Reference)

#### ConvertToTensorProto(DataFlowNode, string)

```csharp
internal TensorProto ConvertToTensorProto(DataFlowNode node, string name)
```

**Purpose**: Converts a data flow node to TensorProto (helper method).

#### InferShape(Tensor)

```csharp
internal TensorShapeProto InferShape(Tensor tensor)
```

**Purpose**: Infers ONNX shape from a TorchSharp tensor. Positive dimensions use `DimValue`; non-positive dimensions use `DimParam = "batch_size"`.

### Usage Examples

```csharp
using TorchSharp.OnnxExporter.Builder;
using TorchSharp.OnnxExporter.DataFlow;

// Method 1: Indirect usage via OnnxExporter.Export() (recommended)
OnnxExporter.Export(model, dummyInput, "output.onnx", "MyModel");

// Method 2: Manual construction
var dataFlowGraph = new DataFlowGraph();
// ... populate dataFlowGraph via SymbolicTraceEngine or manually ...

var builder = new OnnxGraphBuilder();
var modelProto = builder.Build(dataFlowGraph, "ManualModel");

// Serialize to file
using var fs = File.Create("manual_output.onnx");
using var coded = new Google.Protobuf.CodedOutputStream(fs);
modelProto.WriteTo(coded);
```

---

## 9. SymbolicTraceEngine Class

| Item | Description |
|------|-------------|
| **Namespace** | `TorchSharp.OnnxExporter.Tracing` |
| **Type** | `class` |
| **File** | [SymbolicTraceEngine.cs](../Tracing/SymbolicTraceEngine.cs) |
| **Summary** | Symbolic tracing engine — the core component of the export pipeline. Responsible for simulating the model's `forward` execution process, recording the input/output dependency relationships of each operation without performing actual numerical computation, and building a complete data flow graph (`DataFlowGraph`). Supports four tracing strategies: `Sequential`, `ModuleList`, standard module recursion, and reflection-discovered operator fields. |

### Constructor

```csharp
public SymbolicTraceEngine(int maxRecursionDepth = 100)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `maxRecursionDepth` | `int` | `100` | Maximum recursion depth to prevent infinite recursion |

---

### Public Methods

#### Trace(Module, Tensor)

```csharp
public DataFlowGraph Trace(Module model, Tensor dummyInput)
```

**Purpose**: Traces the model and builds a complete data flow graph. This is one of the most critical steps in the export process.

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | `Module` | The TorchSharp model to trace |
| `dummyInput` | `Tensor` | Dummy input tensor (used for shape inference) |

**Returns**: `DataFlowGraph` — The completed data flow graph.

**Exceptions**:

| Exception Type | Condition |
|----------------|-----------|
| `ArgumentNullException` | `model` or `dummyInput` is null |
| `InvalidOperationException` | Error occurs during tracing (includes model type, input shape, detailed suggestions) |

**Execution Flow**:

```
1. Create DataFlowGraph and TraceContext
2. Register input ("input") and its shape
3. Call TraceModule(model, "input") to begin tracing
4. Register final output
5. Infer intermediate tensor shapes (InferIntermediateShapesFromTrace)
6. Return DataFlowGraph
```

---

#### TraceModule(Module, string)

```csharp
public string TraceModule(Module module, string inputName)
```

**Purpose**: Traces a single module, automatically dispatching to the appropriate tracing strategy.

| Parameter | Type | Description |
|-----------|------|-------------|
| `module` | `Module` | The module to trace |
| `inputName` | `string` | The ONNX name of the input tensor |

**Returns**: `string` — The ONNX name of the output tensor.

**Dispatch Strategy**:

| Module Type | Tracing Method |
|-------------|---------------|
| `Sequential` | `TraceSequential()` — Sequentially traces all child modules |
| `ModuleList*` | `TraceModuleList()` — Sequentially traces all child modules |
| Others | `TraceStandardModule()` — Standard module handling |

---

#### TraceSequential(Module, string)

```csharp
public string TraceSequential(Module module, string inputName)
```

**Purpose**: Traces a `Sequential` module. Iterates through all child modules, passing each child module's output as the next child module's input, executing sequentially.

**Returns**: `string` — The output name of the last child module.

---

#### TraceModuleList(Module, string)

```csharp
public string TraceModuleList(Module moduleList, string inputName)
```

**Purpose**: Traces a `ModuleList` module. Logic is identical to `TraceSequential`.

**Returns**: `string` — The output name of the last child module.

### Internal Tracing Strategy (TraceStandardModule)

`TraceStandardModule()` is the core tracing method, attempting the following strategies in priority order:

```
Priority 1: Look up registered INodeProcessor
         ↓ Found → ProcessWithProcessor() generates DataFlowNode
         ↓ Not found ↓

Priority 2: Check if Sequential / ModuleList
         ↓ Yes → Recursive call to TraceSequential / TraceModuleList
         ↓ No ↓

Priority 3: Check for child modules (children())
         ↓ Has children → Recursively trace each child module
         ↓ No children ↓

Priority 4: Reflection-discover Operator fields (DiscoverOperatorFields)
         ↓ Found → Export tensor parameters + recursively trace each operator
         ↓ Not found ↓

Priority 5: Pass-through (return inputName, emit warning log)
```

### Usage Examples

```csharp
using TorchSharp.OnnxExporter.Tracing;

var engine = new SymbolicTraceEngine(maxRecursionDepth: 50);
var dataFlowGraph = engine.Trace(myModel, torch.randn(1, 3, 224, 224));

// dataFlowGraph now contains the complete ONNX graph structure
Console.WriteLine($"Node count: {dataFlowGraph.Nodes.Count}");
Console.WriteLine($"Inputs: {string.Join(", ", dataFlowGraph.Inputs)}");
Console.WriteLine($"Outputs: {string.Join(", ", dataFlowGraph.Outputs)}");
```

---

## Appendix A: Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                     User Entry Point                         │
│              OnnxExporter.Export() / ExportAsync()            │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│              Step 1: RegisterDefaultProcessors()             │
│            ModuleProcessorRegistry (Singleton Registry)       │
│  ┌─────────────────────────────────────────────────────┐     │
│  │ Type → INodeProcessor mapping (100+ built-in processors) │ │
│  │ Name → INodeProcessor mapping (fuzzy name matching)     │ │
│  └─────────────────────────────────────────────────────┘     │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│           Step 2: SymbolicTraceEngine.Trace()                 │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐  │
│  │ TraceContext │←→│DataFlowGraph │  │ ModuleProcessor    │  │
│  │ (value/shape │  │ (graph struct)│  │ Registry (lookup) │  │
│  │  management) │  │              │  │                    │  │
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
│  DataFlowGraph ──convert──→ ModelProto (ONNX protobuf)        │
│       │                      │                                │
│       ▼                      ▼                                │
│  GraphProto              CodedOutputStream                   │
│  (NodeProto × N)         → .onnx file                        │
│  (TensorProto weights)                                     │
└──────────────────────────────────────────────────────────────┘
```

## Appendix B: Namespace Index

| Namespace | Primary Types |
|----------|--------------|
| `TorchSharp.OnnxExporter` | `OnnxExporter`, `ModuleProcessorRegistry` |
| `TorchSharp.OnnxExporter.Modules` | `Operator`, `IOperator`, `Add`, `Sub`, `Mul`, `Div`, `MatMul`, `Pow`, `Sqrt`, `Exp`, `Log`, `AddWithBias`, `LinearOperator`, `Concat`, `Stack`, `ReshapeOp`, `TransposeOp`, `SqueezeOp`, `UnsqueezeOp`, `ClampOp`, `WhereOp`, `SumOp`, `MeanOp`, `OperatorBuilder` |
| `TorchSharp.OnnxExporter.Processors` | `INodeProcessor`, `BaseProcessor<T>`, `ElementWiseProcessor`, and 100+ concrete Processor classes |
| `TorchSharp.OnnxExporter.DataFlow` | `DataFlowGraph`, `DataFlowNode`, `TraceContext` |
| `TorchSharp.OnnxExporter.Builder` | `OnnxGraphBuilder` |
| `TorchSharp.OnnxExporter.Tracing` | `SymbolicTraceEngine` |

---

*Document generated: 2026-04-18 | Based on TorchSharp.OnnxExporter v1.0.0 source code*
