using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using TorchSharp;
using TorchSharp.OnnxExporter.DataFlow;
using TorchSharp.OnnxExporter.Processors;
using Tensor = TorchSharp.torch.Tensor;
using Module = TorchSharp.torch.nn.Module;

namespace TorchSharp.OnnxExporter.Tracing
{
    /// <summary>
    /// 符号跟踪引擎 - 核心组件
    /// 负责模拟模块的forward执行，记录每个操作的输入输出关系，构建数据流图
    /// </summary>
    public class SymbolicTraceEngine
    {
        private readonly ModuleProcessorRegistry _registry;
        private readonly int _maxRecursionDepth;

        public SymbolicTraceEngine(int maxRecursionDepth = 100)
        {
            _registry = ModuleProcessorRegistry.Instance;
            _maxRecursionDepth = maxRecursionDepth;
        }

        /// <summary>
        /// 跟踪模块并构建数据流图
        /// </summary>
        /// <param name="model">要跟踪的模块</param>
        /// <param name="dummyInput">虚拟输入，用于形状推断</param>
        /// <returns>构建完成的数据流图</returns>
        /// <exception cref="ArgumentNullException">当model或dummyInput为null时抛出</exception>
        /// <exception cref="InvalidOperationException">当跟踪过程中发生错误时抛出</exception>
        public DataFlowGraph Trace(Module model, Tensor dummyInput)
        {
            if (model == null)
                throw new ArgumentNullException(nameof(model),
                    "[ONNX导出错误] 模型不能为null。" +
                    "\n  → 请确保您传入了一个有效的TorchSharp模型对象。");

            if (dummyInput is null)
                throw new ArgumentNullException(nameof(dummyInput),
                    "[ONNX导出错误] 虚拟输入不能为null。" +
                    "\n  → 请使用 torch.randn() 或 torch.zeros() 创建一个有效的输入张量。" +
                    "\n  → 输入形状应与模型实际输入一致，例如：torch.randn(1, 3, 224, 224)");

            try
            {
                var graph = new DataFlowGraph();
                var context = new TraceContext(graph);
                _currentContext = context;

                var inputName = "input";
                graph.AddInput(inputName, dummyInput);
                context.SetNamedValue("input", inputName);
                context.SetShape(inputName, dummyInput.shape.ToList());

                var outputName = TraceModule(model, inputName);
                graph.AddOutput(outputName, dummyInput);

                InferIntermediateShapesFromTrace(graph, context);

                _currentContext = null;
                return graph;
            }
            catch (ArgumentNullException)
            {
                throw;
            }
            catch (InvalidOperationException)
            {
                throw;
            }
            catch (Exception ex)
            {
                var modelType = model.GetType().Name;
                var inputShape = dummyInput is not null ? string.Join(",", dummyInput.shape) : "unknown";

                throw new InvalidOperationException(
                    $"[ONNX导出错误] 符号跟踪失败。" +
                    $"\n  → 模型类型: {modelType}" +
                    $"\n  → 输入形状: [{inputShape}]" +
                    $"\n  → 错误详情: {ex.Message}" +
                    $"\n" +
                    $"\n【建议排查步骤】" +
                    $"\n  1. 检查模型结构是否包含不支持的操作（如动态控制流）" +
                    $"\n  2. 确认虚拟输入形状与模型实际输入一致" +
                    $"\n  3. 检查是否有模块类型未被处理器支持", ex);
            }
        }

        private void InferIntermediateShapesFromTrace(DataFlowGraph graph, TraceContext context)
        {
            try
            {
                foreach (var kvp in context.ValueShapes)
                {
                    var name = kvp.Key;
                    var shape = kvp.Value;
                    if (!string.IsNullOrEmpty(name) && name.StartsWith("tmp_"))
                    {
                        graph.AddIntermediateShape(name, shape);
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[ONNX形状推断警告] 无法推断中间形状: {ex.Message}");
            }
        }

        /// <summary>
        /// 跟踪单个模块
        /// 自动识别模块类型（Sequential、ModuleList或其他）并调用相应的跟踪方法
        /// </summary>
        public string TraceModule(Module module, string inputName)
        {
            var moduleTypeName = module.GetType().Name;

            Console.WriteLine($"[DEBUG TraceModule] Called with module={moduleTypeName}, input={inputName}");

            if (moduleTypeName == "Sequential")
            {
                Console.WriteLine($"[DEBUG TraceModule] Processing as Sequential: {moduleTypeName}");
                return TraceSequential(module, inputName);
            }

            if (moduleTypeName.StartsWith("ModuleList"))
            {
                Console.WriteLine($"[DEBUG TraceModule] Processing as ModuleList: {moduleTypeName}");
                return TraceModuleList(module, inputName);
            }

            Console.WriteLine($"[DEBUG TraceModule] Calling TraceStandardModule for: {moduleTypeName}");
            return TraceStandardModule(module, inputName);
        }

        /// <summary>
        /// 跟踪Sequential模块
        /// 按顺序跟踪所有子模块，每个子模块的输出作为下一个子模块的输入
        /// </summary>
        public string TraceSequential(Module module, string inputName)
        {
            var currentInput = inputName;

            foreach (var child in module.children())
            {
                currentInput = TraceModule(child, currentInput);
            }

            return currentInput;
        }

        /// <summary>
        /// 跟踪ModuleList模块
        /// 按顺序跟踪所有子模块
        /// </summary>
        public string TraceModuleList(Module moduleList, string inputName)
        {
            var currentInput = inputName;

            foreach (var module in moduleList.children())
            {
                currentInput = TraceModule(module, currentInput);
            }

            return currentInput;
        }

        /// <summary>
        /// 跟踪标准模块（核心方法）
        /// 1. 获取模块的处理器
        /// 2. 如果没有处理器但有子模块，递归处理子模块
        /// 3. 如果都没有，返回输入（透传）
        /// </summary>
        /// <param name="module">要跟踪的模块</param>
        /// <param name="inputName">输入张量名称</param>
        /// <param name="depth">当前递归深度，防止无限递归</param>
        /// <returns>输出张量名称</returns>
        private string TraceStandardModule(Module module, string inputName, int depth = 0)
        {
            var moduleTypeName = module.GetType().Name;

            if (depth >= _maxRecursionDepth)
            {
                throw new InvalidOperationException(
                    $"[ONNX导出错误] 递归深度超限。" +
                    $"\n  → 当前模块: {moduleTypeName}" +
                    $"\n  → 当前深度: {depth}" +
                    $"\n  → 最大深度: {_maxRecursionDepth}" +
                    $"\n" +
                    $"\n【原因分析】" +
                    $"\n  这通常表示模型结构中存在过深的嵌套或循环依赖。" +
                    $"\n  可能的场景：" +
                    $"\n    - 递归模块定义" +
                    $"\n    - 过度嵌套的Sequential结构" +
                    $"\n    - 循环模块引用");
            }

            var originalInput = _currentContext?.GetOriginalInput();
            if (string.IsNullOrEmpty(originalInput))
            {
                _currentContext?.SetOriginalInput(inputName);
            }

            var processor = _registry.GetProcessor(module);

            if (processor != null)
            {
                Console.WriteLine($"[DEBUG TraceStandardModule] module={moduleTypeName}, processor=FOUND ({processor.GetType().Name})");
                return ProcessWithProcessor(processor, module, inputName);
            }

            Console.WriteLine($"[DEBUG TraceStandardModule] module={moduleTypeName}, processor=NOT_FOUND");

            if (moduleTypeName == "Sequential")
            {
                Console.WriteLine($"[DEBUG TraceStandardModule] Processing as Sequential: {moduleTypeName}");
                return TraceSequential(module, inputName);
            }

            if (moduleTypeName.StartsWith("ModuleList"))
            {
                Console.WriteLine($"[DEBUG TraceStandardModule] Processing as ModuleList: {moduleTypeName}");
                return TraceModuleList(module, inputName);
            }

            var hasChildren = module.children().Any();
            Console.WriteLine($"[DEBUG TraceStandardModule] module={moduleTypeName}, hasChildren={hasChildren}");

            if (hasChildren)
            {
                var currentInput = inputName;
                foreach (var child in module.children())
                {
                    currentInput = TraceStandardModule(child, currentInput, depth + 1);
                }
                return currentInput;
            }

            var operatorFields = DiscoverOperatorFields(module).ToList();
            if (operatorFields.Any())
            {
                Console.WriteLine($"[DEBUG TraceStandardModule] Found {operatorFields.Count} operator fields in {moduleTypeName}");
                var currentInput = inputName;
                foreach (var opField in operatorFields)
                {
                    ExportOperatorTensors(opField, _currentContext);
                    currentInput = TraceModule(opField, currentInput);
                }
                return currentInput;
            }

            Console.WriteLine($"[ONNX导出警告] 未找到模块类型的处理器: {moduleTypeName}，输入将透传。");
            Console.WriteLine($"  提示：如果这是自定义模块，请确保已注册相应的INodeProcessor。");
            Console.WriteLine($"  建议：查看 https://github.com/你的仓库/TorchSharp.OnnxExporter#自定义处理器注册 了解如何注册自定义处理器。");

            return inputName;
        }

        /// <summary>
        /// 使用处理器处理模块
        /// </summary>
        /// <exception cref="InvalidOperationException">当TraceContext未初始化或处理器处理失败时抛出</exception>
        private string ProcessWithProcessor(INodeProcessor processor, Module module, string inputName)
        {
            if (_currentContext == null || _currentContext.Graph == null)
            {
                throw new InvalidOperationException(
                    "[ONNX导出错误] TraceContext或DataFlowGraph未初始化。" +
                    "\n  → 这通常是因为在调用Trace()方法之前尝试使用了TraceContext。" +
                    "\n  → 请确保通过 OnnxExporter.Export() 方法进行导出。");
            }

            _currentContext.SetCurrentValue(inputName);
            _currentContext.SetNamedValue("current_input", inputName);

            try
            {
                var node = processor.Process(module, _currentContext);

                if (node.Outputs.Count > 0)
                {
                    _currentContext.SetCurrentValue(node.Outputs[0]);
                    return node.Outputs[0];
                }

                return inputName;
            }
            catch (ArgumentNullException ex)
            {
                throw new InvalidOperationException(
                    $"[ONNX导出错误] 处理器处理失败：参数为空。" +
                    $"\n  → 处理器: {processor.GetType().Name}" +
                    $"\n  → 模块: {module.GetType().Name}" +
                    $"\n  → 错误参数: {ex.ParamName}" +
                    $"\n" +
                    $"\n【建议】" +
                    $"\n  请检查模块参数是否正确初始化。", ex);
            }
            catch (InvalidOperationException)
            {
                throw;
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException(
                    $"[ONNX导出错误] 处理器处理失败。" +
                    $"\n  → 处理器: {processor.GetType().Name}" +
                    $"\n  → 模块: {module.GetType().Name}" +
                    $"\n  → 操作类型: {processor.OpType}" +
                    $"\n  → 错误详情: {ex.Message}" +
                    $"\n" +
                    $"\n【建议排查步骤】" +
                    $"\n  1. 检查模块的权重参数是否正确初始化" +
                    $"\n  2. 确认输入形状是否与模块期望一致" +
                    $"\n  3. 检查是否有不支持的参数配置", ex);
            }
        }

        /// <summary>
        /// 通过反射发现模块中的 operator 字段
        /// 这些字段包含 TorchSharp.OnnxExporter.Modules 命名空间下的算子模块
        /// </summary>
        private IEnumerable<Module> DiscoverOperatorFields(Module module)
        {
            var results = new List<Module>();
            var fields = module.GetType().GetFields(
                BindingFlags.NonPublic | BindingFlags.Instance);

            foreach (var field in fields)
            {
                if (typeof(Module).IsAssignableFrom(field.FieldType))
                {
                    try
                    {
                        var value = field.GetValue(module);
                        if (value is Module m && IsOperatorModule(m))
                        {
                            Console.WriteLine($"[DEBUG DiscoverOperatorFields] Found operator field: {field.Name} -> {m.GetType().Name}");
                            results.Add(m);
                        }
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"[DEBUG DiscoverOperatorFields] Failed to get field value {field.Name}: {ex.Message}");
                    }
                }
            }

            return results;
        }

        /// <summary>
        /// 判断模块是否为 Operator 模块（位于 TorchSharp.OnnxExporter.Modules 命名空间）
        /// </summary>
        private bool IsOperatorModule(Module module)
        {
            var typeName = module.GetType().FullName ?? "";
            return typeName.Contains("TorchSharp.OnnxExporter.Modules");
        }

        /// <summary>
        /// 通过反射发现 operator 模块中的 tensor 类型字段
        /// 这些字段包含权重、偏置等需要导出为 ONNX initializer 的张量
        /// </summary>
        private IEnumerable<(string name, Tensor tensor)> DiscoverOperatorTensors(Module opField)
        {
            var results = new List<(string name, Tensor tensor)>();
            var fields = opField.GetType().GetFields(
                BindingFlags.NonPublic | BindingFlags.Instance);

            foreach (var field in fields)
            {
                if (field.FieldType == typeof(Tensor))
                {
                    try
                    {
                        var tensor = (Tensor)field.GetValue(opField);
                        if (!object.ReferenceEquals(tensor, null))
                        {
                            Console.WriteLine($"[DEBUG DiscoverOperatorTensors] Found tensor field: {field.Name}, shape=[{string.Join(",", tensor.shape)}]");
                            results.Add((field.Name, tensor));
                        }
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"[DEBUG DiscoverOperatorTensors] Failed to get tensor field {field.Name}: {ex.Message}");
                    }
                }
            }

            return results;
        }

        /// <summary>
        /// 将 operator 模块的 tensor 参数导出为 ONNX initializer
        /// </summary>
        private void ExportOperatorTensors(Module opField, TraceContext context)
        {
            if (context?.Graph == null) return;

            var tensors = DiscoverOperatorTensors(opField);
            foreach (var (name, tensor) in tensors)
            {
                var initializerName = $"{opField.GetType().Name}_{name}";
                Console.WriteLine($"[DEBUG ExportOperatorTensors] Adding initializer: {initializerName}, shape=[{string.Join(",", tensor.shape)}]");
                context.Graph.AddInitializer(initializerName, tensor);
            }
        }

        private TraceContext? _currentContext;
    }
}