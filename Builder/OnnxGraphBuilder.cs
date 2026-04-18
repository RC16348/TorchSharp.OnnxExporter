using System;
using System.Collections.Generic;
using Onnx;
using Google.Protobuf;
using TorchSharp.OnnxExporter.DataFlow;
using Tensor = TorchSharp.torch.Tensor;

namespace TorchSharp.OnnxExporter.Builder
{
    /// <summary>
    /// ONNX图构建器 - 核心组件
    /// 负责将数据流图（DataFlowGraph）转换为ONNX protobuf格式
    /// </summary>
    public sealed class OnnxGraphBuilder
    {
        private readonly ModelProto _model;
        private readonly GraphProto _graph;
        private readonly List<ValueInfoProto> _inputs = new();
        private readonly List<ValueInfoProto> _outputs = new();
        private readonly List<TensorProto> _initializers = new();
        private readonly List<NodeProto> _nodes = new();

        public OnnxGraphBuilder()
        {
            _model = new ModelProto
            {
                IrVersion = 7,
                ProducerName = "TorchSharp.OnnxExporter",
                ProducerVersion = "1.0.0"
            };

            _graph = new GraphProto
            {
                Name = "graph"
            };

            // 【重要】ONNX操作集版本配置
            // ai.onnx: 标准ONNX操作集
            // "": 旧版操作集兼容
            var opsetImport = new OperatorSetIdProto
            {
                Domain = "ai.onnx",
                Version = 14
            };
            _model.OpsetImport.Add(opsetImport);

            var opsetImportEmpty = new OperatorSetIdProto
            {
                Domain = "",
                Version = 14
            };
            _model.OpsetImport.Add(opsetImportEmpty);
        }

        /// <summary>
        /// 添加输入张量信息
        /// </summary>
        /// <exception cref="ArgumentNullException">当input为null时抛出</exception>
        public void AddInput(ValueInfoProto input)
        {
            if (input == null)
                throw new ArgumentNullException(nameof(input), "[ONNX图构建错误] 输入张量不能为null");

            _inputs.Add(input);
        }

        /// <summary>
        /// 添加初始化器（模型权重）
        /// </summary>
        /// <exception cref="ArgumentNullException">当initializer为null时抛出</exception>
        public void AddInitializer(TensorProto initializer)
        {
            if (initializer == null)
                throw new ArgumentNullException(nameof(initializer), "[ONNX图构建错误] 初始化器不能为null");

            _initializers.Add(initializer);
        }

        /// <summary>
        /// 添加输出张量信息
        /// </summary>
        /// <exception cref="ArgumentNullException">当output为null时抛出</exception>
        public void AddOutput(ValueInfoProto output)
        {
            if (output == null)
                throw new ArgumentNullException(nameof(output), "[ONNX图构建错误] 输出张量不能为null");

            _outputs.Add(output);
        }

        /// <summary>
        /// 构建ONNX ModelProto
        /// 将数据流图转换为ONNX protobuf格式
        /// </summary>
        /// <param name="dataFlow">数据流图</param>
        /// <param name="modelName">模型名称</param>
        /// <returns>ONNX ModelProto</returns>
        /// <exception cref="ArgumentNullException">当dataFlow为null时抛出</exception>
        /// <exception cref="InvalidOperationException">当图构建过程中发生错误时抛出</exception>
        public ModelProto Build(DataFlowGraph dataFlow, string modelName)
        {
            if (dataFlow == null)
                throw new ArgumentNullException(nameof(dataFlow), "[ONNX图构建错误] 数据流图不能为null");

            if (string.IsNullOrEmpty(modelName))
                modelName = "TorchSharpModel";

            try
            {
                // 【异常处理】处理输入
                foreach (var inputName in dataFlow.Inputs)
                {
                    if (string.IsNullOrEmpty(inputName))
                    {
                        Console.WriteLine($"[ONNX图构建警告] 跳过空的输入名称");
                        continue;
                    }

                    var inputProto = new ValueInfoProto { Name = inputName };
                    var inputType = new TypeProto
                    {
                        TensorType = new TypeProto.Types.Tensor()
                    };
                    inputType.TensorType.ElemType = (int)TensorProto.Types.DataType.Float;
                    inputType.TensorType.Shape = new TensorShapeProto();

                    if (dataFlow.InputShapes.TryGetValue(inputName, out var inputShape))
                    {
                        foreach (var dim in inputShape)
                        {
                            var dimValue = new TensorShapeProto.Types.Dimension();
                            if (dim > 0)
                            {
                                dimValue.DimValue = dim;
                            }
                            else
                            {
                                dimValue.DimParam = "dynamic";
                            }
                            inputType.TensorType.Shape.Dim.Add(dimValue);
                        }
                    }

                    inputProto.Type = inputType;
                    _inputs.Add(inputProto);
                }

                // 处理初始化器
                foreach (var initializer in dataFlow.Initializers)
                {
                    _initializers.Add(initializer);
                }

                // 【核心】处理节点
                foreach (var node in dataFlow.Nodes)
                {
                    if (node == null)
                    {
                        Console.WriteLine($"[ONNX图构建警告] 跳过null节点");
                        continue;
                    }

                    if (node.OpType == "Conv")
                    {
                        Console.WriteLine($"[DEBUG BeforeToNodeProto] Node={node.Name} OpType={node.OpType}");
                        foreach (var attr in node.Attributes)
                        {
                            string attrStr = AttrToString(attr.Key, attr.Value);
                            Console.WriteLine($"  attr={attr.Key} value={attrStr}");
                        }
                    }

                    try
                    {
                        _nodes.Add(node.ToNodeProto());
                    }
                    catch (Exception ex)
                    {
                        throw new InvalidOperationException(
                            $"[ONNX图构建错误] 节点转换失败: 节点={node.OpType}, " +
                            $"输入={string.Join(",", node.Inputs)}, 错误={ex.Message}", ex);
                    }
                }

                // 处理输出
                foreach (var outputName in dataFlow.Outputs)
                {
                    if (string.IsNullOrEmpty(outputName))
                    {
                        Console.WriteLine($"[ONNX图构建警告] 跳过空的输出名称");
                        continue;
                    }

                    var outputProto = new ValueInfoProto { Name = outputName };
                    var outputType = new TypeProto
                    {
                        TensorType = new TypeProto.Types.Tensor()
                    };
                    outputType.TensorType.ElemType = (int)TensorProto.Types.DataType.Float;
                    outputType.TensorType.Shape = new TensorShapeProto();

                    if (dataFlow.OutputShapes.TryGetValue(outputName, out var outputShape))
                    {
                        foreach (var dim in outputShape)
                        {
                            var dimValue = new TensorShapeProto.Types.Dimension();
                            if (dim > 0)
                            {
                                dimValue.DimValue = dim;
                            }
                            else
                            {
                                dimValue.DimParam = "dynamic";
                            }
                            outputType.TensorType.Shape.Dim.Add(dimValue);
                        }
                    }

                    outputProto.Type = outputType;
                    _outputs.Add(outputProto);
                }

                // 添加到图
                foreach (var input in _inputs)
                {
                    _graph.Input.Add(input);
                }

                foreach (var node in _nodes)
                {
                    _graph.Node.Add(node);
                }

                foreach (var initializer in _initializers)
                {
                    _graph.Initializer.Add(initializer);
                }

                foreach (var output in _outputs)
                {
                    _graph.Output.Add(output);
                }

                AddIntermediateValueInfo(_graph, dataFlow);

                _model.Graph = _graph;
                _model.DocString = modelName;

                return _model;
            }
            catch (InvalidOperationException)
            {
                throw;
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException(
                    $"[ONNX图构建错误] 图构建失败: 模型={modelName}, 错误={ex.Message}", ex);
            }
        }

        private static string AttrToString(string key, object value)
        {
            switch (value)
            {
                case int intVal:
                    return $"int:{intVal}";
                case long longVal:
                    return $"long:{longVal}";
                case float floatVal:
                    return $"float:{floatVal}";
                case double doubleVal:
                    return $"double:{doubleVal}";
                case string strVal:
                    return $"string:{strVal}";
                case int[] intArr:
                    return $"int[]:[{string.Join(",", intArr)}]";
                case long[] longArr:
                    return $"long[]:[{string.Join(",", longArr)}]";
                case float[] floatArr:
                    return $"float[]:[{string.Join(",", floatArr)}]";
                case double[] doubleArr:
                    return $"double[]:[{string.Join(",", doubleArr)}]";
                case IEnumerable<int> intEnum:
                    return $"ienumerable_int:[{string.Join(",", intEnum)}]";
                case IEnumerable<long> longEnum:
                    return $"ienumerable_long:[{string.Join(",", longEnum)}]";
                default:
                    return $"unknown:{value?.GetType().Name ?? "null"}";
            }
        }

        /// <summary>
        /// 将DataFlowNode转换为TensorProto
        /// </summary>
        internal TensorProto ConvertToTensorProto(DataFlowNode node, string name)
        {
            var tensorProto = new TensorProto
            {
                Name = name,
                DataType = (int)TensorProto.Types.DataType.Float
            };

            tensorProto.Dims.Add(0);

            return tensorProto;
        }

        /// <summary>
        /// 从TorchSharp张量推断ONNX形状
        /// </summary>
        /// <param name="tensor">TorchSharp张量</param>
        /// <returns>ONNX张量形状</returns>
        internal TensorShapeProto InferShape(Tensor tensor)
        {
            var shape = new TensorShapeProto();

            if (tensor is null || tensor.shape is null)
            {
                Console.WriteLine("[ONNX图构建警告] 张量或形状为null，返回空形状");
                return shape;
            }

            foreach (var dim in tensor.shape)
            {
                var dimValue = new TensorShapeProto.Types.Dimension();
                if (dim > 0)
                {
                    dimValue.DimValue = dim;
                }
                else
                {
                    // 【动态维度处理】
                    // 负数维度通常表示动态维度（如batch_size）
                    // 在ONNX中使用dim_param标记
                    dimValue.DimParam = "batch_size";
                }
                shape.Dim.Add(dimValue);
            }

            return shape;
        }

        private void AddIntermediateValueInfo(GraphProto graph, DataFlowGraph dataFlow)
        {
            if (dataFlow.IntermediateShapes == null || dataFlow.IntermediateShapes.Count == 0)
            {
                Console.WriteLine("[DEBUG AddIntermediateValueInfo] No intermediate shapes to add");
                return;
            }

            Console.WriteLine($"[DEBUG AddIntermediateValueInfo] Adding shapes for {dataFlow.IntermediateShapes.Count} intermediate tensors");

            foreach (var kvp in dataFlow.IntermediateShapes)
            {
                var name = kvp.Key;
                var shape = kvp.Value;

                var valueInfo = new ValueInfoProto { Name = name };
                var tensorType = new TypeProto
                {
                    TensorType = new TypeProto.Types.Tensor()
                };
                tensorType.TensorType.ElemType = (int)TensorProto.Types.DataType.Float;
                tensorType.TensorType.Shape = new TensorShapeProto();

                foreach (var dim in shape)
                {
                    var dimValue = new TensorShapeProto.Types.Dimension();
                    if (dim > 0)
                    {
                        dimValue.DimValue = dim;
                    }
                    else
                    {
                        dimValue.DimParam = "dynamic";
                    }
                    tensorType.TensorType.Shape.Dim.Add(dimValue);
                }

                valueInfo.Type = tensorType;
                graph.ValueInfo.Add(valueInfo);

                Console.WriteLine($"[DEBUG AddIntermediateValueInfo] Added shape for '{name}': [{string.Join(",", shape)}]");
            }
        }
    }
}