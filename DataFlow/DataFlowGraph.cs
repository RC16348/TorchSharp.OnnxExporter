using System;
using System.Collections.Generic;
using Onnx;
using static TorchSharp.torch;
using Tensor = TorchSharp.torch.Tensor;

namespace TorchSharp.OnnxExporter.DataFlow
{
    public class DataFlowGraph
    {
        public List<DataFlowNode> Nodes { get; } = new();
        public List<string> Inputs { get; } = new();
        public List<string> Outputs { get; } = new();
        public List<TensorProto> Initializers { get; } = new();
        public Dictionary<string, List<long>> InputShapes { get; } = new();
        public Dictionary<string, List<long>> OutputShapes { get; } = new();
        public Dictionary<string, List<long>> IntermediateShapes { get; } = new();

        public DataFlowNode AddNode(DataFlowNode node)
        {
            Nodes.Add(node);
            return node;
        }

        public void AddInitializer(string name, Tensor tensor)
        {
            if (tensor is null) return;

            var existing = Initializers.FirstOrDefault(i => i.Name == name);
            if (existing != null) return;

            var initializer = new TensorProto
            {
                Name = name,
                DataType = (int)(tensor.dtype == ScalarType.Float32 ? TensorProto.Types.DataType.Float : TensorProto.Types.DataType.Double)
            };

            foreach (var dim in tensor.shape)
            {
                initializer.Dims.Add(dim);
            }

            if (tensor.dtype == ScalarType.Float32)
            {
                var data = tensor.data<float>();
                foreach (var value in data)
                {
                    initializer.FloatData.Add(value);
                }
            }
            else
            {
                var data = tensor.data<double>();
                foreach (var value in data)
                {
                    initializer.DoubleData.Add(value);
                }
            }

            Initializers.Add(initializer);
        }

        public void AddInitializer(string name, float[] data, long[] dims)
        {
            if (data is null) return;

            var initializer = new TensorProto
            {
                Name = name,
                DataType = (int)TensorProto.Types.DataType.Float
            };

            foreach (var dim in dims)
            {
                initializer.Dims.Add(dim);
            }

            foreach (var value in data)
            {
                initializer.FloatData.Add(value);
            }

            Initializers.Add(initializer);
        }

        public void AddInitializer(string name, long[] data)
        {
            if (data is null) return;

            var initializer = new TensorProto
            {
                Name = name,
                DataType = (int)TensorProto.Types.DataType.Int64
            };

            initializer.Dims.Add(data.Length);

            foreach (var value in data)
            {
                initializer.Int64Data.Add(value);
            }

            Initializers.Add(initializer);
        }

        public void AddInput(string name, Tensor tensor)
        {
            Inputs.Add(name);
            if (!(tensor is null))
            {
                var shape = tensor.shape.ToList();
                Console.WriteLine($"[DEBUG AddInput] Storing shape for {name}: [{string.Join(",", shape)}]");
                InputShapes[name] = shape;
            }
        }

        public void AddOutput(string name, Tensor tensor)
        {
            Outputs.Add(name);
            if (!(tensor is null))
                OutputShapes[name] = tensor.shape.ToList();
        }

        public void AddIntermediateShape(string name, List<long> shape)
        {
            if (!string.IsNullOrEmpty(name) && shape != null)
            {
                IntermediateShapes[name] = shape;
            }
        }

        public GraphProto ToGraphProto()
        {
            var graphProto = new GraphProto();

            Console.WriteLine($"[DEBUG ToGraphProto] Inputs count: {Inputs.Count}");
            Console.WriteLine($"[DEBUG ToGraphProto] InputShapes count: {InputShapes.Count}");
            foreach (var input in Inputs)
            {
                Console.WriteLine($"[DEBUG ToGraphProto] Processing input: {input}");
                var valueInfo = new ValueInfoProto { Name = input };
                Console.WriteLine($"[DEBUG ToGraphProto] Checking InputShapes for key '{input}'");
                Console.WriteLine($"[DEBUG ToGraphProto] InputShapes keys: [{string.Join(",", InputShapes.Keys)}]");
                if (InputShapes.TryGetValue(input, out var shape))
                {
                    Console.WriteLine($"[DEBUG ToGraphProto] Found shape for {input}: [{string.Join(",", shape)}]");
                    var inputType = new TypeProto
                    {
                        TensorType = new TypeProto.Types.Tensor()
                    };
                    inputType.TensorType.ElemType = (int)TensorProto.Types.DataType.Float;
                    inputType.TensorType.Shape = new TensorShapeProto();
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
                        inputType.TensorType.Shape.Dim.Add(dimValue);
                    }
                    valueInfo.Type = inputType;
                }
                else
                {
                    Console.WriteLine($"[DEBUG ToGraphProto] No shape found for {input}");
                }
                graphProto.Input.Add(valueInfo);
            }

            foreach (var node in Nodes)
            {
                graphProto.Node.Add(node.ToNodeProto());
            }

            foreach (var output in Outputs)
            {
                var valueInfo = new ValueInfoProto { Name = output };
                if (OutputShapes.TryGetValue(output, out var shape))
                {
                    var outputType = new TypeProto
                    {
                        TensorType = new TypeProto.Types.Tensor()
                    };
                    outputType.TensorType.ElemType = (int)TensorProto.Types.DataType.Float;
                    outputType.TensorType.Shape = new TensorShapeProto();
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
                        outputType.TensorType.Shape.Dim.Add(dimValue);
                    }
                    valueInfo.Type = outputType;
                }
                graphProto.Output.Add(valueInfo);
            }

            foreach (var initializer in Initializers)
            {
                graphProto.Initializer.Add(initializer);
            }

            return graphProto;
        }
    }
}