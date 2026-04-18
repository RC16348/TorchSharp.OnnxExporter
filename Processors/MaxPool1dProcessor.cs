using System;
using System.Collections.Generic;
using TorchSharp.Modules;
using TorchSharp.OnnxExporter.DataFlow;

namespace TorchSharp.OnnxExporter.Processors
{
    public class MaxPool1dProcessor : BaseProcessor<MaxPool1d>
    {
        public override string OpType => "MaxPool";

        public override DataFlowNode Process(MaxPool1d module, TraceContext context)
        {
            var inputName = context.GetCurrentValue();
            var outputName = context.CreateTempName();

            var node = new DataFlowNode(OpType, new[] { inputName }, new[] { outputName });

            var kernelShape = new List<int>();
            kernelShape.Add((int)module.kernel_size);
            node.Attributes["kernel_shape"] = kernelShape;

            var strides = new List<int>();
            strides.Add(module.stride > 0 ? (int)module.stride : 2);
            node.Attributes["strides"] = strides;

            var pads = new List<int>();
            pads.Add((int)module.padding);
            pads.Add((int)module.padding);
            node.Attributes["pads"] = pads;

            context.Graph?.AddNode(node);
            return node;
        }
    }
}