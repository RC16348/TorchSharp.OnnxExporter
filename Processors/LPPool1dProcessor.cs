using System;
using System.Collections.Generic;
using TorchSharp.Modules;
using TorchSharp.OnnxExporter.DataFlow;

namespace TorchSharp.OnnxExporter.Processors
{
    public class LPPool1dProcessor : BaseProcessor<LPPool1d>
    {
        public override string OpType => "LPPool";

        public override DataFlowNode Process(LPPool1d module, TraceContext context)
        {
            var inputName = context.GetCurrentValue();
            var outputName = context.CreateTempName();

            var node = new DataFlowNode(OpType, new[] { inputName }, new[] { outputName });

            var kernelShape = new List<int>();
            kernelShape.Add((int)module.kernel_size);
            node.Attributes["kernel_shape"] = kernelShape;

            node.Attributes["p"] = (int)module.norm_type;

            var strides = new List<int>();
            strides.Add(module.stride > 0 ? (int)module.stride : 1);
            node.Attributes["strides"] = strides;

            context.Graph?.AddNode(node);
            return node;
        }
    }
}