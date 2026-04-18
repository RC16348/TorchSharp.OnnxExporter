using System;
using System.Collections.Generic;
using TorchSharp.Modules;
using TorchSharp.OnnxExporter.DataFlow;

namespace TorchSharp.OnnxExporter.Processors
{
    public class AvgPool1dProcessor : BaseProcessor<AvgPool1d>
    {
        public override string OpType => "AveragePool";

        public override DataFlowNode Process(AvgPool1d module, TraceContext context)
        {
            var inputName = context.GetCurrentValue();
            var outputName = context.CreateTempName();

            var node = new DataFlowNode(OpType, new[] { inputName }, new[] { outputName });

            var kernelShape = new List<int>();
            kernelShape.Add((int)module.kernel_size);
            node.Attributes["kernel_shape"] = kernelShape;

            var strides = new List<int>();
            if (module.stride.HasValue)
            {
                strides.Add((int)module.stride.Value);
            }
            else
            {
                strides.Add((int)module.kernel_size);
            }
            node.Attributes["strides"] = strides;

            var pads = new List<int>();
            if (module.padding.HasValue)
            {
                pads.Add((int)module.padding.Value);
                pads.Add((int)module.padding.Value);
            }
            else
            {
                pads.AddRange(new[] { 0, 0 });
            }
            node.Attributes["pads"] = pads;

            node.Attributes["count_include_pad"] = module.count_include_pad;

            context.Graph?.AddNode(node);
            return node;
        }
    }
}