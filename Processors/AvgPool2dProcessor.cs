using System;
using System.Collections.Generic;
using TorchSharp.Modules;
using TorchSharp.OnnxExporter.DataFlow;

namespace TorchSharp.OnnxExporter.Processors
{
    public class AvgPool2dProcessor : BaseProcessor<AvgPool2d>
    {
        public override string OpType => "AveragePool";

        public override DataFlowNode Process(AvgPool2d module, TraceContext context)
        {
            var inputName = context.GetCurrentValue();
            var outputName = context.CreateTempName();

            var node = new DataFlowNode(OpType, new[] { inputName }, new[] { outputName });

            var kernelShape = new List<int>();
            if (module.kernel_size != null && module.kernel_size.Length >= 2)
            {
                kernelShape.Add((int)module.kernel_size[0]);
                kernelShape.Add((int)module.kernel_size[1]);
            }
            else
            {
                kernelShape.AddRange(new[] { 2, 2 });
            }
            node.Attributes["kernel_shape"] = kernelShape;

            var strides = new List<int>();
            if (module.stride != null && module.stride.Length >= 2)
            {
                strides.Add((int)module.stride[0]);
                strides.Add((int)module.stride[1]);
            }
            else
            {
                strides.AddRange(new[] { 2, 2 });
            }
            node.Attributes["strides"] = strides;

            var pads = new List<int>();
            if (module.padding != null && module.padding.Length >= 2)
            {
                pads.Add((int)module.padding[0]);
                pads.Add((int)module.padding[1]);
                pads.Add((int)module.padding[0]);
                pads.Add((int)module.padding[1]);
            }
            else
            {
                pads.AddRange(new[] { 0, 0, 0, 0 });
            }
            node.Attributes["pads"] = pads;

            context.Graph?.AddNode(node);
            return node;
        }
    }
}