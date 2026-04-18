using System;
using System.Collections.Generic;
using TorchSharp.Modules;
using TorchSharp.OnnxExporter.DataFlow;

namespace TorchSharp.OnnxExporter.Processors
{
    public class MaxPool3dProcessor : BaseProcessor<MaxPool3d>
    {
        public override string OpType => "MaxPool";

        public override DataFlowNode Process(MaxPool3d module, TraceContext context)
        {
            var inputName = context.GetCurrentValue();
            var outputName = context.CreateTempName();

            var node = new DataFlowNode(OpType, new[] { inputName }, new[] { outputName });

            var kernelShape = new List<int>();
            if (module.kernel_size is not null && module.kernel_size.Length >= 3)
            {
                kernelShape.Add((int)module.kernel_size[0]);
                kernelShape.Add((int)module.kernel_size[1]);
                kernelShape.Add((int)module.kernel_size[2]);
            }
            else
            {
                kernelShape.AddRange(new[] { 2, 2, 2 });
            }
            node.Attributes["kernel_shape"] = kernelShape;

            var strides = new List<int>();
            if (module.stride is not null && module.stride.Length >= 3)
            {
                strides.Add((int)module.stride[0]);
                strides.Add((int)module.stride[1]);
                strides.Add((int)module.stride[2]);
            }
            else
            {
                strides.AddRange(new[] { 2, 2, 2 });
            }
            node.Attributes["strides"] = strides;

            var pads = new List<int>();
            if (module.padding is not null && module.padding.Length >= 3)
            {
                pads.Add((int)module.padding[0]);
                pads.Add((int)module.padding[1]);
                pads.Add((int)module.padding[2]);
                pads.Add((int)module.padding[0]);
                pads.Add((int)module.padding[1]);
                pads.Add((int)module.padding[2]);
            }
            else
            {
                pads.AddRange(new[] { 0, 0, 0, 0, 0, 0 });
            }
            node.Attributes["pads"] = pads;

            context.Graph?.AddNode(node);
            return node;
        }
    }
}