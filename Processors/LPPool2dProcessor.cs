using System;
using System.Collections.Generic;
using TorchSharp.Modules;
using TorchSharp.OnnxExporter.DataFlow;

namespace TorchSharp.OnnxExporter.Processors
{
    public class LPPool2dProcessor : BaseProcessor<LPPool2d>
    {
        public override string OpType => "LPPool";

        public override DataFlowNode Process(LPPool2d module, TraceContext context)
        {
            var inputName = context.GetCurrentValue();
            var outputName = context.CreateTempName();

            var node = new DataFlowNode(OpType, new[] { inputName }, new[] { outputName });

            var kernelShape = new List<int>();
            if (module.kernel_size is not null && module.kernel_size.Length >= 2)
            {
                kernelShape.Add((int)module.kernel_size[0]);
                kernelShape.Add((int)module.kernel_size[1]);
            }
            else
            {
                kernelShape.AddRange(new[] { 2, 2 });
            }
            node.Attributes["kernel_shape"] = kernelShape;

            node.Attributes["p"] = (int)module.norm_type;

            var strides = new List<int>();
            if (module.stride is not null && module.stride.Length >= 2)
            {
                strides.Add((int)module.stride[0]);
                strides.Add((int)module.stride[1]);
            }
            else
            {
                strides.AddRange(new[] { 1, 1 });
            }
            node.Attributes["strides"] = strides;

            context.Graph?.AddNode(node);
            return node;
        }
    }
}