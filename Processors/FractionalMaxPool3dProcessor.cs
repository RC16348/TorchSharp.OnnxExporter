using System;
using System.Collections.Generic;
using TorchSharp.Modules;
using TorchSharp.OnnxExporter.DataFlow;

namespace TorchSharp.OnnxExporter.Processors
{
    public class FractionalMaxPool3dProcessor : BaseProcessor<FractionalMaxPool3d>
    {
        public override string OpType => "FractionalMaxPool";

        public override DataFlowNode Process(FractionalMaxPool3d module, TraceContext context)
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

            if (module.output_ratio is not null && module.output_ratio.Length >= 3)
            {
                node.Attributes["ratio"] = new[] { (float)module.output_ratio[0], (float)module.output_ratio[1], (float)module.output_ratio[2] };
            }
            else
            {
                node.Attributes["ratio"] = new[] { 0.5f, 0.5f, 0.5f };
            }

            node.Attributes["sample_index"] = false;

            context.Graph?.AddNode(node);
            return node;
        }
    }
}