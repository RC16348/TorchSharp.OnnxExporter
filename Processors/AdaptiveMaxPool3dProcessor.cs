using System;
using System.Collections.Generic;
using TorchSharp.Modules;
using TorchSharp.OnnxExporter.DataFlow;

namespace TorchSharp.OnnxExporter.Processors
{
    public class AdaptiveMaxPool3dProcessor : BaseProcessor<AdaptiveMaxPool3d>
    {
        public override string OpType => "AdaptiveMaxPool";

        public override DataFlowNode Process(AdaptiveMaxPool3d module, TraceContext context)
        {
            var inputName = context.GetCurrentValue();
            var outputName = context.CreateTempName();

            var node = new DataFlowNode(OpType, new[] { inputName }, new[] { outputName });

            if (module.output_size is not null && module.output_size.Length >= 3)
            {
                var outputShape = new List<int>();
                for (int i = 0; i < 3; i++)
                {
                    outputShape.Add((int)module.output_size[i]);
                }
                node.Attributes["output_shape"] = outputShape;
            }
            else
            {
                node.Attributes["output_shape"] = new[] { 1, 1, 1 };
            }

            context.Graph?.AddNode(node);
            return node;
        }
    }
}