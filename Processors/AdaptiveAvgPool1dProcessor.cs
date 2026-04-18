using System;
using System.Collections.Generic;
using TorchSharp.Modules;
using TorchSharp.OnnxExporter.DataFlow;

namespace TorchSharp.OnnxExporter.Processors
{
    public class AdaptiveAvgPool1dProcessor : BaseProcessor<AdaptiveAvgPool1d>
    {
        public override string OpType => "AdaptiveAveragePool";

        public override DataFlowNode Process(AdaptiveAvgPool1d module, TraceContext context)
        {
            var inputName = context.GetCurrentValue();
            var outputName = context.CreateTempName();

            var node = new DataFlowNode(OpType, new[] { inputName }, new[] { outputName });

            if (module.output_size != null)
            {
                node.Attributes["output_shape"] = new[] { (int)module.output_size };
            }
            else
            {
                node.Attributes["output_shape"] = new[] { 1 };
            }

            context.Graph?.AddNode(node);
            return node;
        }
    }
}