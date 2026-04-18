using System;
using System.Collections.Generic;
using TorchSharp.Modules;
using TorchSharp.OnnxExporter.DataFlow;

namespace TorchSharp.OnnxExporter.Processors
{
    public class AdaptiveMaxPool1dProcessor : BaseProcessor<AdaptiveMaxPool1d>
    {
        public override string OpType => "AdaptiveMaxPool";

        public override DataFlowNode Process(AdaptiveMaxPool1d module, TraceContext context)
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