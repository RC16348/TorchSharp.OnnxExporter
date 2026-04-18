using System;
using System.Collections.Generic;
using TorchSharp.Modules;
using TorchSharp.OnnxExporter.DataFlow;

namespace TorchSharp.OnnxExporter.Processors
{
    public class HardtanhProcessor : BaseProcessor<Hardtanh>
    {
        public override string OpType => "Clip";

        public override DataFlowNode Process(Hardtanh module, TraceContext context)
        {
            var inputName = context.GetCurrentValue();
            var outputName = context.CreateTempName();

            var node = new DataFlowNode(OpType, new[] { inputName }, new[] { outputName });
            node.Attributes["min"] = module.min_val;
            node.Attributes["max"] = module.max_val;

            context.Graph?.AddNode(node);
            return node;
        }
    }
}