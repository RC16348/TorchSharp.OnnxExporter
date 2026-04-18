using System;
using System.Collections.Generic;
using TorchSharp.Modules;
using TorchSharp.OnnxExporter.DataFlow;

namespace TorchSharp.OnnxExporter.Processors
{
    public class HardshrinkProcessor : BaseProcessor<Hardshrink>
    {
        public override string OpType => "Hardshrink";

        public override DataFlowNode Process(Hardshrink module, TraceContext context)
        {
            var inputName = context.GetCurrentValue();
            var outputName = context.CreateTempName();

            var node = new DataFlowNode(OpType, new[] { inputName }, new[] { outputName });
            node.Attributes["lambda"] = module.lambda;

            context.Graph?.AddNode(node);
            return node;
        }
    }
}