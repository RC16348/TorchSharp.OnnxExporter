using System;
using System.Collections.Generic;
using TorchSharp.Modules;
using TorchSharp.OnnxExporter.DataFlow;

namespace TorchSharp.OnnxExporter.Processors
{
    public class SoftshrinkProcessor : BaseProcessor<Softshrink>
    {
        public override string OpType => "Softshrink";

        public override DataFlowNode Process(Softshrink module, TraceContext context)
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