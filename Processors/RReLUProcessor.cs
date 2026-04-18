using System;
using System.Collections.Generic;
using TorchSharp.Modules;
using TorchSharp.OnnxExporter.DataFlow;

namespace TorchSharp.OnnxExporter.Processors
{
    public class RReLUProcessor : BaseProcessor<RReLU>
    {
        public override string OpType => "LeakyRelu";

        public override DataFlowNode Process(RReLU module, TraceContext context)
        {
            var inputName = context.GetCurrentValue();
            var outputName = context.CreateTempName();

            var node = new DataFlowNode(OpType, new[] { inputName }, new[] { outputName });
            var lower = module.lower;
            var upper = module.upper;
            node.Attributes["alpha"] = (lower + upper) / 2.0;

            context.Graph?.AddNode(node);
            return node;
        }
    }
}