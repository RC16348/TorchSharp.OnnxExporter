using System;
using System.Collections.Generic;
using TorchSharp.Modules;
using TorchSharp.OnnxExporter.DataFlow;

namespace TorchSharp.OnnxExporter.Processors
{
    public class LeakyReLUProcessor : BaseProcessor<LeakyReLU>
    {
        public override string OpType => "LeakyRelu";

        public override DataFlowNode Process(LeakyReLU module, TraceContext context)
        {
            var inputName = context.GetCurrentValue();
            var outputName = context.CreateTempName();

            var node = new DataFlowNode(OpType, new[] { inputName }, new[] { outputName });
            node.Attributes["alpha"] = module.negative_slope;

            context.Graph?.AddNode(node);
            return node;
        }
    }
}