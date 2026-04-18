using System;
using System.Collections.Generic;
using TorchSharp.Modules;
using TorchSharp.OnnxExporter.DataFlow;

namespace TorchSharp.OnnxExporter.Processors
{
    public class HardsigmoidProcessor : BaseProcessor<Hardsigmoid>
    {
        public override string OpType => "HardSigmoid";

        public override DataFlowNode Process(Hardsigmoid module, TraceContext context)
        {
            var inputName = context.GetCurrentValue();
            var outputName = context.CreateTempName();

            if (context.TryGetShape(inputName, out var inputShape))
            {
                context.SetShape(outputName, inputShape);
            }

            var node = new DataFlowNode(OpType, new[] { inputName }, new[] { outputName });
            node.Attributes["alpha"] = 1.0 / 6.0;
            node.Attributes["beta"] = 0.5;

            context.SetCurrentValue(outputName);
            context.Graph?.AddNode(node);
            return node;
        }
    }
}