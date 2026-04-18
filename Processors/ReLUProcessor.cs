using System;
using System.Collections.Generic;
using TorchSharp.Modules;
using TorchSharp.OnnxExporter.DataFlow;

namespace TorchSharp.OnnxExporter.Processors
{
    public class ReLUProcessor : BaseProcessor<ReLU>
    {
        public override string OpType => "Relu";

        public override DataFlowNode Process(ReLU module, TraceContext context)
        {
            var inputName = context.GetCurrentValue();
            var outputName = context.CreateTempName();

            if (context.TryGetShape(inputName, out var inputShape))
            {
                context.SetShape(outputName, inputShape);
            }

            var node = new DataFlowNode(OpType, new[] { inputName }, new[] { outputName });

            context.SetCurrentValue(outputName);
            context.Graph?.AddNode(node);
            return node;
        }
    }
}