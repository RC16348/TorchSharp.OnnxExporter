using System;
using System.Collections.Generic;
using TorchSharp.Modules;
using TorchSharp.OnnxExporter.DataFlow;

namespace TorchSharp.OnnxExporter.Processors
{
    public class SigmoidProcessor : BaseProcessor<Sigmoid>
    {
        public override string OpType => "Sigmoid";

        public override DataFlowNode Process(Sigmoid module, TraceContext context)
        {
            var inputName = context.GetCurrentValue();
            var outputName = context.CreateTempName();

            var node = new DataFlowNode(OpType, new[] { inputName }, new[] { outputName });

            context.Graph?.AddNode(node);
            return node;
        }
    }
}