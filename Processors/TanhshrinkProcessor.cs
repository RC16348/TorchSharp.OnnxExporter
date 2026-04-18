using System;
using System.Collections.Generic;
using TorchSharp.Modules;
using TorchSharp.OnnxExporter.DataFlow;

namespace TorchSharp.OnnxExporter.Processors
{
    public class TanhshrinkProcessor : BaseProcessor<Tanhshrink>
    {
        public override string OpType => "Tanhshrink";

        public override DataFlowNode Process(Tanhshrink module, TraceContext context)
        {
            var inputName = context.GetCurrentValue();
            var tanhOutputName = context.CreateTempName();
            var outputName = context.CreateTempName();

            var tanhNode = new DataFlowNode("Tanh", new[] { inputName }, new[] { tanhOutputName });
            context.Graph?.AddNode(tanhNode);

            var subNode = new DataFlowNode("Sub", new[] { inputName, tanhOutputName }, new[] { outputName });
            context.Graph?.AddNode(subNode);

            return subNode;
        }
    }
}