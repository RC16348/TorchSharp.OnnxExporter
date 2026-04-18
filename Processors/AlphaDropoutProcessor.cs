using System;
using System.Collections.Generic;
using TorchSharp.Modules;
using TorchSharp.OnnxExporter.DataFlow;

namespace TorchSharp.OnnxExporter.Processors
{
    public class AlphaDropoutProcessor : BaseProcessor<AlphaDropout>
    {
        public override string OpType => "Dropout";

        public override DataFlowNode Process(AlphaDropout module, TraceContext context)
        {
            var inputName = context.GetCurrentValue();
            var outputName = context.CreateTempName();

            var node = new DataFlowNode(OpType, new[] { inputName }, new[] { outputName });
            node.Attributes["ratio"] = 0.5f;

            context.Graph?.AddNode(node);
            return node;
        }
    }
}