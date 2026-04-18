using System;
using System.Collections.Generic;
using TorchSharp.Modules;
using TorchSharp.OnnxExporter.DataFlow;

namespace TorchSharp.OnnxExporter.Processors
{
    public class SoftminProcessor : BaseProcessor<Softmin>
    {
        public override string OpType => "Softmin";

        public override DataFlowNode Process(Softmin module, TraceContext context)
        {
            var inputName = context.GetCurrentValue();
            var outputName = context.CreateTempName();

            var node = new DataFlowNode(OpType, new[] { inputName }, new[] { outputName });
            node.Attributes["axis"] = -1;

            context.Graph?.AddNode(node);
            return node;
        }
    }
}