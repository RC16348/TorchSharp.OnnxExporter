using System;
using System.Collections.Generic;
using TorchSharp.Modules;
using TorchSharp.OnnxExporter.DataFlow;

namespace TorchSharp.OnnxExporter.Processors
{
    public class SoftmaxProcessor : BaseProcessor<Softmax>
    {
        public override string OpType => "Softmax";

        public override DataFlowNode Process(Softmax module, TraceContext context)
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