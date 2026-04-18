using System;
using System.Collections.Generic;
using TorchSharp.Modules;
using TorchSharp.OnnxExporter.DataFlow;

namespace TorchSharp.OnnxExporter.Processors
{
    public class ReLU6Processor : BaseProcessor<ReLU6>
    {
        public override string OpType => "Clip";

        public override DataFlowNode Process(ReLU6 module, TraceContext context)
        {
            var inputName = context.GetCurrentValue();
            var outputName = context.CreateTempName();

            var node = new DataFlowNode(OpType, new[] { inputName }, new[] { outputName });
            node.Attributes["min"] = 0.0;
            node.Attributes["max"] = 6.0;

            context.Graph?.AddNode(node);
            return node;
        }
    }
}