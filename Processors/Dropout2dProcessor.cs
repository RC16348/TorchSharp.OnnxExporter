using System;
using System.Collections.Generic;
using TorchSharp.Modules;
using TorchSharp.OnnxExporter.DataFlow;

namespace TorchSharp.OnnxExporter.Processors
{
    public class Dropout2dProcessor : BaseProcessor<Dropout2d>
    {
        public override string OpType => "Dropout";

        public override DataFlowNode Process(Dropout2d module, TraceContext context)
        {
            var inputName = context.GetCurrentValue();
            var outputName = context.CreateTempName();

            var node = new DataFlowNode(OpType, new[] { inputName }, new[] { outputName });

            context.Graph?.AddNode(node);
            return node;
        }
    }
}