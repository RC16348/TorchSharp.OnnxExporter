using System;
using System.Collections.Generic;
using TorchSharp.Modules;
using TorchSharp.OnnxExporter.DataFlow;

namespace TorchSharp.OnnxExporter.Processors
{
    public class Dropout3dProcessor : BaseProcessor<Dropout3d>
    {
        public override string OpType => "Dropout";

        public override DataFlowNode Process(Dropout3d module, TraceContext context)
        {
            var inputName = context.GetCurrentValue();
            var outputName = context.CreateTempName();

            var node = new DataFlowNode(OpType, new[] { inputName }, new[] { outputName });
            node.Attributes["ratio"] = (float)module.p;

            context.Graph?.AddNode(node);
            return node;
        }
    }
}