using System;
using System.Collections.Generic;
using TorchSharp.Modules;
using TorchSharp.OnnxExporter.DataFlow;

namespace TorchSharp.OnnxExporter.Processors
{
    public class CELUProcessor : BaseProcessor<CELU>
    {
        public override string OpType => "Celu";

        public override DataFlowNode Process(CELU module, TraceContext context)
        {
            var inputName = context.GetCurrentValue();
            var outputName = context.CreateTempName();

            var node = new DataFlowNode(OpType, new[] { inputName }, new[] { outputName });
            node.Attributes["alpha"] = module.alpha;

            context.Graph?.AddNode(node);
            return node;
        }
    }
}