using System;
using System.Collections.Generic;
using TorchSharp.Modules;
using TorchSharp.OnnxExporter.DataFlow;

namespace TorchSharp.OnnxExporter.Processors
{
    public class EmbeddingBagProcessor : BaseProcessor<EmbeddingBag>
    {
        public override string OpType => "ReduceMean";

        public override DataFlowNode Process(EmbeddingBag module, TraceContext context)
        {
            var inputName = context.GetCurrentValue();
            var outputName = context.CreateTempName();

            var weightName = $"{outputName}_weight";

            var gatherOutputName = context.CreateTempName();
            var gatherNode = new DataFlowNode("Gather", new[] { weightName, inputName }, new[] { gatherOutputName });
            gatherNode.Attributes["axis"] = 0;
            context.Graph?.AddInitializer(weightName, module.weight);
            context.Graph?.AddNode(gatherNode);

            var reduceNode = new DataFlowNode(OpType, new[] { gatherOutputName }, new[] { outputName });
            reduceNode.Attributes["axes"] = new[] { 0 };
            reduceNode.Attributes["keepdims"] = 0;

            context.Graph?.AddNode(reduceNode);
            return reduceNode;
        }
    }
}