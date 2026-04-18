using System;
using System.Collections.Generic;
using TorchSharp.Modules;
using TorchSharp.OnnxExporter.DataFlow;

namespace TorchSharp.OnnxExporter.Processors
{
    public class EmbeddingProcessor : BaseProcessor<Embedding>
    {
        public override string OpType => "Gather";

        public override DataFlowNode Process(Embedding module, TraceContext context)
        {
            var indicesName = context.GetCurrentValue();
            var outputName = context.CreateTempName();

            var weightName = $"{outputName}_weight";

            var node = new DataFlowNode(OpType, new[] { weightName, indicesName }, new[] { outputName });
            node.Attributes["axis"] = 0;

            context.Graph?.AddInitializer(weightName, module.weight);

            context.Graph?.AddNode(node);
            return node;
        }
    }
}