using System;
using System.Collections.Generic;
using TorchSharp.Modules;
using TorchSharp.OnnxExporter.DataFlow;

namespace TorchSharp.OnnxExporter.Processors
{
    public class CosineSimilarityProcessor : BaseProcessor<CosineSimilarity>
    {
        public override string OpType => "Cosine";

        public override DataFlowNode Process(CosineSimilarity module, TraceContext context)
        {
            var inputName = context.GetCurrentValue();
            var outputName = context.CreateTempName();

            var inputs = new List<string> { inputName };
            var outputs = new List<string> { outputName };

            var node = new DataFlowNode(OpType, inputs, outputs);
            node.Attributes["axis"] = (int)module.dim;
            node.Attributes["eps"] = (float)module.eps;

            context.Graph?.AddNode(node);
            return node;
        }
    }
}