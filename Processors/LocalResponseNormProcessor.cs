using System;
using System.Collections.Generic;
using TorchSharp.Modules;
using TorchSharp.OnnxExporter.DataFlow;

namespace TorchSharp.OnnxExporter.Processors
{
    public class LocalResponseNormProcessor : BaseProcessor<LocalResponseNorm>
    {
        public override string OpType => "LRN";

        public override DataFlowNode Process(LocalResponseNorm module, TraceContext context)
        {
            var inputName = context.GetCurrentValue();
            var outputName = context.CreateTempName();

            var node = new DataFlowNode(OpType, new[] { inputName }, new[] { outputName });

            node.Attributes["alpha"] = (float)module.alpha;
            node.Attributes["beta"] = (float)module.beta;
            node.Attributes["bias"] = (float)module.k;
            node.Attributes["size"] = module.size;

            context.Graph?.AddNode(node);
            return node;
        }
    }
}