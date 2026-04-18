using System;
using System.Collections.Generic;
using System.Linq;
using TorchSharp;
using TorchSharp.OnnxExporter.DataFlow;
using Module = TorchSharp.torch.nn.Module;

namespace TorchSharp.OnnxExporter.Processors
{
    public class ChunkProcessor : INodeProcessor
    {
        public string OpType => "Split";

        public bool CanProcess(Module module)
        {
            return module.GetType().Name == "Chunk";
        }

        public DataFlowNode Process(Module module, TraceContext context)
        {
            var inputName = context.GetCurrentValue();
            var outputName = context.CreateTempName();

            var chunkModule = (dynamic)module;
            int chunks = chunkModule.chunks;
            int dim = (int)chunkModule.dim;

            var splitDimName = $"{outputName}_splitDim";
            var numSplitsName = $"{outputName}_numSplits";

            var inputs = new[] { inputName, splitDimName, numSplitsName };
            var outputs = Enumerable.Range(0, chunks).Select(i => $"{outputName}_{i}").ToArray();

            var node = new DataFlowNode(OpType, inputs, outputs);
            node.Attributes["axis"] = dim;

            context.Graph?.AddInitializer(splitDimName, new long[] { dim });
            context.Graph?.AddInitializer(numSplitsName, new long[] { chunks });

            context.Graph?.AddNode(node);
            return node;
        }
    }
}