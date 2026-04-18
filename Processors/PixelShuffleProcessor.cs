using System;
using TorchSharp.Modules;
using TorchSharp.OnnxExporter.DataFlow;

namespace TorchSharp.OnnxExporter.Processors
{
    public class PixelShuffleProcessor : BaseProcessor<PixelShuffle>
    {
        public override string OpType => "DepthToSpace";

        public override DataFlowNode Process(PixelShuffle module, TraceContext context)
        {
            var inputName = context.GetCurrentValue();
            var outputName = context.CreateTempName();

            var node = new DataFlowNode(OpType, new[] { inputName }, new[] { outputName });
            node.Attributes["blocksize"] = (long)module.upscale_factor;
            node.Attributes["mode"] = "CRD";

            context.Graph?.AddNode(node);
            return node;
        }
    }
}