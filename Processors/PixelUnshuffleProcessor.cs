using System;
using System.Collections.Generic;
using TorchSharp.Modules;
using TorchSharp.OnnxExporter.DataFlow;

namespace TorchSharp.OnnxExporter.Processors
{
    public class PixelUnshuffleProcessor : BaseProcessor<PixelUnshuffle>
    {
        public override string OpType => "PixelUnshuffle";

        public override DataFlowNode Process(PixelUnshuffle module, TraceContext context)
        {
            var inputName = context.GetCurrentValue();
            var outputName = context.CreateTempName();

            var node = new DataFlowNode(OpType, new[] { inputName }, new[] { outputName });
            node.Attributes["blocksize"] = module.downscale_factor;

            context.Graph?.AddNode(node);
            return node;
        }
    }
}