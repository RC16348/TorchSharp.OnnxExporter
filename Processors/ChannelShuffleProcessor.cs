using System;
using System.Collections.Generic;
using TorchSharp.Modules;
using TorchSharp.OnnxExporter.DataFlow;

namespace TorchSharp.OnnxExporter.Processors
{
    public class ChannelShuffleProcessor : BaseProcessor<ChannelShuffle>
    {
        public override string OpType => "ChannelShuffle";

        public override DataFlowNode Process(ChannelShuffle module, TraceContext context)
        {
            var inputName = context.GetCurrentValue();
            var outputName = context.CreateTempName();

            var node = new DataFlowNode(OpType, new[] { inputName }, new[] { outputName });

            node.Attributes["group"] = (int)module.groups;

            context.Graph?.AddNode(node);
            return node;
        }
    }
}