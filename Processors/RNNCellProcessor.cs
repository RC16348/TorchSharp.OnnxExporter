using System;
using System.Collections.Generic;
using TorchSharp.Modules;
using TorchSharp.OnnxExporter.DataFlow;

namespace TorchSharp.OnnxExporter.Processors
{
    public class RNNCellProcessor : BaseProcessor<RNNCell>
    {
        public override string OpType => "RNN";

        public override DataFlowNode Process(RNNCell module, TraceContext context)
        {
            var inputName = context.GetCurrentValue();
            var hiddenStateName = context.GetNextValue();
            var outputName = context.CreateTempName();

            var weightIhName = $"{outputName}_weight_ih";
            var weightHhName = $"{outputName}_weight_hh";
            var biasIhName = $"{outputName}_bias_ih";
            var biasHhName = $"{outputName}_bias_hh";

            var weightIh = module.weight_ih;
            var weightHh = module.weight_hh;
            var biasIh = module.bias_ih;
            var biasHh = module.bias_hh;

            long hiddenSize = weightHh is not null ? weightHh.shape[^1] : 0L;

            var node = new DataFlowNode(OpType,
                new[] { inputName, weightIhName, weightHhName, biasIhName, biasHhName, hiddenStateName },
                new[] { outputName, hiddenStateName });

            node.Attributes["direction"] = "forward";
            node.Attributes["hidden_size"] = hiddenSize;
            node.Attributes["layout"] = 0;
            node.Attributes["linear_before"] = 0;

            if (weightIh is not null) context.Graph?.AddInitializer(weightIhName, weightIh);
            if (weightHh is not null) context.Graph?.AddInitializer(weightHhName, weightHh);
            if (biasIh is not null) context.Graph?.AddInitializer(biasIhName, biasIh);
            if (biasHh is not null) context.Graph?.AddInitializer(biasHhName, biasHh);

            context.Graph?.AddNode(node);
            return node;
        }
    }
}