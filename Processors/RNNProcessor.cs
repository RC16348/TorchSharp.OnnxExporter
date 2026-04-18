using System;
using System.Collections.Generic;
using System.Linq;
using TorchSharp.Modules;
using TorchSharp.OnnxExporter.DataFlow;

namespace TorchSharp.OnnxExporter.Processors
{
    public class RNNProcessor : BaseProcessor<RNN>
    {
        public override string OpType => "RNN";

        public override DataFlowNode Process(RNN module, TraceContext context)
        {
            var inputName = context.GetCurrentValue();
            var outputName = context.CreateTempName();
            var hiddenStateName = context.CreateTempName();

            var weightIhName = $"{outputName}_weight_ih";
            var weightHhName = $"{outputName}_weight_hh";
            var biasIhName = $"{outputName}_bias_ih";
            var biasHhName = $"{outputName}_bias_hh";

            var weightIh = module.get_weight_ih(0);
            var weightHh = module.get_weight_hh(0);
            var biasIh = module.get_bias_ih(0);
            var biasHh = module.get_bias_hh(0);

            long hiddenSize = weightHh is not null ? weightHh.shape[^1] : 0L;
            bool bidirectional = weightIh is not null && weightIh.shape[0] == 2;

            var node = new DataFlowNode(OpType,
                new[] { inputName, weightIhName, weightHhName, biasIhName, biasHhName, hiddenStateName },
                new[] { outputName, hiddenStateName });

            node.Attributes["direction"] = bidirectional ? "bidirectional" : "forward";
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