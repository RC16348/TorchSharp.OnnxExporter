using System;
using System.Collections.Generic;
using System.Linq;
using TorchSharp.Modules;
using TorchSharp.OnnxExporter.DataFlow;

namespace TorchSharp.OnnxExporter.Processors
{
    public class GRUProcessor : BaseProcessor<GRU>
    {
        public override string OpType => "GRU";

        public override DataFlowNode Process(GRU module, TraceContext context)
        {
            var inputName = context.GetCurrentValue();
            var outputName = context.CreateTempName();
            var hiddenStateName = context.CreateTempName();

            var weightIhName = $"{outputName}_weight_ih";
            var weightHhName = $"{outputName}_weight_hh";
            var biasIhName = $"{outputName}_bias_ih";
            var biasHhName = $"{outputName}_bias_hh";

            var parameters = module.named_parameters().ToDictionary(p => p.name, p => p.parameter);
            parameters.TryGetValue("weight_ih", out var weightIh);
            parameters.TryGetValue("weight_hh", out var weightHh);
            parameters.TryGetValue("bias_ih", out var biasIh);
            parameters.TryGetValue("bias_hh", out var biasHh);

            long hiddenSize = weightHh is not null ? weightHh.shape[^1] : 0L;
            bool bidirectional = parameters.ContainsKey("weight_hh_reverse");

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