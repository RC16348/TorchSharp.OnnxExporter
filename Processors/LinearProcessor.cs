using System;
using System.Collections.Generic;
using TorchSharp.Modules;
using TorchSharp.OnnxExporter.DataFlow;

namespace TorchSharp.OnnxExporter.Processors
{
    public class LinearProcessor : BaseProcessor<Linear>
    {
        public override string OpType => "Gemm";

        public override DataFlowNode Process(Linear module, TraceContext context)
        {
            var inputName = context.GetCurrentValue();
            var outputName = context.CreateTempName();

            var weightName = $"{outputName}_weight";
            var biasName = $"{outputName}_bias";

            if (context.TryGetShape(inputName, out var inputShape) && inputShape.Count >= 2)
            {
                var n = inputShape[0];
                context.SetShape(outputName, new List<long> { n, module.out_features });
            }

            var inputs = new List<string> { inputName, weightName };
            if (module.bias is not null)
            {
                inputs.Add(biasName);
            }

            var outputs = new List<string> { outputName };

            var node = new DataFlowNode(OpType, inputs, outputs);
            node.Attributes["alpha"] = 1.0f;
            node.Attributes["beta"] = 1.0f;
            node.Attributes["transA"] = 0;
            node.Attributes["transB"] = 1;

            context.Graph?.AddInitializer(weightName, module.weight);
            if (module.bias is not null)
            {
                context.Graph?.AddInitializer(biasName, module.bias);
            }

            context.SetCurrentValue(outputName);
            context.Graph?.AddNode(node);
            return node;
        }
    }
}