using System;
using System.Collections.Generic;
using TorchSharp.Modules;
using TorchSharp.OnnxExporter.DataFlow;

namespace TorchSharp.OnnxExporter.Processors
{
    public class LayerNormProcessor : BaseProcessor<LayerNorm>
    {
        public override string OpType => "LayerNormalization";

        public override DataFlowNode Process(LayerNorm module, TraceContext context)
        {
            var inputName = context.GetCurrentValue();
            var outputName = context.CreateTempName();

            var scaleName = $"{outputName}_scale";
            var biasName = $"{outputName}_bias";

            var inputs = new List<string> { inputName, scaleName };
            if (module.bias is not null)
            {
                inputs.Add(biasName);
            }

            var outputs = new List<string> { outputName };

            var node = new DataFlowNode(OpType, inputs, outputs);
            node.Attributes["epsilon"] = (float)module.eps;
            node.Attributes["axis"] = -1;

            context.Graph?.AddInitializer(scaleName, module.weight);
            if (module.bias is not null)
            {
                context.Graph?.AddInitializer(biasName, module.bias);
            }

            context.Graph?.AddNode(node);
            return node;
        }
    }
}