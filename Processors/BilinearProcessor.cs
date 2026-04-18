using System;
using System.Collections.Generic;
using TorchSharp.Modules;
using TorchSharp.OnnxExporter.DataFlow;

namespace TorchSharp.OnnxExporter.Processors
{
    public class BilinearProcessor : BaseProcessor<Bilinear>
    {
        public override string OpType => "Bilinear";

        public override DataFlowNode Process(Bilinear module, TraceContext context)
        {
            var inputName = context.GetCurrentValue();
            var outputName = context.CreateTempName();

            var weightName = $"{outputName}_weight";
            var biasName = $"{outputName}_bias";

            var inputs = new List<string> { inputName, weightName };
            if (!object.ReferenceEquals(module.bias, null))
            {
                inputs.Add(biasName);
            }

            var outputs = new List<string> { outputName };

            var node = new DataFlowNode(OpType, inputs, outputs);

            context.Graph?.AddInitializer(weightName, module.weight);
            if (!object.ReferenceEquals(module.bias, null))
            {
                context.Graph?.AddInitializer(biasName, module.bias);
            }

            context.Graph?.AddNode(node);
            return node;
        }
    }
}