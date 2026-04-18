using System;
using System.Collections.Generic;
using TorchSharp.Modules;
using TorchSharp.OnnxExporter.DataFlow;

namespace TorchSharp.OnnxExporter.Processors
{
    public class BatchNormProcessor : BaseProcessor<BatchNorm2d>
    {
        public override string OpType => "BatchNormalization";

        public override DataFlowNode Process(BatchNorm2d module, TraceContext context)
        {
            var inputName = context.GetCurrentValue();
            var outputName = context.CreateTempName();

            if (context.TryGetShape(inputName, out var inputShape))
            {
                context.SetShape(outputName, inputShape);
            }

            var scaleName = $"{outputName}_scale";
            var biasName = $"{outputName}_bias";
            var meanName = $"{outputName}_mean";
            var varName = $"{outputName}_var";

            var inputs = new List<string> { inputName, scaleName };
            if (module.bias is not null)
            {
                inputs.Add(biasName);
            }
            inputs.Add(meanName);
            inputs.Add(varName);

            var outputs = new List<string> { outputName };

            var node = new DataFlowNode(OpType, inputs, outputs);
            node.Attributes["epsilon"] = (float)module.eps;
            node.Attributes["momentum"] = 0.9f;

            context.Graph?.AddInitializer(scaleName, module.weight);
            if (module.bias is not null)
            {
                context.Graph?.AddInitializer(biasName, module.bias);
            }
            context.Graph?.AddInitializer(meanName, module.running_mean);
            context.Graph?.AddInitializer(varName, module.running_var);

            context.SetCurrentValue(outputName);
            context.Graph?.AddNode(node);
            return node;
        }
    }
}