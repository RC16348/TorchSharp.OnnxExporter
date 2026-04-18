using System;
using System.Collections.Generic;
using TorchSharp.Modules;
using TorchSharp.OnnxExporter.DataFlow;

namespace TorchSharp.OnnxExporter.Processors
{
    public class GroupNormProcessor : BaseProcessor<GroupNorm>
    {
        public override string OpType => "GroupNormalization";

        public override DataFlowNode Process(GroupNorm module, TraceContext context)
        {
            var inputName = context.GetCurrentValue();
            var outputName = context.CreateTempName();

            var scaleName = $"{outputName}_scale";
            var biasName = $"{outputName}_bias";

            var inputs = new List<string> { inputName, scaleName, biasName };
            var outputs = new List<string> { outputName };

            var node = new DataFlowNode(OpType, inputs, outputs);
            node.Attributes["epsilon"] = (float)module.eps;
            node.Attributes["num_groups"] = (int)module.num_groups;

            if (module.weight is not null)
            {
                context.Graph?.AddInitializer(scaleName, module.weight);
            }

            if (module.bias is not null)
            {
                context.Graph?.AddInitializer(biasName, module.bias);
            }

            context.Graph?.AddNode(node);
            return node;
        }
    }
}