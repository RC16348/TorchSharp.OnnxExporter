using System;
using System.Collections.Generic;
using TorchSharp;
using TorchSharp.OnnxExporter.DataFlow;
using Module = TorchSharp.torch.nn.Module;

namespace TorchSharp.OnnxExporter.Processors
{
    public class SqueezeProcessor : INodeProcessor
    {
        public string OpType => "Squeeze";

        public bool CanProcess(Module module)
        {
            return module.GetType().Name == "Squeeze";
        }

        public DataFlowNode Process(Module module, TraceContext context)
        {
            var inputName = context.GetNextValue();

            var squeezeModule = (dynamic)module;
            var dim = (int)squeezeModule.dim;

            var outputName = context.CreateTempName();
            var inputs = new[] { inputName };
            var outputs = new[] { outputName };

            var node = new DataFlowNode(OpType, inputs, outputs);
            node.Attributes["axes"] = new[] { dim };

            context.Graph?.AddNode(node);
            context.SetCurrentValue(outputName);
            return node;
        }
    }
}