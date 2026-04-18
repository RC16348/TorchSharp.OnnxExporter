using System;
using System.Collections.Generic;
using TorchSharp.OnnxExporter.DataFlow;
using Module = TorchSharp.torch.nn.Module;

namespace TorchSharp.OnnxExporter.Processors
{
    public class IdentityProcessor : INodeProcessor
    {
        public string OpType => "Identity";

        public bool CanProcess(Module module)
        {
            var name = module.GetType().Name;
            return name == "Identity";
        }

        public DataFlowNode Process(Module module, TraceContext context)
        {
            var inputName = context.GetCurrentValue();
            var outputName = context.CreateTempName();

            var node = new DataFlowNode(OpType, new[] { inputName }, new[] { outputName });

            context.Graph?.AddNode(node);
            return node;
        }
    }
}