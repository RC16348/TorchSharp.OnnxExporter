using System;
using System.Collections.Generic;
using TorchSharp.OnnxExporter.DataFlow;
using Module = TorchSharp.torch.nn.Module;

namespace TorchSharp.OnnxExporter.Processors
{
    public class ConcatProcessor : INodeProcessor
    {
        public string OpType => "Concat";

        public bool CanProcess(Module module)
        {
            var name = module.GetType().Name;
            return name == "Concat" || name == "Concatenate";
        }

        public DataFlowNode Process(Module module, TraceContext context)
        {
            var inputName = context.GetCurrentValue();
            var outputName = context.CreateTempName();

            var node = new DataFlowNode(OpType, new[] { inputName }, new[] { outputName });
            node.Attributes["axis"] = 1;

            context.Graph?.AddNode(node);
            return node;
        }
    }
}