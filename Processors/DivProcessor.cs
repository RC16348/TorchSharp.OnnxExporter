using System.Collections.Generic;
using TorchSharp.OnnxExporter.DataFlow;
using Module = TorchSharp.torch.nn.Module;

namespace TorchSharp.OnnxExporter.Processors
{
    public class DivProcessor : INodeProcessor
    {
        public string OpType => "Div";

        public bool CanProcess(Module module)
        {
            return module.GetType().Name == "Div" || module.GetType().Name == "div";
        }

        public DataFlowNode Process(Module module, TraceContext context)
        {
            var inputs = new List<string>();
            var currentInput = context.GetCurrentValue();
            inputs.Add(currentInput);

            foreach (var child in module.named_children())
            {
                var childOutput = context.GetValue(child.name);
                if (!string.IsNullOrEmpty(childOutput))
                {
                    inputs.Add(childOutput);
                }
            }

            if (inputs.Count < 2)
            {
                inputs.Add(currentInput);
            }

            var outputName = context.CreateTempName();
            var node = new DataFlowNode(OpType, inputs, new[] { outputName });

            context.Graph?.AddNode(node);
            return node;
        }
    }
}
