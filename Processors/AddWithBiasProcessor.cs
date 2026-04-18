using System;
using System.Linq;
using TorchSharp.OnnxExporter.DataFlow;
using Module = TorchSharp.torch.nn.Module;

namespace TorchSharp.OnnxExporter.Processors
{
    public class AddWithBiasProcessor : INodeProcessor
    {
        public string OpType => "AddWithBias";

        public bool CanProcess(Module module)
        {
            return module.GetType().Name == "AddWithBias";
        }

        public DataFlowNode Process(Module module, TraceContext context)
        {
            var inputName = context.GetCurrentValue();
            var outputName = context.CreateTempName();

            var addWithBias = (dynamic)module;
            var bias = addWithBias.bias;

            var biasName = $"{module.GetType().Name}_bias";

            if (!context.Graph!.Initializers.Any(i => i.Name == biasName))
            {
                context.Graph?.AddInitializer(biasName, bias);
            }

            var inputs = new[] { inputName, biasName };
            var outputs = new[] { outputName };
            var node = new DataFlowNode("Add", inputs, outputs);
            context.Graph?.AddNode(node);

            context.SetCurrentValue(outputName);

            return node;
        }
    }
}