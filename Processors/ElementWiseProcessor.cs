using System;
using TorchSharp.OnnxExporter.DataFlow;
using Module = TorchSharp.torch.nn.Module;

namespace TorchSharp.OnnxExporter.Processors
{
    public class ElementWiseProcessor : INodeProcessor
    {
        private readonly string opType;

        public ElementWiseProcessor(string opType)
        {
            this.opType = opType;
        }

        public string OpType => opType;

        public bool CanProcess(Module module)
        {
            var typeName = module.GetType().FullName ?? module.GetType().Name;
            return typeName.Contains("Modules.");
        }

        public DataFlowNode Process(Module module, TraceContext context)
        {
            var currentInput = context.GetCurrentValue();
            var outputName = context.CreateTempName();

            var node = new DataFlowNode(OpType,
                new[] { currentInput },
                new[] { outputName });

            context.Graph?.AddNode(node);
            context.SetCurrentValue(outputName);

            return node;
        }
    }
}
