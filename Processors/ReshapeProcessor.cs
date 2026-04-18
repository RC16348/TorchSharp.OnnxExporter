using System;
using System.Linq;
using TorchSharp;
using TorchSharp.OnnxExporter.DataFlow;
using Module = TorchSharp.torch.nn.Module;

namespace TorchSharp.OnnxExporter.Processors
{
    public class ReshapeProcessor : INodeProcessor
    {
        public string OpType => "Reshape";

        public bool CanProcess(Module module)
        {
            return module.GetType().Name == "Reshape";
        }

        public DataFlowNode Process(Module module, TraceContext context)
        {
            var inputName = context.GetCurrentValue();
            var outputName = context.CreateTempName();

            var reshapeModule = (dynamic)module;
            var shapeArray = (long[])reshapeModule.shape;
            int shapeLen = shapeArray.Length;

            var shapeName = $"{outputName}_shape";

            var inputs = new[] { inputName, shapeName };
            var outputs = new[] { outputName };

            var node = new DataFlowNode(OpType, inputs, outputs);

            var floatShape = shapeArray.Select(s => (float)s).ToArray();
            context.Graph?.AddInitializer(shapeName, floatShape, new[] { (long)shapeLen });

            context.SetCurrentValue(outputName);
            context.SetShape(outputName, shapeArray.ToList());

            context.Graph?.AddNode(node);
            return node;
        }
    }
}