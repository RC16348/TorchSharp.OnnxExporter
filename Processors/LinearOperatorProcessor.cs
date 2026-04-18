using System;
using System.Linq;
using TorchSharp.OnnxExporter.DataFlow;
using Module = TorchSharp.torch.nn.Module;

namespace TorchSharp.OnnxExporter.Processors
{
    public class LinearOperatorProcessor : INodeProcessor
    {
        public string OpType => "LinearOperator";

        public bool CanProcess(Module module)
        {
            return module.GetType().Name == "LinearOperator";
        }

        public DataFlowNode Process(Module module, TraceContext context)
        {
            var inputName = context.GetCurrentValue();
            var outputName = context.CreateTempName();

            var linearOp = (dynamic)module;
            var weight = linearOp.weight;
            var bias = linearOp.bias;

            var weightName = $"{module.GetType().Name}_weight";
            var biasName = $"{module.GetType().Name}_bias";

            if (!context.Graph!.Initializers.Any(i => i.Name == weightName))
            {
                context.Graph?.AddInitializer(weightName, weight);
            }
            if (!object.ReferenceEquals(bias, null) && !context.Graph!.Initializers.Any(i => i.Name == biasName))
            {
                context.Graph?.AddInitializer(biasName, bias);
            }

            var matmulOutputName = context.CreateTempName();
            var matmulInputs = new[] { inputName, weightName };
            var matmulOutputs = new[] { matmulOutputName };
            var matmulNode = new DataFlowNode("MatMul", matmulInputs, matmulOutputs);
            context.Graph?.AddNode(matmulNode);

            var addInputs = new[] { matmulOutputName, biasName };
            var addOutputs = new[] { outputName };
            var addNode = new DataFlowNode("Add", addInputs, addOutputs);
            context.Graph?.AddNode(addNode);

            context.SetCurrentValue(outputName);

            return addNode;
        }
    }
}