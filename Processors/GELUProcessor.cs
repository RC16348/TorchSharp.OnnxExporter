using System;
using System.Collections.Generic;
using TorchSharp.Modules;
using TorchSharp.OnnxExporter.DataFlow;

namespace TorchSharp.OnnxExporter.Processors
{
    public class GELUProcessor : BaseProcessor<GELU>
    {
        public override string OpType => "Gelu";

        public override DataFlowNode Process(GELU module, TraceContext context)
        {
            var inputName = context.GetCurrentValue();
            var outputName = context.CreateTempName();

            var xSquaredName = context.CreateTempName();
            var xCubedName = context.CreateTempName();
            var xCubedScaledName = context.CreateTempName();
            var xPlusCubedScaledName = context.CreateTempName();
            var innerProductName = context.CreateTempName();
            var tanhOutputName = context.CreateTempName();
            var onePlusTanhName = context.CreateTempName();
            var halfTimesOnePlusTanhName = context.CreateTempName();

            var mulNode1 = new DataFlowNode("Mul", new[] { inputName, inputName }, new[] { xSquaredName });
            context.Graph?.AddNode(mulNode1);

            var mulNode2 = new DataFlowNode("Mul", new[] { xSquaredName, inputName }, new[] { xCubedName });
            context.Graph?.AddNode(mulNode2);

            var const044715Name = $"{outputName}_c044715";
            var mulNode3 = new DataFlowNode("Mul", new[] { xCubedName, const044715Name }, new[] { xCubedScaledName });
            context.Graph?.AddInitializer(const044715Name, 0.044715f);
            context.Graph?.AddNode(mulNode3);

            var addNode1 = new DataFlowNode("Add", new[] { inputName, xCubedScaledName }, new[] { xPlusCubedScaledName });
            context.Graph?.AddNode(addNode1);

            var constSqrt2PiName = $"{outputName}_sqrt2pi";
            var mulNode4 = new DataFlowNode("Mul", new[] { xPlusCubedScaledName, constSqrt2PiName }, new[] { innerProductName });
            context.Graph?.AddInitializer(constSqrt2PiName, (float)Math.Sqrt(2.0 / Math.PI));
            context.Graph?.AddNode(mulNode4);

            var tanhNode = new DataFlowNode("Tanh", new[] { innerProductName }, new[] { tanhOutputName });
            context.Graph?.AddNode(tanhNode);

            var constOneName = $"{outputName}_one";
            var addNode2 = new DataFlowNode("Add", new[] { tanhOutputName, constOneName }, new[] { onePlusTanhName });
            context.Graph?.AddInitializer(constOneName, 1.0f);
            context.Graph?.AddNode(addNode2);

            var constHalfName = $"{outputName}_half";
            var mulNode5 = new DataFlowNode("Mul", new[] { onePlusTanhName, constHalfName }, new[] { halfTimesOnePlusTanhName });
            context.Graph?.AddInitializer(constHalfName, 0.5f);
            context.Graph?.AddNode(mulNode5);

            var mulNode6 = new DataFlowNode("Mul", new[] { inputName, halfTimesOnePlusTanhName }, new[] { outputName });
            context.Graph?.AddNode(mulNode6);

            return mulNode6;
        }
    }
}