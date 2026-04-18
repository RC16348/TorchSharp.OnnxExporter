using System;
using System.Collections.Generic;
using TorchSharp.Modules;
using TorchSharp.OnnxExporter.DataFlow;

namespace TorchSharp.OnnxExporter.Processors
{
    public class MishProcessor : BaseProcessor<Mish>
    {
        public override string OpType => "Mish";

        public override DataFlowNode Process(Mish module, TraceContext context)
        {
            var inputName = context.GetCurrentValue();
            var softplusOutputName = context.CreateTempName();
            var tanhOutputName = context.CreateTempName();
            var outputName = context.CreateTempName();

            var softplusNode = new DataFlowNode("Softplus", new[] { inputName }, new[] { softplusOutputName });
            context.Graph?.AddNode(softplusNode);

            var tanhNode = new DataFlowNode("Tanh", new[] { softplusOutputName }, new[] { tanhOutputName });
            context.Graph?.AddNode(tanhNode);

            var mulNode = new DataFlowNode("Mul", new[] { inputName, tanhOutputName }, new[] { outputName });
            context.Graph?.AddNode(mulNode);

            return mulNode;
        }
    }
}