using System;
using System.Collections.Generic;
using TorchSharp.Modules;
using TorchSharp.OnnxExporter.DataFlow;

namespace TorchSharp.OnnxExporter.Processors
{
    public class SiLUProcessor : BaseProcessor<SiLU>
    {
        public override string OpType => "SiLU";

        public override DataFlowNode Process(SiLU module, TraceContext context)
        {
            var inputName = context.GetCurrentValue();
            var sigmoidOutputName = context.CreateTempName();
            var outputName = context.CreateTempName();

            var sigmoidNode = new DataFlowNode("Sigmoid", new[] { inputName }, new[] { sigmoidOutputName });
            context.Graph?.AddNode(sigmoidNode);

            var mulNode = new DataFlowNode("Mul", new[] { inputName, sigmoidOutputName }, new[] { outputName });
            context.Graph?.AddNode(mulNode);

            return mulNode;
        }
    }
}