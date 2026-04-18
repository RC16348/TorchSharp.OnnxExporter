using System;
using System.Collections.Generic;
using TorchSharp.Modules;
using TorchSharp.OnnxExporter.DataFlow;

namespace TorchSharp.OnnxExporter.Processors
{
    public class LogSigmoidProcessor : BaseProcessor<LogSigmoid>
    {
        public override string OpType => "LogSigmoid";

        public override DataFlowNode Process(LogSigmoid module, TraceContext context)
        {
            var inputName = context.GetCurrentValue();
            var sigmoidOutputName = context.CreateTempName();
            var outputName = context.CreateTempName();

            var sigmoidNode = new DataFlowNode("Sigmoid", new[] { inputName }, new[] { sigmoidOutputName });
            context.Graph?.AddNode(sigmoidNode);

            var logNode = new DataFlowNode("Log", new[] { sigmoidOutputName }, new[] { outputName });
            context.Graph?.AddNode(logNode);

            return logNode;
        }
    }
}