using System;
using System.Collections.Generic;
using TorchSharp.Modules;
using TorchSharp.OnnxExporter.DataFlow;

namespace TorchSharp.OnnxExporter.Processors
{
    public class LogSoftmaxProcessor : BaseProcessor<LogSoftmax>
    {
        public override string OpType => "LogSoftmax";

        public override DataFlowNode Process(LogSoftmax module, TraceContext context)
        {
            var inputName = context.GetCurrentValue();
            var outputName = context.CreateTempName();

            var node = new DataFlowNode(OpType, new[] { inputName }, new[] { outputName });
            node.Attributes["axis"] = -1;

            context.Graph?.AddNode(node);
            return node;
        }
    }
}