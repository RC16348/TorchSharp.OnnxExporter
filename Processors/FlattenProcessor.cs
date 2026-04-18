using System;
using System.Collections.Generic;
using TorchSharp.Modules;
using TorchSharp.OnnxExporter.DataFlow;

namespace TorchSharp.OnnxExporter.Processors
{
    public class FlattenProcessor : BaseProcessor<Flatten>
    {
        public override string OpType => "Flatten";

        public override DataFlowNode Process(Flatten module, TraceContext context)
        {
            var inputName = context.GetCurrentValue();
            var outputName = context.CreateTempName();

            var node = new DataFlowNode(OpType, new[] { inputName }, new[] { outputName });
            node.Attributes["axis"] = 1;

            if (context.TryGetShape(inputName, out var inputShape))
            {
                long n = inputShape.Count > 0 ? inputShape[0] : 1;
                long flattenedSize = 1;
                for (int i = 1; i < inputShape.Count; i++)
                {
                    flattenedSize *= inputShape[i];
                }
                context.SetShape(outputName, new List<long> { n, flattenedSize });
            }

            context.SetCurrentValue(outputName);
            context.Graph?.AddNode(node);
            return node;
        }
    }
}