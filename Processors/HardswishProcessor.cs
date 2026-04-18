using System;
using System.Collections.Generic;
using TorchSharp.Modules;
using TorchSharp.OnnxExporter.DataFlow;

namespace TorchSharp.OnnxExporter.Processors
{
    public class HardswishProcessor : BaseProcessor<Hardswish>
    {
        public override string OpType => "HardSwish";

        public override DataFlowNode Process(Hardswish module, TraceContext context)
        {
            var inputName = context.GetCurrentValue();
            Console.WriteLine($"[DEBUG HardswishProcessor] input={inputName}, hasShape={context.TryGetShape(inputName, out var s)}");
            var outputName = context.CreateTempName();
            Console.WriteLine($"[DEBUG HardswishProcessor] output={outputName}");

            if (context.TryGetShape(inputName, out var inputShape))
            {
                context.SetShape(outputName, inputShape);
                Console.WriteLine($"[DEBUG HardswishProcessor] Set shape for {outputName}");
            }
            else
            {
                Console.WriteLine($"[DEBUG HardswishProcessor] WARNING: no shape for {inputName}");
            }

            var node = new DataFlowNode(OpType, new[] { inputName }, new[] { outputName });

            context.SetCurrentValue(outputName);
            context.Graph?.AddNode(node);
            return node;
        }
    }
}