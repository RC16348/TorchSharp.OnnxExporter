using System;
using System.Collections.Generic;
using TorchSharp.Modules;
using TorchSharp.OnnxExporter.DataFlow;

namespace TorchSharp.OnnxExporter.Processors
{
    public class ThresholdProcessor : BaseProcessor<Threshold>
    {
        public override string OpType => "Clip";

        public override DataFlowNode Process(Threshold module, TraceContext context)
        {
            var inputName = context.GetCurrentValue();
            var outputName = context.CreateTempName();

            var thresholdName = $"threshold_{context.CreateTempName()}";

            context.Graph?.AddInitializer(thresholdName, new float[] { (float)module.threshold }, new long[] { 1 });

            var greaterOutput = context.CreateTempName();
            var greaterNode = new DataFlowNode("Greater", new[] { inputName, thresholdName }, new[] { greaterOutput });
            context.Graph?.AddNode(greaterNode);

            var notOutput = context.CreateTempName();
            var notNode = new DataFlowNode("Not", new[] { greaterOutput }, new[] { notOutput });
            context.Graph?.AddNode(notNode);

            var valueName = $"threshold_value_{context.CreateTempName()}";
            context.Graph?.AddInitializer(valueName, new float[] { (float)module.value }, new long[] { 1 });

            var node = new DataFlowNode("Where", new[] { notOutput, valueName, inputName }, new[] { outputName });
            context.SetCurrentValue(outputName);
            context.Graph?.AddNode(node);

            return node;
        }
    }
}