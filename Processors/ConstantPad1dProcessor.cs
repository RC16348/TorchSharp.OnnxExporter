using System;
using System.Collections.Generic;
using TorchSharp.Modules;
using TorchSharp.OnnxExporter.DataFlow;

namespace TorchSharp.OnnxExporter.Processors
{
    public class ConstantPad1dProcessor : BaseProcessor<ConstantPad1d>
    {
        public override string OpType => "Pad";

        public override DataFlowNode Process(ConstantPad1d module, TraceContext context)
        {
            var inputName = context.GetCurrentValue();
            var outputName = context.CreateTempName();

            var pads = new List<int>();
            if (module.padding != null && module.padding.Length >= 2)
            {
                pads.Add((int)module.padding[0]);
                pads.Add((int)module.padding[1]);
            }
            else
            {
                pads.AddRange(new[] { 0, 0 });
            }

            var inputs = new List<string> { inputName };
            var outputs = new List<string> { outputName };

            var node = new DataFlowNode(OpType, inputs, outputs);
            node.Attributes["mode"] = "constant";
            node.Attributes["pads"] = pads;
            node.Attributes["value"] = (float)module.value;

            context.Graph?.AddNode(node);
            return node;
        }
    }
}