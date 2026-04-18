using System;
using System.Collections.Generic;
using TorchSharp.Modules;
using TorchSharp.OnnxExporter.DataFlow;

namespace TorchSharp.OnnxExporter.Processors
{
    public class ConstantPad2dProcessor : BaseProcessor<ConstantPad2d>
    {
        public override string OpType => "Pad";

        public override DataFlowNode Process(ConstantPad2d module, TraceContext context)
        {
            var inputName = context.GetCurrentValue();
            var outputName = context.CreateTempName();

            long padH = 0, padW = 0;
            if (module.padding != null)
            {
                if (module.padding.Length == 4)
                {
                    padH = (long)module.padding[0];
                    padW = (long)module.padding[1];
                }
                else if (module.padding.Length == 2)
                {
                    padH = (long)module.padding[0];
                    padW = (long)module.padding[0];
                }
            }

            var padsList = new long[] { 0, 0, padH, padW, 0, 0, padH, padW };

            var padsName = $"const_pads_{context.CreateTempName()}";
            context.Graph?.AddInitializer(padsName, padsList);

            var valueName = $"const_value_{context.CreateTempName()}";
            context.Graph?.AddInitializer(valueName, new float[] { (float)module.value }, new long[] { 1 });

            var inputs = new List<string> { inputName, padsName, valueName };
            var outputs = new List<string> { outputName };

            var node = new DataFlowNode(OpType, inputs, outputs);
            node.Attributes["mode"] = "constant";

            context.Graph?.AddNode(node);
            return node;
        }
    }
}