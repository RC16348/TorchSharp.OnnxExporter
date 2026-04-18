using System;
using System.Collections.Generic;
using TorchSharp.Modules;
using TorchSharp.OnnxExporter.DataFlow;

namespace TorchSharp.OnnxExporter.Processors
{
    public class ConvTranspose1dProcessor : BaseProcessor<ConvTranspose1d>
    {
        public override string OpType => "ConvTranspose";

        public override DataFlowNode Process(ConvTranspose1d module, TraceContext context)
        {
            var inputName = context.GetCurrentValue();
            var outputName = context.CreateTempName();

            var weightName = $"{outputName}_weight";
            var biasName = $"{outputName}_bias";

            var inputs = new List<string> { inputName, weightName };
            if (module.bias is not null)
            {
                inputs.Add(biasName);
            }

            var node = new DataFlowNode(OpType, inputs, new[] { outputName });

            node.Attributes["auto_pad"] = "NOTSET";

            if (module.weight is not null)
            {
                var weightShape = module.weight.shape;
                if (weightShape != null && weightShape.Length >= 3)
                {
                    node.Attributes["kernel_shape"] = new[] { (int)weightShape[2] };
                }
                else
                {
                    node.Attributes["kernel_shape"] = new[] { 3 };
                }
            }
            else
            {
                node.Attributes["kernel_shape"] = new[] { 3 };
            }

            if (module.stride is not null && module.stride.Length >= 1)
            {
                int s0 = (int)module.stride[0];
                node.Attributes["strides"] = new[] { 1, s0 };
            }
            else
            {
                node.Attributes["strides"] = new[] { 1, 1 };
            }

            int[] pads;
            if (module.padding is not null && module.padding.Length >= 1)
            {
                int p0 = (int)module.padding[0];
                pads = new[] { 0, p0, 0, p0 };
            }
            else
            {
                pads = new[] { 0, 1, 0, 1 };
            }
            node.Attributes["pads"] = pads;

            node.Attributes["group"] = module.groups > 0 ? (int)module.groups : 1;

            node.Attributes["dilations"] = new[] { 1, 1 };

            context.Graph?.AddInitializer(weightName, module.weight);
            if (module.bias is not null)
            {
                context.Graph?.AddInitializer(biasName, module.bias);
            }

            context.Graph?.AddNode(node);
            return node;
        }
    }
}