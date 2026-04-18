using System;
using System.Collections.Generic;
using TorchSharp.Modules;
using TorchSharp.OnnxExporter.DataFlow;

namespace TorchSharp.OnnxExporter.Processors
{
    public class ConvTranspose3dProcessor : BaseProcessor<ConvTranspose3d>
    {
        public override string OpType => "ConvTranspose";

        public override DataFlowNode Process(ConvTranspose3d module, TraceContext context)
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
                if (weightShape != null && weightShape.Length >= 5)
                {
                    node.Attributes["kernel_shape"] = new[] { (int)weightShape[2], (int)weightShape[3], (int)weightShape[4] };
                }
                else
                {
                    node.Attributes["kernel_shape"] = new[] { 3, 3, 3 };
                }
            }
            else
            {
                node.Attributes["kernel_shape"] = new[] { 3, 3, 3 };
            }

            if (module.stride is not null && module.stride.Length >= 3)
            {
                node.Attributes["strides"] = new[] { (int)module.stride[0], (int)module.stride[1], (int)module.stride[2] };
            }
            else
            {
                node.Attributes["strides"] = new[] { 1, 1, 1 };
            }

            var pads = new List<int>();
            if (module.padding is not null && module.padding.Length >= 3)
            {
                pads.Add((int)module.padding[0]);
                pads.Add((int)module.padding[1]);
                pads.Add((int)module.padding[2]);
                pads.Add((int)module.padding[0]);
                pads.Add((int)module.padding[1]);
                pads.Add((int)module.padding[2]);
            }
            else
            {
                pads.AddRange(new[] { 1, 1, 1, 1, 1, 1 });
            }
            node.Attributes["pads"] = pads;

            node.Attributes["group"] = module.groups > 0 ? (int)module.groups : 1;

            node.Attributes["dilations"] = new[] { 1, 1, 1 };

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