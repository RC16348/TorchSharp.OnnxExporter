using System;
using System.Collections.Generic;
using TorchSharp.Modules;
using TorchSharp.OnnxExporter.DataFlow;

namespace TorchSharp.OnnxExporter.Processors
{
    public class Conv1dProcessor : BaseProcessor<Conv1d>
    {
        public override string OpType => "Conv";

        public override DataFlowNode Process(Conv1d module, TraceContext context)
        {
            var inputName = context.GetCurrentValue();
            var outputName = context.CreateTempName();

            var inputs = new List<string> { inputName };
            var outputs = new List<string> { outputName };

            var weightName = $"{outputName}_weight";
            var biasName = $"{outputName}_bias";

            inputs.Add(weightName);
            if (module.bias is not null)
            {
                inputs.Add(biasName);
            }

            var node = new DataFlowNode(OpType, inputs, outputs);

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

            Console.WriteLine($"[DEBUG Conv1d] Processing module, stride.Length={module.stride?.Length ?? -1}");

            int[] strides;
            if (module.stride is not null && module.stride.Length >= 1)
            {
                strides = new[] { (int)module.stride[0] };
            }
            else
            {
                strides = new[] { 1 };
            }
            node.Attributes["strides"] = strides;

            int[] pads;
            if (module.padding is not null && module.padding.Length >= 1)
            {
                int p0 = (int)module.padding[0];
                pads = new[] { p0, p0 };
            }
            else
            {
                pads = new[] { 0, 0 };
            }
            node.Attributes["pads"] = pads;

            node.Attributes["group"] = module.groups > 0 ? (int)module.groups : 1;

            int[] dilations;
            if (module.dilation is not null && module.dilation.Length >= 1)
            {
                dilations = new[] { (int)module.dilation[0] };
            }
            else
            {
                dilations = new[] { 1 };
            }
            node.Attributes["dilations"] = dilations;

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