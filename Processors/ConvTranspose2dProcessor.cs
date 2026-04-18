using System;
using System.Collections.Generic;
using System.Linq;
using TorchSharp.Modules;
using TorchSharp.OnnxExporter.DataFlow;

namespace TorchSharp.OnnxExporter.Processors
{
    public class ConvTranspose2dProcessor : BaseProcessor<ConvTranspose2d>
    {
        public override string OpType => "ConvTranspose";

        public override DataFlowNode Process(ConvTranspose2d module, TraceContext context)
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
                if (weightShape != null && weightShape.Length >= 4)
                {
                    node.Attributes["kernel_shape"] = new[] { (int)weightShape[2], (int)weightShape[3] };
                }
                else
                {
                    node.Attributes["kernel_shape"] = new[] { 3, 3 };
                }
            }
            else
            {
                node.Attributes["kernel_shape"] = new[] { 3, 3 };
            }

            int[] dilations;
            if (module.dilation is not null && module.dilation.Length >= 2)
            {
                dilations = new[] { (int)module.dilation[0], (int)module.dilation[1] };
            }
            else if (module.dilation is not null && module.dilation.Length == 1)
            {
                int d = (int)module.dilation[0];
                dilations = new[] { d, d };
            }
            else
            {
                dilations = new[] { 1, 1 };
            }
            node.Attributes["dilations"] = dilations;

            int[] strides;
            if (module.stride is not null && module.stride.Length >= 2)
            {
                strides = new[] { (int)module.stride[0], (int)module.stride[1] };
            }
            else
            {
                strides = new[] { 1, 1 };
            }
            node.Attributes["strides"] = strides;

            int[] kernelShape = (int[])node.Attributes["kernel_shape"];
            int kernel_h = kernelShape[0];
            int kernel_w = kernelShape[1];

            int stride_h = strides[0];
            int stride_w = strides[1];

            int[] modulePads = module.padding is not null && module.padding.Length >= 2
                ? new[] { (int)module.padding[0], (int)module.padding[1] }
                : new[] { 0, 0 };

            int pad_h = modulePads[0];
            int pad_w = modulePads[1];

            int output_padding_h = stride_h * (pad_h + 1) - kernel_h;
            int output_padding_w = stride_w * (pad_w + 1) - kernel_w;

            if (output_padding_h < 0) output_padding_h = 0;
            if (output_padding_w < 0) output_padding_w = 0;

            node.Attributes["pads"] = modulePads.Concat(modulePads).ToArray();
            node.Attributes["output_padding"] = new[] { output_padding_h, output_padding_w };

            node.Attributes["group"] = module.groups > 0 ? (int)module.groups : 1;

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