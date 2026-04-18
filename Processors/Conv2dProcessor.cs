using System;
using System.Collections.Generic;
using TorchSharp.Modules;
using TorchSharp.OnnxExporter.DataFlow;

namespace TorchSharp.OnnxExporter.Processors
{
    public class Conv2dProcessor : BaseProcessor<Conv2d>
    {
        public override string OpType => "Conv";

        public override DataFlowNode Process(Conv2d module, TraceContext context)
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

            int[] strides;
            if (module.stride is not null && module.stride.Length >= 2)
            {
                strides = new[] { (int)module.stride[0], (int)module.stride[1] };
            }
            else if (module.stride is not null && module.stride.Length == 1)
            {
                int stride1 = (int)module.stride[0];
                strides = new[] { stride1, stride1 };
            }
            else if (module.stride is not null && module.stride.Length > 0)
            {
                int stride1 = (int)module.stride[0];
                strides = new[] { stride1, stride1 };
            }
            else
            {
                strides = new[] { 1, 1 };
            }
            node.Attributes["strides"] = strides;
            Console.WriteLine($"[DEBUG Conv2d] Node={node.Name} Set strides=[{strides[0]},{strides[1]}]");

            int[] pads;
            if (module.padding is not null)
            {
                if (module.padding.Length == 2)
                {
                    pads = new[] { (int)module.padding[0], (int)module.padding[1], (int)module.padding[0], (int)module.padding[1] };
                }
                else if (module.padding.Length == 4)
                {
                    pads = new[] { (int)module.padding[0], (int)module.padding[1], (int)module.padding[2], (int)module.padding[3] };
                }
                else
                {
                    pads = new[] { 0, 0, 0, 0 };
                }
            }
            else
            {
                pads = new[] { 0, 0, 0, 0 };
            }
            node.Attributes["pads"] = pads;

            int groupValue = module.groups > 0 ? (int)module.groups : 1;
            node.Attributes["group"] = groupValue;
            Console.WriteLine($"[DEBUG Conv2d] Node={node.Name} group={groupValue}");

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
            Console.WriteLine($"[DEBUG Conv2d] Node={node.Name} dilation[0]={dilations[0]} dilation[1]={dilations[1]}");

            var stridesAttr = node.Attributes["strides"];
            string stridesStr = stridesAttr is int[] s ? $"[{s[0]},{s[1]}]" : "NOT_SET";
            Console.WriteLine($"[DEBUG Conv2d] Node={node.Name} Final strides attr: {stridesStr}");

            context.Graph?.AddInitializer(weightName, module.weight);
            if (module.bias is not null)
            {
                context.Graph?.AddInitializer(biasName, module.bias);
            }

            if (context.TryGetShape(inputName, out var inputShape) && inputShape.Count >= 4)
            {
                var n = inputShape[0];
                var c = module.out_channels;

                var kernelH = node.Attributes.TryGetValue("kernel_shape", out var ksh) && ksh is int[] ks ? ks[0] : 3;
                var kernelW = node.Attributes.TryGetValue("kernel_shape", out var ksw) && ksw is int[] kss ? kss[1] : 3;
                var strideH = node.Attributes.TryGetValue("strides", out var sth) && sth is int[] s1 ? s1[0] : 1;
                var strideW = node.Attributes.TryGetValue("strides", out var stw) && stw is int[] s2 ? s2[1] : 1;
                var padH = node.Attributes.TryGetValue("pads", out var ph) && ph is int[] p1 ? p1[0] : 0;
                var padW = node.Attributes.TryGetValue("pads", out var pw) && pw is int[] p2 ? p2[1] : 0;
                var dilH = node.Attributes.TryGetValue("dilations", out var dh) && dh is int[] d1 ? d1[0] : 1;
                var dilW = node.Attributes.TryGetValue("dilations", out var dw) && dw is int[] d2 ? d2[1] : 1;

                var inH = inputShape[2];
                var inW = inputShape[3];

                var outH = (long)Math.Floor((double)(inH + 2 * padH - dilH * (kernelH - 1) - 1) / strideH + 1);
                var outW = (long)Math.Floor((double)(inW + 2 * padW - dilW * (kernelW - 1) - 1) / strideW + 1);

                context.SetCurrentValue(outputName);
                context.SetShape(outputName, new List<long> { n, c, outH, outW });
            }
            else
            {
                context.SetCurrentValue(outputName);
                context.SetShape(outputName, new List<long> { 1, module.out_channels, -1, -1 });
            }

            context.Graph?.AddNode(node);
            return node;
        }
    }
}