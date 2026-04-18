using System;
using System.Collections.Generic;
using TorchSharp.Modules;
using TorchSharp.OnnxExporter.DataFlow;

namespace TorchSharp.OnnxExporter.Processors
{
    public class AdaptiveAvgPool2dProcessor : BaseProcessor<AdaptiveAvgPool2d>
    {
        public override string OpType => "AveragePool";

        public override DataFlowNode Process(AdaptiveAvgPool2d module, TraceContext context)
        {
            var inputName = context.GetCurrentValue();
            var outputName = context.CreateTempName();

            var node = new DataFlowNode(OpType, new[] { inputName }, new[] { outputName });

            if (context.TryGetShape(inputName, out var inputShape) && inputShape.Count >= 4)
            {
                var n = inputShape[0];
                var c = inputShape[1];
                var h = inputShape[2];
                var w = inputShape[3];

                long[] outputSize = module.output_size;
                int outputH, outputW;
                if (outputSize != null && outputSize.Length >= 2)
                {
                    outputH = (int)outputSize[0];
                    outputW = (int)outputSize[1];
                }
                else
                {
                    outputH = 1;
                    outputW = 1;
                }

                int kernelH = (int)(h / outputH);
                int kernelW = (int)(w / outputW);

                context.SetShape(outputName, new List<long> { n, c, outputH, outputW });

                node.Attributes["kernel_shape"] = new[] { kernelH, kernelW };
                node.Attributes["strides"] = new[] { kernelH, kernelW };
                node.Attributes["pads"] = new[] { 0, 0, 0, 0 };
                node.Attributes["auto_pad"] = "NOTSET";
            }

            context.SetCurrentValue(outputName);
            context.Graph?.AddNode(node);

            return node;
        }
    }
}