using System;
using System.Collections.Generic;
using TorchSharp.Modules;
using TorchSharp.OnnxExporter.DataFlow;

namespace TorchSharp.OnnxExporter.Processors
{
    public class UpsampleProcessor : BaseProcessor<Upsample>
    {
        public override string OpType => "Resize";

        public override DataFlowNode Process(Upsample module, TraceContext context)
        {
            var inputName = context.GetCurrentValue();
            var outputName = context.CreateTempName();

            var roiName = $"{outputName}_roi";
            var scalesName = $"{outputName}_scales";

            var inputs = new List<string> { inputName, roiName, scalesName };
            var outputs = new List<string> { outputName };

            var node = new DataFlowNode(OpType, inputs, outputs);
            node.Attributes["mode"] = "nearest";
            node.Attributes["coordinate_transformation_mode"] = "asymmetric";
            node.Attributes["nearest_mode"] = "floor";

            var roiTensor = new List<float>();
            context.Graph?.AddInitializer(roiName, roiTensor.ToArray(), new long[] { 0 });

            var scales = new List<float> { 1.0f, 1.0f, 2.0f, 2.0f };
            context.Graph?.AddInitializer(scalesName, scales.ToArray(), new long[] { 4 });

            context.Graph?.AddNode(node);
            return node;
        }
    }
}