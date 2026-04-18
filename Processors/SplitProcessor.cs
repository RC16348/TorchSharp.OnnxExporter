using System;
using System.Collections.Generic;
using System.Linq;
using TorchSharp;
using TorchSharp.OnnxExporter.DataFlow;
using Module = TorchSharp.torch.nn.Module;

namespace TorchSharp.OnnxExporter.Processors
{
    public class SplitProcessor : INodeProcessor
    {
        public string OpType => "Split";

        public bool CanProcess(Module module)
        {
            return module.GetType().Name == "Split";
        }

        public DataFlowNode Process(Module module, TraceContext context)
        {
            var inputName = context.GetCurrentValue();
            var outputName = context.CreateTempName();

            var splitModule = (dynamic)module;
            var splitSize = (int)splitModule.split_size;
            int dim = (int)splitModule.dim;

            var splitSizeName = $"{outputName}_splitSize";
            var dimName = $"{outputName}_dim";

            var inputs = new[] { inputName, splitSizeName, dimName };
            var outputs = new[] { $"{outputName}_0", $"{outputName}_1" };

            var node = new DataFlowNode(OpType, inputs, outputs);

            context.Graph?.AddInitializer(splitSizeName, new long[] { splitSize });
            context.Graph?.AddInitializer(dimName, new long[] { dim });

            context.Graph?.AddNode(node);
            return node;
        }
    }
}