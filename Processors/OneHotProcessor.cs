using System;
using System.Collections.Generic;
using TorchSharp.OnnxExporter.DataFlow;
using Module = TorchSharp.torch.nn.Module;

namespace TorchSharp.OnnxExporter.Processors
{
    public class OneHotProcessor : INodeProcessor
    {
        public string OpType => "OneHot";

        public bool CanProcess(Module module)
        {
            var name = module.GetType().Name;
            return name == "OneHot";
        }

        public DataFlowNode Process(Module module, TraceContext context)
        {
            var inputName = context.GetCurrentValue();
            var outputName = context.CreateTempName();

            var oneHotModule = (dynamic)module;
            int numClasses = (int)oneHotModule.num_classes;

            var depthName = $"{outputName}_depth";
            var valuesName = $"{outputName}_values";

            var node = new DataFlowNode(OpType, new[] { inputName, depthName, valuesName }, new[] { outputName });

            context.Graph?.AddInitializer(depthName, new long[] { numClasses });
            context.Graph?.AddInitializer(valuesName, new long[] { 0, 1 });

            context.Graph?.AddNode(node);
            return node;
        }
    }
}