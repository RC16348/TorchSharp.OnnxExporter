using System;
using System.Collections.Generic;
using TorchSharp.OnnxExporter.DataFlow;
using Module = TorchSharp.torch.nn.Module;

namespace TorchSharp.OnnxExporter.Processors
{
    public class PairwiseDistanceProcessor : INodeProcessor
    {
        public string OpType => "PairwiseDistance";

        public bool CanProcess(Module module)
        {
            var name = module.GetType().Name;
            return name == "PairwiseDistance";
        }

        public DataFlowNode Process(Module module, TraceContext context)
        {
            var input1Name = context.GetCurrentValue();
            var input2Name = context.GetNextValue();
            var outputName = context.CreateTempName();

            var pairwiseModule = (dynamic)module;
            float p = (float)pairwiseModule.p;
            float eps = (float)pairwiseModule.eps;
            bool keepdim = (bool)pairwiseModule.keepdim;

            var subNode = new DataFlowNode("Sub", new[] { input1Name, input2Name }, new[] { context.CreateTempName() });
            context.Graph?.AddNode(subNode);

            var absNode = new DataFlowNode("Abs", new[] { subNode.Outputs[0] }, new[] { context.CreateTempName() });
            context.Graph?.AddNode(absNode);

            var reduceNode = new DataFlowNode("ReduceL1", new[] { absNode.Outputs[0] }, new[] { outputName });
            reduceNode.Attributes["keepdims"] = keepdim ? 1 : 0;

            context.Graph?.AddNode(reduceNode);
            return reduceNode;
        }
    }
}