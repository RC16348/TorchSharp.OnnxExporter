using System;
using System.Collections.Generic;
using TorchSharp;
using TorchSharp.OnnxExporter.DataFlow;
using Module = TorchSharp.torch.nn.Module;

namespace TorchSharp.OnnxExporter.Processors
{
    public class TransposeProcessor : INodeProcessor
    {
        public string OpType => "Transpose";

        public bool CanProcess(Module module)
        {
            var typeName = module.GetType().Name;
            return typeName == "Transpose" || typeName == "Permute" || typeName == "TransposeOp";
        }

        public DataFlowNode Process(Module module, TraceContext context)
        {
            var inputName = context.GetNextValue();

            var transposeModule = (dynamic)module;
            var typeName = module.GetType().Name;

            int[] dims;
            if (typeName == "TransposeOp")
            {
                var dim0 = (int)transposeModule.dim0;
                var dim1 = (int)transposeModule.dim1;
                dims = new[] { 0, 2, 1 };
                if (dim0 == 1 && dim1 == 2)
                {
                    dims = new[] { 0, 2, 1 };
                }
                else if (dim0 == 0 && dim1 == 1)
                {
                    dims = new[] { 1, 0 };
                }
                else
                {
                    dims = new[] { 0, 2, 1 };
                }
            }
            else
            {
                dims = (int[])transposeModule.dims;
            }

            var outputName = context.CreateTempName();
            var inputs = new[] { inputName };
            var outputs = new[] { outputName };

            var node = new DataFlowNode(OpType, inputs, outputs);
            node.Attributes["perm"] = dims;

            context.Graph?.AddNode(node);
            context.SetCurrentValue(outputName);
            return node;
        }
    }
}