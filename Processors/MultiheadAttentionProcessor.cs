using System;
using System.Collections.Generic;
using System.Linq;
using TorchSharp.Modules;
using TorchSharp.OnnxExporter.DataFlow;

namespace TorchSharp.OnnxExporter.Processors
{
    public class MultiheadAttentionProcessor : BaseProcessor<MultiheadAttention>
    {
        public override string OpType => "MultiHeadAttention";

        public override DataFlowNode Process(MultiheadAttention module, TraceContext context)
        {
            // 【修复】使用实际的输入名称，而不是创建新的临时名称
            // 对于 self-attention: query = key = value = input
            // 对于 cross-attention，需要跟踪三个不同的输入
            var inputName = context.GetCurrentValue();
            var outputName = context.CreateTempName();

            var weightName = $"{outputName}_weight";
            var biasName = $"{outputName}_bias";

            var parameters = module.named_parameters().ToDictionary(p => p.name, p => p.parameter);
            parameters.TryGetValue("in_proj_weight", out var weight);
            parameters.TryGetValue("in_proj_bias", out var bias);

            long numHeads = 1;
            if (weight is not null)
            {
                long embedDim = weight.shape[0];
                numHeads = Math.Max(1, embedDim / 64);
            }

            // 【修复】ONNX MultiHeadAttention 的输入格式：
            // (query, key, value, weight, bias) - 或可选的 mask
            // 对于 self-attention，query = key = value = inputName
            var node = new DataFlowNode(OpType,
                new[] { inputName, inputName, inputName, weightName, biasName },
                new[] { outputName });

            node.Attributes["num_heads"] = numHeads;

            if (weight is not null) context.Graph?.AddInitializer(weightName, weight);
            if (bias is not null) context.Graph?.AddInitializer(biasName, bias);

            context.SetCurrentValue(outputName);
            context.Graph?.AddNode(node);

            return node;
        }
    }
}