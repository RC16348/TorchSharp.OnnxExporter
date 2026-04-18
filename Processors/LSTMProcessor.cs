using System;
using System.Collections.Generic;
using System.Linq;
using TorchSharp.Modules;
using TorchSharp.OnnxExporter.DataFlow;

namespace TorchSharp.OnnxExporter.Processors
{
    public class LSTMProcessor : BaseProcessor<LSTM>
    {
        public override string OpType => "LSTM";

        public override DataFlowNode Process(LSTM module, TraceContext context)
        {
            var inputName = context.GetCurrentValue();
            var outputName = context.CreateTempName();

            // 【修复】不要让隐藏状态同时作为输入和输出
            // 使用独立的初始和最终隐藏状态名称
            var initialHiddenName = context.CreateTempName();
            var initialCellName = context.CreateTempName();
            var finalHiddenName = context.CreateTempName();
            var finalCellName = context.CreateTempName();

            var weightIhName = $"{outputName}_weight_ih";
            var weightHhName = $"{outputName}_weight_hh";
            var biasIhName = $"{outputName}_bias_ih";
            var biasHhName = $"{outputName}_bias_hh";

            var parameters = module.named_parameters().ToDictionary(p => p.name, p => p.parameter);
            parameters.TryGetValue("weight_ih_l0", out var weightIh);
            parameters.TryGetValue("weight_hh_l0", out var weightHh);
            parameters.TryGetValue("bias_ih_l0", out var biasIh);
            parameters.TryGetValue("bias_hh_l0", out var biasHh);

            long hiddenSize = weightHh is not null ? weightHh.shape[^1] : 0L;
            bool bidirectional = parameters.ContainsKey("weight_ih_l0_reverse");

            // 【修复】输入和输出使用不同的名称，避免循环依赖
            // ONNX LSTM 的输入：(input, weight_ih, weight_hh, bias_ih, bias_hh, initial_H, initial_C)
            // ONNX LSTM 的输出：(output, final_H, final_C)
            var node = new DataFlowNode(OpType,
                new[] { inputName, weightIhName, weightHhName, biasIhName, biasHhName, initialHiddenName, initialCellName },
                new[] { outputName, finalHiddenName, finalCellName });

            node.Attributes["direction"] = bidirectional ? "bidirectional" : "forward";
            node.Attributes["hidden_size"] = hiddenSize;
            node.Attributes["layout"] = 0;
            node.Attributes["num_layers"] = 1;

            if (weightIh is not null) context.Graph?.AddInitializer(weightIhName, weightIh);
            if (weightHh is not null) context.Graph?.AddInitializer(weightHhName, weightHh);
            if (biasIh is not null) context.Graph?.AddInitializer(biasIhName, biasIh);
            if (biasHh is not null) context.Graph?.AddInitializer(biasHhName, biasHh);

            // 【修复】添加零初始化的隐藏状态作为常量，而不是作为图的输入
            // 这样可以打破循环依赖
            int numLayers = 1;
            int numDirections = bidirectional ? 2 : 1;
            var zeroHidden = torch.zeros(numLayers * numDirections, 1, hiddenSize);
            var zeroCell = torch.zeros(numLayers * numDirections, 1, hiddenSize);

            context.Graph?.AddInitializer(initialHiddenName, zeroHidden);
            context.Graph?.AddInitializer(initialCellName, zeroCell);

            context.SetCurrentValue(outputName);
            context.Graph?.AddNode(node);
            return node;
        }
    }
}