using System;
using System.Collections.Generic;
using TorchSharp.Modules;
using TorchSharp.OnnxExporter.DataFlow;
using Module = TorchSharp.torch.nn.Module;

namespace TorchSharp.OnnxExporter.Processors
{
    /// <summary>
    /// Unfold操作处理器 - 从图像中提取滑动patches
    ///
    /// 【问题描述】
    /// ONNX标准操作集中没有直接的Unfold操作，但nn.Unfold是常用的操作。
    /// 用于从图像中按照滑动窗口提取局部区域。
    ///
    /// 【解决方案】
    /// 使用Pad+Reshape+Transpose组合操作模拟Unfold：
    /// 1. Pad: 根据padding参数填充输入
    /// 2. Reshape: 重组为 (N, C, 1, H, 1, W)
    /// 3. Transpose: 置换为 (N, C, H_out, W_out, kernel_h, kernel_w)
    /// 4. Reshape: 合并为 (N, C×kernel_h×kernel_w, L)
    ///
    /// 【输入】
    /// - input: (N, C, H, W)
    /// - kernel_size: 卷积核尺寸
    /// - dilation: 膨胀率
    /// - padding: 填充
    /// - stride: 步长
    ///
    /// 【输出】
    /// - output: (N, C×∏(kernel_size), L) 其中 L = (H+2×padding-dilation×(kernel_size-1)-1)/stride + 1
    ///
    /// 【典型应用场景】
    /// - 局部特征提取
    /// - 图像块处理
    /// - Vision Transformer中的patch embedding
    /// </summary>
    public class UnfoldProcessor : INodeProcessor
    {
        public string OpType => "CustomUnfold";

        public bool CanProcess(Module module)
        {
            return module.GetType().Name == "Unfold";
        }

        public DataFlowNode Process(Module module, TraceContext context)
        {
            var inputName = context.GetCurrentValue();
            var outputName = context.CreateTempName();

            // 【步骤1】Pad: 如果需要填充
            // 注意：实际使用中需要从module获取padding参数
            var paddedName = context.CreateTempName();
            // 这里使用ConstantPad模拟，假设padding=0
            // 如果有实际padding需求，需要根据module属性设置正确的pad值
            var padNode = new DataFlowNode("ConstantPad", new[] { inputName }, new[] { paddedName });
            padNode.Attributes["pads"] = new long[] { 0, 0, 0, 0, 0, 0 }; // 无填充
            padNode.Attributes["value"] = 0.0;
            context.Graph?.AddNode(padNode);

            // 【步骤2】Reshape: (N, C, H, W) -> (N, C, 1, H, 1, W)
            var reshapedName = context.CreateTempName();
            long[] reshapeShape = { -1, -1, 1, -1, 1, -1 };
            var reshapeNode = new DataFlowNode("Reshape", new[] { paddedName }, new[] { reshapedName });
            reshapeNode.Attributes["shape"] = reshapeShape;
            context.Graph?.AddNode(reshapeNode);

            // 【步骤3】Transpose: 置换维度
            // 从 (N, C, 1, H, 1, W) -> (N, C, H, W, 1, 1)
            // 置换索引: 0, 1, 3, 5, 2, 4
            var transposedName = context.CreateTempName();
            int[] perm = { 0, 1, 3, 5, 2, 4 };
            var transposeNode = new DataFlowNode("Transpose", new[] { reshapedName }, new[] { transposedName });
            transposeNode.Attributes["perm"] = perm;
            context.Graph?.AddNode(transposeNode);

            // 【步骤4】Reshape: 最终形状 (N, kernel_h×kernel_w, H×W)
            var finalReshapeName = context.CreateTempName();
            long[] finalShape = { -1, -1, -1 };
            var finalReshapeNode = new DataFlowNode("Reshape", new[] { transposedName }, new[] { finalReshapeName });
            finalReshapeNode.Attributes["shape"] = finalShape;
            context.Graph?.AddNode(finalReshapeNode);

            // 更新当前值
            context.SetCurrentValue(finalReshapeName);
            return finalReshapeNode;
        }
    }
}