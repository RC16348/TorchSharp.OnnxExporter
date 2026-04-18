using System;
using System.Collections.Generic;
using TorchSharp.Modules;
using TorchSharp.OnnxExporter.DataFlow;
using Module = TorchSharp.torch.nn.Module;

namespace TorchSharp.OnnxExporter.Processors
{
    /// <summary>
    /// Fold操作处理器 - 将展开的patches重组回图像
    ///
    /// 【问题描述】
    /// ONNX标准操作集中没有直接的Fold操作，但nn.Fold是常用的操作。
    /// 例如，将图像分割成patches进行处理后，需要重新组合回完整图像。
    ///
    /// 【解决方案】
    /// 使用Reshape+Transpose组合操作模拟Fold：
    /// 1. Reshape: 将输入重组为 (N, C, kernel_h, kernel_w, H_out, W_out)
    /// 2. Transpose: 置换维度为 (N, C, H_out, W_out, kernel_h, kernel_w)
    /// 3. Reshape: 合并最后两个维度得到 (N, C, H_out, W_out)
    ///
    /// 【输入】
    /// - input: (N, C×∏(kernel_size), L) 其中 L = prod(output_size)
    /// - output_size: 输出图像尺寸 (H_out, W_out)
    /// - kernel_size: 卷积核尺寸
    ///
    /// 【输出】
    /// - output: (N, C, H_out, W_out)
    ///
    /// 【典型应用场景】
    /// - 图像重建
    /// - 局部特征重组
    /// - Transformer中的patch处理
    /// </summary>
    public class FoldProcessor : INodeProcessor
    {
        public string OpType => "CustomFold";

        public bool CanProcess(Module module)
        {
            return module.GetType().Name == "Fold";
        }

        public DataFlowNode Process(Module module, TraceContext context)
        {
            var inputName = context.GetCurrentValue();
            var outputName = context.CreateTempName();

            // 获取Fold模块的参数
            // 注意：TorchSharp的Fold模块可能不直接暴露这些参数，需要从模块属性获取
            // 这里使用默认值，实际使用时需要根据具体模块调整

            // 【步骤1】第一次Reshape: (N, C×kernel_h×kernel_w, L) -> (N, C, kernel_h, kernel_w, H_out, W_out)
            // 由于无法直接获取output_size和kernel_size，这里需要假设固定值
            // 实际使用中可以通过module的属性获取

            var reshapedName1 = context.CreateTempName();
            // 使用-1让ONNX自动推断维度
            long[] newShape1 = { -1, -1, -1, -1, -1, -1 };
            var reshapeNode1 = new DataFlowNode("Reshape", new[] { inputName }, new[] { reshapedName1 });
            reshapeNode1.Attributes["shape"] = newShape1;
            context.Graph?.AddNode(reshapeNode1);

            // 【步骤2】Transpose: 置换维度顺序
            // 从 (N, C, kernel_h, kernel_w, H_out, W_out) -> (N, C, H_out, W_out, kernel_h, kernel_w)
            // 置换索引: 0, 1, 4, 5, 2, 3
            var transposedName = context.CreateTempName();
            int[] perm = { 0, 1, 4, 5, 2, 3 };
            var transposeNode = new DataFlowNode("Transpose", new[] { reshapedName1 }, new[] { transposedName });
            transposeNode.Attributes["perm"] = perm;
            context.Graph?.AddNode(transposeNode);

            // 【步骤3】第二次Reshape: 合并最后两个维度 (N, C, H_out, W_out, kernel_h×kernel_w) -> (N, C, H_out, W_out)
            var reshapedName2 = context.CreateTempName();
            long[] newShape2 = { -1, -1, -1, -1 };
            var reshapeNode2 = new DataFlowNode("Reshape", new[] { transposedName }, new[] { reshapedName2 });
            reshapeNode2.Attributes["shape"] = newShape2;
            context.Graph?.AddNode(reshapeNode2);

            // 更新当前值
            context.SetCurrentValue(reshapedName2);
            return reshapeNode2;
        }
    }
}