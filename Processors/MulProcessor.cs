using System;
using System.Collections.Generic;
using System.Linq;
using TorchSharp.OnnxExporter.DataFlow;
using Module = TorchSharp.torch.nn.Module;

namespace TorchSharp.OnnxExporter.Processors
{
    /// <summary>
    /// Mul操作处理器 - 解决通道门控问题
    ///
    /// 【问题描述】
    /// 通道门控操作（如SqueezeExcitation中的 x = x * sigmoid(SE(x))）使用Mul操作。
    /// 与残差连接的Add类似，Mul操作也需要知道两个输入的值。
    ///
    /// 【解决方案】
    /// 与AddProcessor相同的模式：通过TraceContext.GetOriginalInput()获取原始输入。
    ///
    /// 【典型应用场景】
    /// - SqueezeExcitation: x = x * sigmoid(SE(x))
    /// - HardSwish: x = x * ReLU6(x+3)/6
    /// - GoogLeNet Inception: 多尺度特征融合
    /// </summary>
    public class MulProcessor : INodeProcessor
    {
        public string OpType => "Mul";

        public bool CanProcess(Module module)
        {
            return module.GetType().Name == "Mul" || module.GetType().Name == "mul";
        }

        public DataFlowNode Process(Module module, TraceContext context)
        {
            var inputs = new List<string>();
            var currentInput = context.GetCurrentValue();
            inputs.Add(currentInput);

            // 检查模块是否有子模块
            var hasChildModules = module.named_children().Any();
            if (hasChildModules)
            {
                foreach (var child in module.named_children())
                {
                    var childOutput = context.GetValue(child.name);
                    if (!string.IsNullOrEmpty(childOutput))
                    {
                        inputs.Add(childOutput);
                    }
                }
            }

            // 【关键】如果只有一个输入，尝试获取原始输入
            // 这解决了通道门控（SqueezeExcitation）中的Mul问题
            if (inputs.Count < 2)
            {
                var originalInput = context.GetOriginalInput();
                if (!string.IsNullOrEmpty(originalInput) && originalInput != currentInput)
                {
                    inputs.Add(originalInput);
                }
                else
                {
                    inputs.Add(currentInput);
                }
            }

            // 【异常处理】确保有足够的输入
            if (inputs.Count < 2)
            {
                throw new InvalidOperationException(
                    $"[ONNX导出错误] Mul操作缺少足够的输入: 当前输入数量={inputs.Count}。" +
                    $"这通常表示模型中的Mul操作结构异常。");
            }

            var outputName = context.CreateTempName();
            var node = new DataFlowNode(OpType, inputs, new[] { outputName });

            context.Graph?.AddNode(node);

            // 更新当前值
            context.SetCurrentValue(outputName);

            return node;
        }
    }
}