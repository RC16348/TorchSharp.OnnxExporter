using System;
using System.Collections.Generic;
using System.Linq;
using TorchSharp.OnnxExporter.DataFlow;
using Module = TorchSharp.torch.nn.Module;

namespace TorchSharp.OnnxExporter.Processors
{
    /// <summary>
    /// Add操作处理器 - 解决残差连接问题
    ///
    /// 【问题描述】
    /// 残差连接（如ResidualBlock中的 x += residual）是深度学习模型中的常见模式。
    /// 在SymbolicTraceEngine跟踪时，当前输入是上一个操作的输出，而不是原始输入。
    /// 当执行 x += residual 时，Add操作需要知道residual的值，但此时currentInput已经是x的值。
    ///
    /// 【解决方案】
    /// 通过TraceContext.GetOriginalInput()获取原始输入，确保残差连接正确。
    ///
    /// 【典型应用场景】
    /// - ResidualBlock: x = conv1(x); x = conv2(x); x += residual
    /// - Bottleneck: x = conv1(x); x = conv2(x); x += input (shortcut)
    /// - SqueezeExcitation: x = x * sigmoid(SE(x))
    /// </summary>
    public class AddProcessor : INodeProcessor
    {
        public string OpType => "Add";

        public bool CanProcess(Module module)
        {
            var typeName = module.GetType().Name;
            return typeName == "Add" || typeName == "add"
                || typeName == "Modules.Add" || typeName.EndsWith("+Add");
        }

        public DataFlowNode Process(Module module, TraceContext context)
        {
            var inputs = new List<string>();
            var currentInput = context.GetCurrentValue();
            inputs.Add(currentInput);

            // 检查模块是否有子模块（如torch.add是函数而非模块）
            var hasChildModules = module.named_children().Any();
            if (hasChildModules)
            {
                // 如果Add模块有子模块，尝试从子模块获取第二个输入
                foreach (var child in module.named_children())
                {
                    var childOutput = context.GetValue(child.name);
                    if (!string.IsNullOrEmpty(childOutput))
                    {
                        inputs.Add(childOutput);
                    }
                }
            }

            // 【关键】如果只有一个输入，尝试获取原始输入作为残差
            // 这是解决残差连接问题的核心逻辑
            if (inputs.Count < 2)
            {
                var originalInput = context.GetOriginalInput();
                if (!string.IsNullOrEmpty(originalInput) && originalInput != currentInput)
                {
                    // 使用原始输入作为第二个参数（残差）
                    inputs.Add(originalInput);
                }
                else
                {
                    // 如果没有原始输入，复制当前输入（等价于 x + x）
                    inputs.Add(currentInput);
                }
            }

            // 【异常处理】确保有足够的输入
            if (inputs.Count < 2)
            {
                throw new InvalidOperationException(
                    $"[ONNX导出错误] Add操作缺少足够的输入: 当前输入数量={inputs.Count}。" +
                    $"这通常表示模型中的Add操作结构异常。");
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