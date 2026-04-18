using System;
using System.IO;
using System.Threading.Tasks;
using Google.Protobuf;
using Onnx;
using TorchSharp;
using TorchSharp.OnnxExporter.Builder;
using TorchSharp.OnnxExporter.Tracing;
using Tensor = TorchSharp.torch.Tensor;
using Module = TorchSharp.torch.nn.Module;

namespace TorchSharp.OnnxExporter
{
    /// <summary>
    /// ONNX导出器 - 统一导出API
    ///
    /// 【功能说明】
    /// 提供简洁的API将TorchSharp模型导出为ONNX格式。
    /// 使用符号跟踪引擎（SymbolicTraceEngine）构建数据流图，
    /// 然后通过ONNX图构建器（OnnxGraphBuilder）生成ONNX protobuf。
    ///
    /// 【使用示例】
    /// var model = nn.Linear(10, 10);
    /// var input = torch.randn(1, 10);
    /// OnnxExporter.Export(model, input, "model.onnx", "MyModel");
    ///
    /// 【异常处理】
    /// 所有异常都带有 [ONNX导出错误] 前缀，方便开发者快速定位问题。
    /// </summary>
    public static class OnnxExporter
    {
        /// <summary>
        /// 同步导出模型到ONNX格式
        /// </summary>
        /// <param name="model">要导出的TorchSharp模型</param>
        /// <param name="dummyInput">虚拟输入，用于形状推断和跟踪</param>
        /// <param name="outputPath">输出ONNX文件路径</param>
        /// <param name="modelName">模型名称（可选，默认"model"）</param>
        /// <exception cref="ArgumentNullException">
        /// - model为null: 模型不能为空
        /// - dummyInput为null: 虚拟输入不能为空
        /// - outputPath为null或空: 输出路径不能为空
        /// </exception>
        /// <exception cref="InvalidOperationException">
        /// - 符号跟踪失败
        /// - 图构建失败
        /// - 文件写入失败
        /// </exception>
        public static void Export(Module model, Tensor dummyInput, string outputPath, string modelName = "model")
        {
            // 【异常处理】参数验证
            if (model == null)
                throw new ArgumentNullException(nameof(model), "[ONNX导出错误] 模型不能为null。请确保传入有效的TorchSharp模型。");

            if (dummyInput is null)
                throw new ArgumentNullException(nameof(dummyInput), "[ONNX导出错误] 虚拟输入不能为null。请使用torch.randn()创建有效输入。");

            if (string.IsNullOrEmpty(outputPath))
                throw new ArgumentNullException(nameof(outputPath), "[ONNX导出错误] 输出路径不能为空。请指定有效的ONNX文件路径。");

            if (string.IsNullOrEmpty(modelName))
                modelName = "model";

            try
            {
                // 1. 注册所有默认处理器
                // 这确保了所有支持的模块类型都能被正确处理
                RegisterDefaultProcessors();

                // 2. 创建符号跟踪引擎并跟踪模型
                // 【关键步骤】
                // - 构建数据流图
                // - 记录每个操作的输入输出
                // - 解决残差连接等问题
                var traceEngine = new SymbolicTraceEngine();
                var dataFlowGraph = traceEngine.Trace(model, dummyInput);

                // 3. 创建ONNX图构建器并生成ONNX ModelProto
                var graphBuilder = new OnnxGraphBuilder();
                var modelProto = graphBuilder.Build(dataFlowGraph, modelName);

                // 4. 写入文件
                // 使用CodedOutputStream进行高效的protobuf编码
                using var fileStream = File.Create(outputPath);
                using var codedStream = new CodedOutputStream(fileStream);
                modelProto.WriteTo(codedStream);
            }
            catch (ArgumentNullException)
            {
                throw;
            }
            catch (InvalidOperationException)
            {
                throw;
            }
            catch (IOException ex)
            {
                throw new InvalidOperationException(
                    $"[ONNX导出错误] 文件写入失败: 路径={outputPath}, 错误={ex.Message}", ex);
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException(
                    $"[ONNX导出错误] 导出过程发生未知错误: 模型={model.GetType().Name}, " +
                    $"输入形状={string.Join(",", dummyInput.shape)}, 错误={ex.Message}", ex);
            }
        }

        /// <summary>
        /// 异步导出模型到ONNX格式
        /// </summary>
        /// <param name="model">要导出的TorchSharp模型</param>
        /// <param name="dummyInput">虚拟输入，用于形状推断和跟踪</param>
        /// <param name="outputPath">输出ONNX文件路径</param>
        /// <param name="modelName">模型名称（可选，默认"model"）</param>
        /// <exception cref="ArgumentNullException">参见Export方法</exception>
        /// <exception cref="InvalidOperationException">参见Export方法</exception>
        public static async Task ExportAsync(Module model, Tensor dummyInput, string outputPath, string modelName = "model")
        {
            // 【异常处理】参数验证
            if (model == null)
                throw new ArgumentNullException(nameof(model), "[ONNX导出错误] 模型不能为null。请确保传入有效的TorchSharp模型。");

            if (dummyInput is null)
                throw new ArgumentNullException(nameof(dummyInput), "[ONNX导出错误] 虚拟输入不能为null。请使用torch.randn()创建有效输入。");

            if (string.IsNullOrEmpty(outputPath))
                throw new ArgumentNullException(nameof(outputPath), "[ONNX导出错误] 输出路径不能为空。请指定有效的ONNX文件路径。");

            if (string.IsNullOrEmpty(modelName))
                modelName = "model";

            try
            {
                // 1. 注册所有默认处理器
                RegisterDefaultProcessors();

                // 2. 创建符号跟踪引擎并跟踪模型
                var traceEngine = new SymbolicTraceEngine();
                var dataFlowGraph = traceEngine.Trace(model, dummyInput);

                // 3. 创建ONNX图构建器并生成ONNX ModelProto
                var graphBuilder = new OnnxGraphBuilder();
                var modelProto = graphBuilder.Build(dataFlowGraph, modelName);

                // 4. 异步写入文件
                using var fileStream = File.Create(outputPath);
                using var codedStream = new CodedOutputStream(fileStream);
                await Task.Run(() => modelProto.WriteTo(codedStream));
            }
            catch (ArgumentNullException)
            {
                throw;
            }
            catch (InvalidOperationException)
            {
                throw;
            }
            catch (IOException ex)
            {
                throw new InvalidOperationException(
                    $"[ONNX导出错误] 文件写入失败: 路径={outputPath}, 错误={ex.Message}", ex);
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException(
                    $"[ONNX导出错误] 异步导出过程发生未知错误: 模型={model.GetType().Name}, " +
                    $"输入形状={string.Join(",", dummyInput.shape)}, 错误={ex.Message}", ex);
            }
        }

        /// <summary>
        /// 注册默认的模块处理器
        /// 包括：卷积、池化、激活函数、归一化、RNN、注意力等100+处理器
        /// </summary>
        /// <exception cref="InvalidOperationException">当处理器注册失败时抛出</exception>
        private static void RegisterDefaultProcessors()
        {
            try
            {
                ModuleProcessorRegistry.RegisterDefaultProcessors();
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException(
                    $"[ONNX导出错误] 处理器注册失败: 错误={ex.Message}", ex);
            }
        }
    }
}