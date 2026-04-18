using TorchSharp.Modules;
using TorchSharp.OnnxExporter.DataFlow;
using Module = TorchSharp.torch.nn.Module;

namespace TorchSharp.OnnxExporter.Processors
{
    public interface INodeProcessor
    {
        string OpType { get; }
        bool CanProcess(Module module);
        DataFlowNode Process(Module module, TraceContext context);
    }
}