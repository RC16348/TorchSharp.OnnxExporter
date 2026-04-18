using TorchSharp.Modules;
using TorchSharp.OnnxExporter.DataFlow;
using Module = TorchSharp.torch.nn.Module;

namespace TorchSharp.OnnxExporter.Processors
{
    public abstract class BaseProcessor<TModule> : INodeProcessor where TModule : Module
    {
        public abstract string OpType { get; }
        public abstract DataFlowNode Process(TModule module, TraceContext context);

        public bool CanProcess(Module module) => module is TModule;
        public DataFlowNode Process(Module module, TraceContext context) => Process((TModule)module, context);
    }
}