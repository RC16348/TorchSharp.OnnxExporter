using System;
using System.Collections.Generic;
using TorchSharp;
using TorchSharp.OnnxExporter.DataFlow;
using TorchSharp.OnnxExporter.Modules;
using TorchSharp.OnnxExporter.Processors;
using TorchSharp.Modules;

namespace TorchSharp.OnnxExporter
{
    public sealed class ModuleProcessorRegistry
    {
        private static readonly ModuleProcessorRegistry _instance = new ModuleProcessorRegistry();
        private readonly Dictionary<Type, INodeProcessor> _processors = new Dictionary<Type, INodeProcessor>();
        private readonly Dictionary<string, INodeProcessor> _nameProcessors = new Dictionary<string, INodeProcessor>();
        private bool _initialized = false;

        private ModuleProcessorRegistry()
        {
        }

        public static ModuleProcessorRegistry Instance => _instance;

        public static void Register<TModule>(INodeProcessor processor) where TModule : TorchSharp.torch.nn.Module
        {
            Instance.RegisterInternal(typeof(TModule), processor);
        }

        public static void RegisterByName(string moduleName, INodeProcessor processor)
        {
            Instance.EnsureInitialized();
            Instance._nameProcessors[moduleName] = processor;
        }

        private void RegisterInternal(Type moduleType, INodeProcessor processor)
        {
            _processors[moduleType] = processor;
        }

        public INodeProcessor? GetProcessor(TorchSharp.torch.nn.Module module)
        {
            if (module == null)
                return null;

            EnsureInitialized();

            var moduleType = module.GetType();
            Console.WriteLine($"[DEBUG GetProcessor] Looking for: {moduleType.Name}");

            var processor = GetProcessor(moduleType);
            if (processor != null)
                return processor;

            var moduleName = module.GetType().Name;
            if (_nameProcessors.TryGetValue(moduleName, out var nameProcessor))
                return nameProcessor;

            Console.WriteLine($"[DEBUG GetProcessor] NOT FOUND for: {moduleType.Name}");
            return null;
        }

        public INodeProcessor? GetProcessor(Type moduleType)
        {
            if (moduleType == null)
                return null;

            EnsureInitialized();

            if (_processors.TryGetValue(moduleType, out var processor))
                return processor;

            foreach (var kvp in _processors)
            {
                if (kvp.Key.IsAssignableFrom(moduleType))
                    return kvp.Value;
            }

            return null;
        }

        public bool CanProcess(TorchSharp.torch.nn.Module module)
        {
            return GetProcessor(module) != null;
        }

        public static INodeProcessor? GetProcessorStatic(TorchSharp.torch.nn.Module module)
        {
            return Instance.GetProcessor(module);
        }

        public static bool CanProcessStatic(TorchSharp.torch.nn.Module module)
        {
            return Instance.CanProcess(module);
        }

        private void EnsureInitialized()
        {
            if (_initialized)
                return;

            RegisterDefaultProcessors();
            _initialized = true;
        }

        public static void RegisterDefaultProcessors()
        {
            var registry = Instance;

            registry.RegisterInternal(typeof(Conv2d), new Conv2dProcessor());
            registry.RegisterInternal(typeof(ConvTranspose2d), new ConvTranspose2dProcessor());
            registry.RegisterInternal(typeof(Linear), new LinearProcessor());
            registry.RegisterInternal(typeof(ReLU), new ReLUProcessor());
            registry.RegisterInternal(typeof(ReLU6), new ReLU6Processor());
            registry.RegisterInternal(typeof(LeakyReLU), new LeakyReLUProcessor());
            registry.RegisterInternal(typeof(PReLU), new PReLUProcessor());
            registry.RegisterInternal(typeof(RReLU), new RReLUProcessor());
            registry.RegisterInternal(typeof(ELU), new ELUProcessor());
            registry.RegisterInternal(typeof(CELU), new CELUProcessor());
            registry.RegisterInternal(typeof(SELU), new SELUProcessor());
            registry.RegisterInternal(typeof(Mish), new MishProcessor());
            registry.RegisterInternal(typeof(Softplus), new SoftplusProcessor());
            registry.RegisterInternal(typeof(Softshrink), new SoftshrinkProcessor());
            registry.RegisterInternal(typeof(Softsign), new SoftsignProcessor());
            registry.RegisterInternal(typeof(LogSigmoid), new LogSigmoidProcessor());
            registry.RegisterInternal(typeof(Softmax2d), new Softmax2dProcessor());
            registry.RegisterInternal(typeof(Softmin), new SoftminProcessor());
            registry.RegisterInternal(typeof(LogSoftmax), new LogSoftmaxProcessor());
            registry.RegisterInternal(typeof(Hardsigmoid), new HardsigmoidProcessor());
            registry.RegisterInternal(typeof(Hardswish), new HardswishProcessor());
            registry.RegisterInternal(typeof(Hardtanh), new HardtanhProcessor());
            registry.RegisterInternal(typeof(Hardshrink), new HardshrinkProcessor());
            registry.RegisterInternal(typeof(Tanhshrink), new TanhshrinkProcessor());
            registry.RegisterInternal(typeof(Threshold), new ThresholdProcessor());
            registry.RegisterInternal(typeof(SiLU), new SiLUProcessor());
            registry.RegisterInternal(typeof(GELU), new GELUProcessor());
            registry.RegisterInternal(typeof(Sigmoid), new SigmoidProcessor());
            registry.RegisterInternal(typeof(Tanh), new TanhProcessor());
            registry.RegisterInternal(typeof(Softmax), new SoftmaxProcessor());
            registry.RegisterInternal(typeof(BatchNorm2d), new BatchNormProcessor());
            registry.RegisterInternal(typeof(LayerNorm), new LayerNormProcessor());
            registry.RegisterInternal(typeof(GroupNorm), new GroupNormProcessor());
            registry.RegisterInternal(typeof(InstanceNorm2d), new InstanceNormProcessor());
            registry.RegisterInternal(typeof(Dropout), new DropoutProcessor());
            registry.RegisterInternal(typeof(Dropout1d), new Dropout1dProcessor());
            registry.RegisterInternal(typeof(Dropout2d), new Dropout2dProcessor());
            registry.RegisterInternal(typeof(Dropout3d), new Dropout3dProcessor());
            registry.RegisterInternal(typeof(AlphaDropout), new AlphaDropoutProcessor());
            registry.RegisterInternal(typeof(Embedding), new EmbeddingProcessor());
            registry.RegisterInternal(typeof(EmbeddingBag), new EmbeddingBagProcessor());
            registry.RegisterInternal(typeof(PixelShuffle), new PixelShuffleProcessor());
            registry.RegisterInternal(typeof(PixelUnshuffle), new PixelUnshuffleProcessor());
            registry.RegisterInternal(typeof(MultiheadAttention), new MultiheadAttentionProcessor());
            registry.RegisterInternal(typeof(RNN), new RNNProcessor());
            registry.RegisterInternal(typeof(RNNCell), new RNNCellProcessor());
            registry.RegisterInternal(typeof(LSTM), new LSTMProcessor());
            registry.RegisterInternal(typeof(LSTMCell), new LSTMCellProcessor());
            registry.RegisterInternal(typeof(GRU), new GRUProcessor());
            registry.RegisterInternal(typeof(GRUCell), new GRUCellProcessor());
            registry.RegisterInternal(typeof(MaxPool2d), new MaxPool2dProcessor());
            registry.RegisterInternal(typeof(AvgPool2d), new AvgPool2dProcessor());
            registry.RegisterInternal(typeof(AdaptiveAvgPool2d), new AdaptiveAvgPool2dProcessor());
            registry.RegisterInternal(typeof(Flatten), new FlattenProcessor());
            registry.RegisterInternal(typeof(Upsample), new UpsampleProcessor());
            registry.RegisterInternal(typeof(Fold), new FoldProcessor());
            registry.RegisterInternal(typeof(Unfold), new UnfoldProcessor());

            registry.RegisterInternal(typeof(Conv1d), new Conv1dProcessor());
            registry.RegisterInternal(typeof(Conv3d), new Conv3dProcessor());
            registry.RegisterInternal(typeof(ConvTranspose1d), new ConvTranspose1dProcessor());
            registry.RegisterInternal(typeof(ConvTranspose3d), new ConvTranspose3dProcessor());
            registry.RegisterInternal(typeof(ChannelShuffle), new ChannelShuffleProcessor());

            registry.RegisterInternal(typeof(BatchNorm1d), new BatchNorm1dProcessor());
            registry.RegisterInternal(typeof(BatchNorm3d), new BatchNorm3dProcessor());
            registry.RegisterInternal(typeof(InstanceNorm1d), new InstanceNorm1dProcessor());
            registry.RegisterInternal(typeof(InstanceNorm3d), new InstanceNorm3dProcessor());
            registry.RegisterInternal(typeof(LocalResponseNorm), new LocalResponseNormProcessor());

            registry.RegisterInternal(typeof(MaxPool1d), new MaxPool1dProcessor());
            registry.RegisterInternal(typeof(MaxPool3d), new MaxPool3dProcessor());
            registry.RegisterInternal(typeof(MaxUnpool1d), new MaxUnpool1dProcessor());
            registry.RegisterInternal(typeof(MaxUnpool2d), new MaxUnpool2dProcessor());
            registry.RegisterInternal(typeof(MaxUnpool3d), new MaxUnpool3dProcessor());
            registry.RegisterInternal(typeof(AvgPool1d), new AvgPool1dProcessor());
            registry.RegisterInternal(typeof(AvgPool3d), new AvgPool3dProcessor());
            registry.RegisterInternal(typeof(LPPool1d), new LPPool1dProcessor());
            registry.RegisterInternal(typeof(LPPool2d), new LPPool2dProcessor());
            registry.RegisterInternal(typeof(AdaptiveAvgPool1d), new AdaptiveAvgPool1dProcessor());
            registry.RegisterInternal(typeof(AdaptiveAvgPool3d), new AdaptiveAvgPool3dProcessor());
            registry.RegisterInternal(typeof(AdaptiveMaxPool1d), new AdaptiveMaxPool1dProcessor());
            registry.RegisterInternal(typeof(AdaptiveMaxPool2d), new AdaptiveMaxPool2dProcessor());
            registry.RegisterInternal(typeof(AdaptiveMaxPool3d), new AdaptiveMaxPool3dProcessor());
            registry.RegisterInternal(typeof(FractionalMaxPool2d), new FractionalMaxPool2dProcessor());
            registry.RegisterInternal(typeof(FractionalMaxPool3d), new FractionalMaxPool3dProcessor());

            registry.RegisterInternal(typeof(ConstantPad1d), new ConstantPad1dProcessor());
            registry.RegisterInternal(typeof(ConstantPad2d), new ConstantPad2dProcessor());
            registry.RegisterInternal(typeof(ConstantPad3d), new ConstantPad3dProcessor());
            registry.RegisterInternal(typeof(ReflectionPad1d), new ReflectionPad1dProcessor());
            registry.RegisterInternal(typeof(ReflectionPad2d), new ReflectionPad2dProcessor());
            registry.RegisterInternal(typeof(ReflectionPad3d), new ReflectionPad3dProcessor());
            registry.RegisterInternal(typeof(ReplicationPad1d), new ReplicationPad1dProcessor());
            registry.RegisterInternal(typeof(ReplicationPad2d), new ReplicationPad2dProcessor());
            registry.RegisterInternal(typeof(ReplicationPad3d), new ReplicationPad3dProcessor());
            registry.RegisterInternal(typeof(ZeroPad2d), new ZeroPad2dProcessor());
            registry.RegisterInternal(typeof(Bilinear), new BilinearProcessor());
            registry.RegisterInternal(typeof(CosineSimilarity), new CosineSimilarityProcessor());

            registry._nameProcessors["Concat"] = new ConcatProcessor();
            registry._nameProcessors["Concatenate"] = new ConcatProcessor();
            registry._nameProcessors["Chunk"] = new ChunkProcessor();
            registry._nameProcessors["Split"] = new SplitProcessor();
            registry._nameProcessors["Reshape"] = new ReshapeProcessor();
            registry._nameProcessors["Transpose"] = new TransposeProcessor();
            registry._nameProcessors["Permute"] = new TransposeProcessor();
            registry._nameProcessors["Squeeze"] = new SqueezeProcessor();
            registry._nameProcessors["Unsqueeze"] = new UnsqueezeProcessor();
            registry._nameProcessors["Add"] = new AddProcessor();
            registry._nameProcessors["add"] = new AddProcessor();
            registry._nameProcessors["Mul"] = new MulProcessor();
            registry._nameProcessors["mul"] = new MulProcessor();
            registry._nameProcessors["Sub"] = new SubProcessor();
            registry._nameProcessors["Div"] = new DivProcessor();
            registry._nameProcessors["MatMul"] = new MatMulProcessor();
            registry._nameProcessors["Pow"] = new PowProcessor();
            registry._nameProcessors["Identity"] = new IdentityProcessor();
            registry._nameProcessors["OneHot"] = new OneHotProcessor();
            registry._nameProcessors["PairwiseDistance"] = new PairwiseDistanceProcessor();

            // 新增可追踪算子模块注册 (Modules 命名空间)
            registry.RegisterInternal(typeof(Modules.Add), new AddProcessor());
            registry.RegisterInternal(typeof(Modules.Sub), new SubProcessor());
            registry.RegisterInternal(typeof(Modules.Mul), new MulProcessor());
            registry.RegisterInternal(typeof(Modules.Div), new DivProcessor());
            registry.RegisterInternal(typeof(Modules.MatMul), new MatMulProcessor());
            registry.RegisterInternal(typeof(Modules.Pow), new PowProcessor());
            registry.RegisterInternal(typeof(Modules.Sqrt), new ElementWiseProcessor("Sqrt"));
            registry.RegisterInternal(typeof(Modules.Exp), new ElementWiseProcessor("Exp"));
            registry.RegisterInternal(typeof(Modules.Log), new ElementWiseProcessor("Log"));
            registry.RegisterInternal(typeof(Modules.Concat), new ConcatProcessor());
            registry.RegisterInternal(typeof(Modules.Stack), new ConcatProcessor());
            registry.RegisterInternal(typeof(Modules.ReshapeOp), new ReshapeProcessor());
            registry.RegisterInternal(typeof(Modules.TransposeOp), new TransposeProcessor());
            registry.RegisterInternal(typeof(Modules.SqueezeOp), new SqueezeProcessor());
            registry.RegisterInternal(typeof(Modules.UnsqueezeOp), new UnsqueezeProcessor());
            registry.RegisterInternal(typeof(Modules.ClampOp), new ElementWiseProcessor("Clamp"));
            registry.RegisterInternal(typeof(Modules.WhereOp), new ElementWiseProcessor("Where"));
            registry.RegisterInternal(typeof(Modules.SumOp), new ElementWiseProcessor("ReduceSum"));
            registry.RegisterInternal(typeof(Modules.MeanOp), new ElementWiseProcessor("ReduceMean"));
            registry.RegisterInternal(typeof(Modules.AddWithBias), new AddWithBiasProcessor());
            registry.RegisterInternal(typeof(Modules.LinearOperator), new LinearOperatorProcessor());
        }
    }
}
