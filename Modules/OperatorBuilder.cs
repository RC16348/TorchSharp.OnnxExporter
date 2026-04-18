using System;
using Tensor = TorchSharp.torch.Tensor;
using Module = TorchSharp.torch.nn.Module;

namespace TorchSharp.OnnxExporter.Modules
{
    public static class OperatorBuilder
    {
        public static AddWithBiasBuilder CreateAddWithBias()
        {
            return new AddWithBiasBuilder();
        }

        public static LinearOperatorBuilder CreateLinearOperator()
        {
            return new LinearOperatorBuilder();
        }

        public class AddWithBiasBuilder
        {
            private Tensor? _bias;

            public AddWithBiasBuilder Bias(Tensor bias)
            {
                _bias = bias;
                return this;
            }

            public AddWithBias Build()
            {
                if (object.ReferenceEquals(_bias, null))
                {
                    throw new InvalidOperationException("Bias is required for AddWithBias");
                }
                return new AddWithBias(_bias);
            }
        }

        public class LinearOperatorBuilder
        {
            private Tensor? _weight;
            private Tensor? _bias;

            public LinearOperatorBuilder Weight(Tensor weight)
            {
                _weight = weight;
                return this;
            }

            public LinearOperatorBuilder Bias(Tensor bias)
            {
                _bias = bias;
                return this;
            }

            public LinearOperator Build()
            {
                if (object.ReferenceEquals(_weight, null))
                {
                    throw new InvalidOperationException("Weight is required for LinearOperator");
                }
                return new LinearOperator(_weight, _bias);
            }
        }
    }
}