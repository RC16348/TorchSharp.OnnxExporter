using System;
using Module = TorchSharp.torch.nn.Module;
using Tensor = TorchSharp.torch.Tensor;

namespace TorchSharp.OnnxExporter.Modules
{
    public interface IOperator
    {
        Tensor forward(params Tensor[] inputs);
    }

    public abstract class Operator : Module
    {
        protected Operator(string name) : base(name) { }

        public abstract Tensor forward(params Tensor[] inputs);
    }

    public class Add : Operator
    {
        public Add() : base("Add") { }

        public override Tensor forward(params Tensor[] inputs)
        {
            if (inputs.Length < 2)
            {
                throw new System.ArgumentException("Add operator requires at least 2 inputs");
            }

            Tensor result = inputs[0];
            for (int i = 1; i < inputs.Length; i++)
            {
                result = TorchSharp.torch.add(result, inputs[i]);
            }

            return result;
        }
    }

    public class Sub : Operator
    {
        public Sub() : base("Sub") { }

        public override Tensor forward(params Tensor[] inputs)
        {
            if (inputs.Length < 2)
            {
                throw new System.ArgumentException("Sub operator requires at least 2 inputs");
            }

            Tensor result = inputs[0];
            for (int i = 1; i < inputs.Length; i++)
            {
                result = TorchSharp.torch.sub(result, inputs[i]);
            }

            return result;
        }
    }

    public class Mul : Operator
    {
        public Mul() : base("Mul") { }

        public override Tensor forward(params Tensor[] inputs)
        {
            if (inputs.Length < 2)
            {
                throw new System.ArgumentException("Mul operator requires at least 2 inputs");
            }

            Tensor result = inputs[0];
            for (int i = 1; i < inputs.Length; i++)
            {
                result = TorchSharp.torch.mul(result, inputs[i]);
            }

            return result;
        }
    }

    public class Div : Operator
    {
        public Div() : base("Div") { }

        public override Tensor forward(params Tensor[] inputs)
        {
            if (inputs.Length < 2)
            {
                throw new System.ArgumentException("Div operator requires at least 2 inputs");
            }

            Tensor result = inputs[0];
            for (int i = 1; i < inputs.Length; i++)
            {
                result = TorchSharp.torch.div(result, inputs[i]);
            }

            return result;
        }
    }

    public class MatMul : Operator
    {
        public MatMul() : base("MatMul") { }

        public override Tensor forward(params Tensor[] inputs)
        {
            if (inputs.Length < 2)
            {
                throw new System.ArgumentException("MatMul operator requires at least 2 inputs");
            }

            Tensor result = inputs[0];
            for (int i = 1; i < inputs.Length; i++)
            {
                result = TorchSharp.torch.matmul(result, inputs[i]);
            }

            return result;
        }
    }

    public class Pow : Operator
    {
        public Pow() : base("Pow") { }

        public override Tensor forward(params Tensor[] inputs)
        {
            if (inputs.Length < 2)
            {
                throw new System.ArgumentException("Pow operator requires at least 2 inputs (base and exponent)");
            }

            return TorchSharp.torch.pow(inputs[0], inputs[1]);
        }
    }

    public class Sqrt : Operator
    {
        public Sqrt() : base("Sqrt") { }

        public override Tensor forward(params Tensor[] inputs)
        {
            if (inputs.Length < 1)
            {
                throw new System.ArgumentException("Sqrt operator requires at least 1 input");
            }

            return TorchSharp.torch.sqrt(inputs[0]);
        }
    }

    public class Exp : Operator
    {
        public Exp() : base("Exp") { }

        public override Tensor forward(params Tensor[] inputs)
        {
            if (inputs.Length < 1)
            {
                throw new System.ArgumentException("Exp operator requires at least 1 input");
            }

            return TorchSharp.torch.exp(inputs[0]);
        }
    }

    public class Log : Operator
    {
        public Log() : base("Log") { }

        public override Tensor forward(params Tensor[] inputs)
        {
            if (inputs.Length < 1)
            {
                throw new System.ArgumentException("Log operator requires at least 1 input");
            }

            return TorchSharp.torch.log(inputs[0]);
        }
    }

    public class AddWithBias : Operator
    {
        public readonly Tensor bias;

        public AddWithBias(Tensor bias) : base("AddWithBias")
        {
            this.bias = bias;
        }

        public override Tensor forward(params Tensor[] inputs)
        {
            if (inputs.Length < 1)
            {
                throw new System.ArgumentException("AddWithBias requires at least 1 input");
            }

            return TorchSharp.torch.add(inputs[0], bias);
        }
    }

    public class LinearOperator : Operator
    {
        public readonly Tensor weight;
        public readonly Tensor? bias;

        public LinearOperator(Tensor weight, Tensor? bias = null) : base("LinearOperator")
        {
            this.weight = weight;
            this.bias = bias;
        }

        public override Tensor forward(params Tensor[] inputs)
        {
            if (inputs.Length < 1)
            {
                throw new System.ArgumentException("LinearOperator requires at least 1 input");
            }

            var output = TorchSharp.torch.matmul(inputs[0], weight);
            if (!object.ReferenceEquals(bias, null))
            {
                output = TorchSharp.torch.add(output, bias);
            }
            return output;
        }
    }
}