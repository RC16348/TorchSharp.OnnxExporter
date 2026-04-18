using System;
using Module = TorchSharp.torch.nn.Module;
using Tensor = TorchSharp.torch.Tensor;

namespace TorchSharp.OnnxExporter.Modules
{
    public class Concat : Operator
    {
        public int dim = 0;

        public Concat(int dimension = 0) : base("Concat")
        {
            dim = dimension;
        }

        public override Tensor forward(params Tensor[] inputs)
        {
            if (inputs == null || inputs.Length == 0)
            {
                throw new ArgumentException("Concat operator requires at least 1 input");
            }

            return torch.cat(inputs, dim);
        }
    }

    public class Stack : Operator
    {
        public int dim = 0;

        public Stack(int dimension = 0) : base("Stack")
        {
            dim = dimension;
        }

        public override Tensor forward(params Tensor[] inputs)
        {
            if (inputs == null || inputs.Length == 0)
            {
                throw new ArgumentException("Stack operator requires at least 1 input");
            }

            return torch.stack(inputs, dim);
        }
    }

    public class ReshapeOp : Operator
    {
        public long[] shape;

        public ReshapeOp(params long[] shape) : base("ReshapeOp")
        {
            this.shape = shape;
        }

        public override Tensor forward(params Tensor[] inputs)
        {
            if (inputs == null || inputs.Length == 0)
            {
                throw new ArgumentException("Reshape operator requires at least 1 input");
            }

            return torch.reshape(inputs[0], shape);
        }
    }

    public class TransposeOp : Operator
    {
        public int dim0 = 0;
        public int dim1 = 1;

        public TransposeOp(int dim0 = 0, int dim1 = 1) : base("TransposeOp")
        {
            this.dim0 = dim0;
            this.dim1 = dim1;
        }

        public override Tensor forward(params Tensor[] inputs)
        {
            if (inputs == null || inputs.Length == 0)
            {
                throw new ArgumentException("Transpose operator requires at least 1 input");
            }

            return torch.transpose(inputs[0], dim0, dim1);
        }
    }

    public class SqueezeOp : Operator
    {
        public int dim = -1;

        public SqueezeOp(int dimension = -1) : base("SqueezeOp")
        {
            dim = dimension;
        }

        public override Tensor forward(params Tensor[] inputs)
        {
            if (inputs == null || inputs.Length == 0)
            {
                throw new ArgumentException("Squeeze operator requires at least 1 input");
            }

            if (dim >= 0)
            {
                return torch.squeeze(inputs[0], dim);
            }
            else
            {
                return torch.squeeze(inputs[0]);
            }
        }
    }

    public class UnsqueezeOp : Operator
    {
        public int dim = 0;

        public UnsqueezeOp(int dimension = 0) : base("UnsqueezeOp")
        {
            dim = dimension;
        }

        public override Tensor forward(params Tensor[] inputs)
        {
            if (inputs == null || inputs.Length == 0)
            {
                throw new ArgumentException("Unsqueeze operator requires at least 1 input");
            }

            return torch.unsqueeze(inputs[0], dim);
        }
    }

    public class ClampOp : Operator
    {
        public double? min;
        public double? max;

        public ClampOp(double? min = null, double? max = null) : base("ClampOp")
        {
            this.min = min;
            this.max = max;
        }

        public override Tensor forward(params Tensor[] inputs)
        {
            if (inputs == null || inputs.Length == 0)
            {
                throw new ArgumentException("Clamp operator requires at least 1 input");
            }

            var tensor = inputs[0];
            if (min.HasValue && max.HasValue)
            {
                return torch.clamp(tensor, (TorchSharp.Scalar)min.Value, (TorchSharp.Scalar)max.Value);
            }
            else if (min.HasValue)
            {
                return torch.clamp(tensor, (TorchSharp.Scalar)min.Value);
            }
            else if (max.HasValue)
            {
                return torch.clamp(tensor, (TorchSharp.Scalar)max.Value);
            }
            else
            {
                return tensor;
            }
        }
    }

    public class WhereOp : Operator
    {
        public WhereOp() : base("WhereOp") { }

        public override Tensor forward(params Tensor[] inputs)
        {
            if (inputs == null || inputs.Length != 3)
            {
                throw new ArgumentException("Where operator requires exactly 3 inputs (condition, x, y)");
            }

            return torch.where(inputs[0], inputs[1], inputs[2]);
        }
    }

    public class SumOp : Operator
    {
        public int? dim;
        public bool keepdim = false;

        public SumOp(int? dimension = null, bool keepDim = false) : base("SumOp")
        {
            dim = dimension;
            keepdim = keepDim;
        }

        public override Tensor forward(params Tensor[] inputs)
        {
            if (inputs == null || inputs.Length == 0)
            {
                throw new ArgumentException("Sum operator requires at least 1 input");
            }

            if (dim.HasValue)
            {
                return torch.sum(inputs[0], dim.Value, keepdim);
            }
            else
            {
                return torch.sum(inputs[0]);
            }
        }
    }

    public class MeanOp : Operator
    {
        public int? dim;
        public bool keepdim = false;

        public MeanOp(int? dimension = null, bool keepDim = false) : base("MeanOp")
        {
            dim = dimension;
            keepdim = keepDim;
        }

        public override Tensor forward(params Tensor[] inputs)
        {
            if (inputs == null || inputs.Length == 0)
            {
                throw new ArgumentException("Mean operator requires at least 1 input");
            }

            if (dim.HasValue)
            {
                return torch.mean(inputs[0], new long[] { (long)dim.Value }, keepdim);
            }
            else
            {
                return torch.mean(inputs[0]);
            }
        }
    }
}