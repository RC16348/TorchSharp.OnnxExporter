using System;
using System.Collections.Generic;
using Onnx;
using Google.Protobuf;

namespace TorchSharp.OnnxExporter.DataFlow
{
    public class DataFlowNode
    {
        public string Name { get; set; }
        public string OpType { get; private set; }
        public IReadOnlyList<string> Inputs { get; private set; }
        public IReadOnlyList<string> Outputs { get; private set; }
        public Dictionary<string, object> Attributes { get; private set; }

        public DataFlowNode(string opType, IEnumerable<string> inputs, IEnumerable<string> outputs)
        {
            OpType = opType ?? throw new ArgumentNullException(nameof(opType));
            Inputs = inputs != null ? new List<string>(inputs).AsReadOnly() : new List<string>().AsReadOnly();
            Outputs = outputs != null ? new List<string>(outputs).AsReadOnly() : new List<string>().AsReadOnly();
            Attributes = new Dictionary<string, object>();
            Name = $"{opType}_{Guid.NewGuid()}";
        }

        public NodeProto ToNodeProto()
        {
            var node = new NodeProto
            {
                OpType = OpType,
                Name = Name
            };

            foreach (var input in Inputs)
            {
                node.Input.Add(input);
            }

            foreach (var output in Outputs)
            {
                node.Output.Add(output);
            }

            foreach (var attr in Attributes)
            {
                var attrProto = ConvertAttribute(attr.Key, attr.Value);
                if (attrProto != null)
                {
                    node.Attribute.Add(attrProto);
                    Console.WriteLine($"[DEBUG ToNodeProto] Node={Name} OpType={OpType} attr={attr.Key} type={attr.Value.GetType().Name} count={GetAttrCount(attr.Value)}");
                }
            }

            Console.WriteLine($"[DEBUG ToNodeProto] Node={Name} OpType={OpType} total_attrs={Attributes.Count}");
            return node;
        }

        private static int GetAttrCount(object value)
        {
            if (value is IEnumerable<int> intList) return intList.Count();
            if (value is IEnumerable<long> longList) return longList.Count();
            if (value is IEnumerable<float> floatList) return floatList.Count();
            if (value is IEnumerable<double> doubleList) return doubleList.Count();
            if (value is IEnumerable<string> stringList) return stringList.Count();
            return 1;
        }

        private static AttributeProto ConvertAttribute(string name, object value)
        {
            var attrProto = new AttributeProto { Name = name };

            Console.WriteLine($"[DEBUG ConvertAttribute] Converting '{name}' type={value.GetType().Name} value={value}");

            switch (value)
            {
                case int intValue:
                    attrProto.Type = AttributeProto.Types.AttributeType.Int;
                    attrProto.I = intValue;
                    break;
                case long longValue:
                    attrProto.Type = AttributeProto.Types.AttributeType.Int;
                    attrProto.I = longValue;
                    break;
                case float floatValue:
                    attrProto.Type = AttributeProto.Types.AttributeType.Float;
                    attrProto.F = floatValue;
                    break;
                case double doubleValue:
                    attrProto.Type = AttributeProto.Types.AttributeType.Float;
                    attrProto.F = (float)doubleValue;
                    break;
                case string stringValue:
                    attrProto.Type = AttributeProto.Types.AttributeType.String;
                    attrProto.S = ByteString.CopyFromUtf8(stringValue);
                    break;
                case bool boolValue:
                    attrProto.Type = AttributeProto.Types.AttributeType.Int;
                    attrProto.I = boolValue ? 1 : 0;
                    break;
                case IEnumerable<int> intList:
                    attrProto.Type = AttributeProto.Types.AttributeType.Ints;
                    foreach (var item in intList)
                    {
                        attrProto.Ints.Add(item);
                    }
                    break;
                case IEnumerable<long> longList:
                    attrProto.Type = AttributeProto.Types.AttributeType.Ints;
                    foreach (var item in longList)
                    {
                        attrProto.Ints.Add(item);
                    }
                    break;
                case IEnumerable<float> floatList:
                    attrProto.Type = AttributeProto.Types.AttributeType.Floats;
                    foreach (var item in floatList)
                    {
                        attrProto.Floats.Add(item);
                    }
                    break;
                case IEnumerable<double> doubleList:
                    attrProto.Type = AttributeProto.Types.AttributeType.Floats;
                    foreach (var item in doubleList)
                    {
                        attrProto.Floats.Add((float)item);
                    }
                    break;
                case IEnumerable<string> stringList:
                    attrProto.Type = AttributeProto.Types.AttributeType.Strings;
                    foreach (var item in stringList)
                    {
                        attrProto.Strings.Add(ByteString.CopyFromUtf8(item));
                    }
                    break;
                case TensorProto tensorProto:
                    attrProto.Type = AttributeProto.Types.AttributeType.Tensor;
                    attrProto.T = tensorProto;
                    break;
                case GraphProto graphProto:
                    attrProto.Type = AttributeProto.Types.AttributeType.Graph;
                    attrProto.G = graphProto;
                    break;
                default:
                    Console.WriteLine($"Warning: Unsupported attribute type for '{name}': {value.GetType().Name}");
                    return null;
            }

            return attrProto;
        }
    }
}