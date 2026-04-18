using System;
using System.Collections.Generic;
using TorchSharp;
using static TorchSharp.torch;

namespace TorchSharp.OnnxExporter.DataFlow
{
    public class TraceContext
    {
        public DataFlowGraph? Graph { get; set; }
        public Dictionary<string, string> Values { get; } = new();
        public Dictionary<string, string> NamedValues { get; } = new();
        public Dictionary<string, List<long>> ValueShapes { get; } = new();
        private int _tempCounter;
        private string? _currentValue;
        private string? _originalInput;

        public TraceContext()
        {
        }

        public TraceContext(DataFlowGraph graph)
        {
            Graph = graph;
        }

        public string AddValue(string name, Tensor value)
        {
            var onnxName = CreateTempName();
            Values[name] = onnxName;
            if (!(value is null) && !(value.shape is null))
            {
                ValueShapes[onnxName] = value.shape.ToList();
            }
            return onnxName;
        }

        public string GetValue(string name)
        {
            return Values.TryGetValue(name, out var onnxName) ? onnxName : name;
        }

        public void SetNamedValue(string key, string value)
        {
            NamedValues[key] = value;
        }

        public string GetNamedValue(string key)
        {
            return NamedValues.TryGetValue(key, out var value) ? value : key;
        }

        public string CreateTempName()
        {
            return $"tmp_{_tempCounter++}";
        }

        public string GetCurrentValue()
        {
            return _currentValue ?? $"tmp_{_tempCounter - 1}";
        }

        public void SetCurrentValue(string value)
        {
            _currentValue = value;
        }

        public string GetNextValue()
        {
            return $"tmp_{_tempCounter}";
        }

        public void SetOriginalInput(string value)
        {
            _originalInput = value;
        }

        public string? GetOriginalInput()
        {
            return _originalInput;
        }

        public void PushScope(string scopeName)
        {
            SetNamedValue("current_scope", scopeName);
        }

        public void PopScope()
        {
            SetNamedValue("current_scope", "");
        }

        public void SetShape(string name, List<long> shape)
        {
            ValueShapes[name] = shape;
        }

        public bool TryGetShape(string name, out List<long> shape)
        {
            return ValueShapes.TryGetValue(name, out shape);
        }
    }
}