using System;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

class Program
    {
        static void Main(string[] args)
        {
            var modelPath = "random_forest.onnx";
            var inputs = new Tensor<float>[][] { ... }; // Liste des entrées différentes

            var tasks = inputs.Select(input => Task.Run(() => RunInference(modelPath, input))).ToArray();
            Task.WaitAll(tasks);

            foreach (var task in tasks)
            {
                var output = task.Result;
                // Traiter les résultats ici
            }
        }

        static InferenceOutput RunInference(string modelPath, Tensor<float>[] inputs)
        {
            using var session = new InferenceSession(modelPath);
            var inputNames = session.InputMetadata.Keys.ToList();
            var inputsDict = new List<NamedOnnxValue>();

            for (int i = 0; i < inputNames.Count; i++)
            {
                inputsDict.Add(NamedOnnxValue.CreateFromTensor(inputNames[i], inputs[i]));
            }

            using var results = session.Run(inputsDict);
            // Assumons qu'il y a un seul résultat
            return results.First().AsEnumerable<float>().ToArray();
        }
    }

    public class InferenceOutput
    {
        public float[] Results { get; set; }
    }

