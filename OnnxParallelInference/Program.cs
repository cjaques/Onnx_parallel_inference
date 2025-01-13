using System;
using System.Drawing;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Reflection.Metadata;
using System.Diagnostics;

class Program
{
    const string modelPath = "random_forest.onnx";
    const string imagesPath = @"..\..\..\..\data\";
    static void Main(string[] args)
    {
        var timer = Stopwatch.StartNew();
        var inputs = LoadAllBitmapImages(imagesPath);
        RunInferenceOnImages(inputs);
        timer.Stop();
        var elapsedTime = timer.ElapsedMilliseconds;
    }

    private static void RunInferenceOnImages(List<Tensor<float>> inputs)
    {
        var tasks = new List<Task<int>>();

        foreach (var image in inputs)
        {
            var input = new List<NamedOnnxValue>() {
                         NamedOnnxValue.CreateFromTensor<float>("float_input", image),
            };
            Task<int> task = Task<int>.Run(() => RunInference(modelPath, input));
            tasks.Add(task);
        }

        Task.WaitAll(tasks.ToArray());

        foreach (var task in tasks)
        {
            var output = task.Result;
        }
    }

    static int RunInference(string modelPath, List<NamedOnnxValue> inputs)
    {
        using var session = new InferenceSession(modelPath);
        using var results = session.Run(inputs);
        var scores = results.First().AsTensor<long>();
        return (int)scores[0];
    }

    static List<Tensor<float>> LoadAllBitmapImages(string path)
    {
        var images = new List<Tensor<float>>();
        if (Directory.Exists(path))
        {
            foreach(var imName  in Directory.GetFiles(path, "*.png"))
            {
                Bitmap img = new Bitmap(imName);
                var tensor = ConvertImageToFloatStack(img);
                images.Add(tensor);
            }
        }

        return images;
    }

    public static Tensor<float> ConvertImageToFloatStack(Bitmap image)
    {
        // Create the Tensor with the appropiate dimensions  for the NN
        Tensor<float> data = new DenseTensor<float>(new[] { 1, image.Width * image.Height });

        // Iterate over the bitmap width and height and copy each pixel
        for (int x = 0; x < image.Width; x++)
        {
            for (int y = 0; y < image.Height; y++)
            {
                Color color = image.GetPixel(x, y);
                data[0, x * image.Height + y] = color.R / 255.0f;
            }
        }

        return data;
    }
}

