using System;
using System.IO;
using System.Linq;


class codefile
{
    static double[] weights;
    static double bias;
    static double lr = 0.01;

    static void Main(string[] args)

    {
        Console.WriteLine("=== ML Training Pipeline ===");

        var rng = new Random(42);
        int n = 200;
        int split = 160;
        double[][] X = new double[n][];
        double[] Y = new double[n];

        for (int i = 0; i < n; i++)
        {
            X[i] = new double[] { rng.NextDouble() * 10, rng.NextDouble() * 5 };
            Y[i] = 3.0 * X[i][0] + 1.5 * X[i][1] + rng.NextDouble() * 0.5;
        }

        var trainX = X.Take(split).ToArray();
        var trainY = Y.Take(split).ToArray();
        var testX = X.Skip(split).ToArray();
        var testY = Y.Skip(split).ToArray();

        Console.WriteLine($"Train: {trainX.Length}, Test: {testX.Length}");

        int features = trainX[0].Length;
        weights = new double[features];
        bias = 0;

        for (int epoch = 0; epoch < 100; epoch++)
        {
            double[] gw = new double[features];
            double gb = 0;
            for (int i = 0; i < trainX.Length; i++)
            {
                double pred = Predict(trainX[i]);
                double err = pred - trainY[i];
                for (int j = 0; j < features; j++)
                    gw[j] += err * trainX[i][j];
                gb += err;
            }
            for (int j = 0; j < features; j++)
                weights[j] -= lr * gw[j] / trainX.Length;
            bias -= lr * gb / trainX.Length;
        }

        double trainMse = MSE(trainX, trainY);
        double testMse = MSE(testX, testY);
        Console.WriteLine($"Train MSE: {trainMse:F4}");
        Console.WriteLine($"Test MSE: {testMse:F4}");
        Console.WriteLine($"Validation: {(testMse < 1.0 ? "PASS" : "FAIL")}");

        using (var sw = new StreamWriter("model_output.txt"))
        {
            sw.WriteLine("weights=" + string.Join(",", weights.Select(w => w.ToString("F6"))));
            sw.WriteLine("bias=" + bias.ToString("F6"));
        }

        using (var sw = new StreamWriter("metrics.txt"))
        {
            sw.WriteLine($"train_mse={trainMse:F6}");
            sw.WriteLine($"test_mse={testMse:F6}");
            sw.WriteLine($"timestamp={DateTime.UtcNow:o}");
        }

        Console.WriteLine("Artifacts saved.");
    }

    static double Predict(double[] x)
    {
        double s = bias;
        for (int i = 0; i < x.Length; i++) s += weights[i] * x[i];
        return s;
    }

    static double MSE(double[][] X, double[] Y)
    {
        double sum = 0;
        for (int i = 0; i < X.Length; i++)
        {
            double d = Predict(X[i]) - Y[i];
            sum += d * d;
        }
        return sum / X.Length;
    }
}