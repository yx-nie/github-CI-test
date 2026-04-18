public class LinearModel {
    public double[] Weights { get; set; }
    public double Bias { get; set; }
}

public static class Trainer
{
    public static (double[][] X, double[] Y) GenerateData(int n = 200, int seed = 42)
    {
        var rng = new Random(seed);
        double[][] X = new double[n][];
        double[] Y = new double[n];
        for (int i = 0; i < n; i++)
        {
            X[i] = new[] { rng.NextDouble() * 10, rng.NextDouble() * 5 };
            Y[i] = 3.0 * X[i][0] + 1.5 * X[i][1] + rng.NextDouble() * 0.5;
        }
        return (X, Y);
    }

    public static LinearModel Train(double[][] trainX, double[] trainY, int epochs=100, double lr=0.01)
    {
        var model = new LinearModel { Weights = new double[trainX[0].Length], Bias = 0 };
        // ... your exact gradient descent loop from before ...
        for (int epoch = 0; epoch < epochs; epoch++) { /* ... */ }
        return model;
    }

    public static double Predict(LinearModel m, double[] x) => m.Bias + m.Weights[0]*x[0] + m.Weights[1]*x[1];
    // public static double MSE(LinearModel m, double[][] X, double[] Y) => /* your MSE logic */;
}