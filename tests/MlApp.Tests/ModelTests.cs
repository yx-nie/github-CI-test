using Xunit;
using System.Linq;

public class ModelTests
{
    [Fact] // Requirement: data generation shape verification
    public void DataGeneration_Shape_IsCorrect()
    {
        var (X, Y) = Trainer.GenerateData(200, 42);
        Assert.Equal(200, X.Length);
        Assert.Equal(2, X[0].Length); // 2 features
        Assert.Equal(160, X.Take(160).Count()); // train split
        Assert.Equal(40, X.Skip(160).Count());  // test split
    }

    [Fact] // Requirement: training convergence checks
    public void Training_Converges_To_True_Weights()
    {
        var (X, Y) = Trainer.GenerateData(200, 42);
        var model = Trainer.Train(X.Take(160).ToArray(), Y.Take(160).ToArray());
        
        // True weights are 3.0 and 1.5
        Assert.InRange(model.Weights[0], 2.9, 3.1);
        Assert.InRange(model.Weights[1], 1.4, 1.6);
    }

    [Fact] // Requirement: prediction accuracy bounds
    public void Prediction_MSE_Is_Below_Threshold()
    {
        var (X, Y) = Trainer.GenerateData(200, 42);
        var model = Trainer.Train(X.Take(160).ToArray(), Y.Take(160).ToArray());
        double mse = Trainer.MSE(model, X.Skip(160).ToArray(), Y.Skip(160).ToArray());
        
        Assert.True(mse < 1.0, $"Test MSE {mse} exceeds 1.0");
    }
}