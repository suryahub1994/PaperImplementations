#include <iostream>
#include <vector>

class Momentum
{
public:
    Momentum(double lr = 0.01, double momentum = 0.9, int epochs = 100)
        : learning_rate(lr), beta(momentum), max_epochs(epochs)
    {
        std::cout << "Momentum Optimizer initialized with lr = "
                  << learning_rate << ", momentum = " << beta
                  << ", epochs = " << max_epochs << "\n";
    }

    std::vector<double> get_dw(double diff, const std::vector<double> &X)
    {
        std::vector<double> weights(X.size(), 0.0);
        for (int i = 0 ; i < X.size() ; i++)
        {
            weights[i] = diff * X[i];
        }
        return weights;
    }

    void fit(const std::vector<std::vector<double>>& X,
             const std::vector<double>& y)
    {
        std::cout << "Starting training with Momentum...\n";
        weights = std::vector<double>(X[0].size(), 0.0);
        velocity = std::vector<double>(X[0].size()+1, 0.0); // last one for bias
        int count = 0;

        while(count < max_epochs)
        {
            for (int i = 0 ; i < X.size() ; i++)
            {
                double predict_current = predict_one(X[i]);
                double value = y[i];
                double diff = predict_current - value;

                // bias update
                velocity[X[i].size()] = (1 - beta) * diff + beta * velocity[X[i].size()];
                bias -= learning_rate * velocity[X[i].size()];

                // weight update
                std::vector<double> dw = get_dw(diff, X[i]);
                for (int j = 0 ; j < X[i].size() ; j++)
                {
                    velocity[j] = beta * velocity[j] + (1 - beta) * dw[j];
                    weights[j] -= learning_rate * velocity[j];
                }
            }
            count++;
        }
    }

    double predict_one(const std::vector<double>& x) const
    {
        double prediction = 0.0;
        for (int i = 0 ; i < x.size() ; i++)
        {
            prediction += weights[i] * x[i];
        }
        return prediction + bias;
    }

    std::vector<double> predict(const std::vector<std::vector<double>>& X) const
    {
        std::vector<double> predictions;
        for (int i = 0 ; i < X.size(); i++)
        {
            double predict = predict_one(X[i]);
            predictions.push_back(predict);
        }
        return predictions;
    }

    void print_weights() const
    {
        std::cout << "Printing model weights and bias...\n";
        for (auto weight : weights)
        {
            std::cout << weight << " ";
        }
        std::cout << bias << std::endl;
    }

private:
    std::vector<double> weights;
    std::vector<double> velocity; // momentum term
    double bias{};
    double learning_rate;
    double beta; // momentum coefficient
    int max_epochs;
};

// ----------------------------------------------------
// Main driver
// ----------------------------------------------------
int main()
{
    std::cout << "Starting Momentum demo...\n";

    std::vector<std::vector<double>> X = {
        {1.0}, {2.0}, {3.0}, {4.0}, {5.0}
    };
    std::vector<double> y = {3.0, 5.0, 7.0, 9.0, 11.0};

    Momentum model(0.01, 0.9, 100000);
    model.fit(X, y);
    model.print_weights();

    auto preds = model.predict(X);
    std::cout << "Predictions:\n";
    for (auto p : preds) std::cout << p << " ";
    std::cout << "\nDone.\n";

    return 0;
}
