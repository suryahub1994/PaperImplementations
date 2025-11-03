#include <iostream>
#include <vector>
#include <string>
#include <math.h>
class RMSOptimizer
{
public:
    RMSOptimizer(double lr = 0.01, int epochs = 100, double beta = 0.9)
        : learning_rate(lr), max_epochs(epochs), beta(beta)
    {
        std::cout << "SGD initialized with lr = "
                  << learning_rate << ", epochs = " << max_epochs << "\n";
    }

    void fit(const std::vector<std::vector<double>>& X,
             const std::vector<double>& y)
    {
        if(X.size() == 0)
            return ;
        weights = std::vector<double>(X[0].size(), 0);
        average_square_gradient = std::vector<double>(X[0].size(), 0);
        bias = 0;
        bias_sum = 0.0;
        std::cout << "Starting training...\n";
        int count = 0;
        while(count < max_epochs)
        {
            for (int i = 0 ; i < X.size() ; ++i)
            {
               double y_hat = predict_one(X[i]);
               double y_ground = y[i];
               double gradient_bias = get_nudge_bias(y_hat, y_ground, X[i]);
               bias_sum = beta*bias_sum + (1-beta)*gradient_bias*gradient_bias;
               bias -= learning_rate * gradient_bias / (std::sqrt(bias_sum + epsilon));
               for (int j = 0 ; j < X[i].size() ; j++)
               {
                    double instantaneous_gradient = (y_hat-y_ground)*X[i][j];
                    average_square_gradient[j] = beta*average_square_gradient[j] +(1-beta)*instantaneous_gradient*instantaneous_gradient;
                    weights[j] -= learning_rate*instantaneous_gradient/(std::sqrt(average_square_gradient[j] + epsilon));
               }

            }
            count++;
        } 
    }

    std::vector<double> get_nudge(double y_actual, double y_predicted, const std::vector<double> &x) {
        return std::vector<double>(2, 0);
    }

    double get_nudge_bias(double y_hat, double y_ground, const std::vector<double> &x) {
        return (y_hat-y_ground);
    }

    double predict_one(const std::vector<double>& x) const
    {
        double value = 0;
        for (int i = 0 ; i < x.size(); ++i)
        {
            value += weights[i]*x[i];
        }
        return value+bias;
    }

    std::vector<double> predict(const std::vector<std::vector<double>>& X) const
    {
        std::vector<double> predictions;
        for (const std::vector<double> &x: X)
        {
            predictions.push_back(predict_one(x));
        }
        return predictions;
    }

    void print_weights() const
    {
        std::cout << "Printing model weights...: \n";
        for (auto weight : weights)
        {
            std::cout<<weight<<" ";
        }
        std::cout<<"Printing bias: ";
        std::cout<<bias<<std::endl;
    }

private:
    std::vector<double> weights;
    std::vector<double> average_square_gradient;
    double bias_sum = 0.0;
    double bias{};
    double learning_rate;
    int max_epochs;
    double beta;
    const double epsilon = 0.0000001;
};

// ----------------------------------------------------
// Main driver
// ----------------------------------------------------
int main()
{
    std::cout << "Starting RMS Prop optimizer"<< std::endl;

    std::vector<std::vector<double>> X = {
        {1.0}, {2.0}, {3.0}, {4.0}, {5.0}
    };
    std::vector<double> y = {3.0, 5.0, 7.0, 9.0, 11.0};

    RMSOptimizer model(0.01, 1000000, 0.9);
    model.fit(X, y);
    model.print_weights();

    auto preds = model.predict(X);
    std::cout << "Predictions:\n";
    for (auto p : preds) std::cout << p << " ";
    std::cout << "\n";

    std::cout << "Done.\n";
    return 0;
}
