#include <iostream>
#include <vector>
#include <string>

class SGD
{
public:
    SGD(double lr = 0.01, int epochs = 100)
        : learning_rate(lr), max_epochs(epochs)
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
        bias = 0;
        std::cout << "Starting training...\n";
        int count = 0;
        while(count < max_epochs)
        {
            for (int i = 0 ; i < X.size() ; ++i)
            {
               double y_hat = predict_one(X[i]);
               auto dw = get_nudge(y[i], y_hat, X[i]);
               bias -= learning_rate * get_nudge_bias(y[i], y_hat, X[i]);
               for (int j = 0 ; j < X[i].size() ; ++j)
               {
                    weights[j] -= learning_rate*dw[j];
               }
            }
            count++;
        } 
    }

    std::vector<double> get_nudge(double y_actual, double y_predicted, const std::vector<double> &x) {
        std::vector<double> nudge_vector(x.size(), 0);
        for (int i = 0 ; i < x.size() ; i++)
        {
            nudge_vector[i] = (y_predicted - y_actual)*x[i];
        }
        return nudge_vector;
    }

    double get_nudge_bias(double y_actual, double y_predicted, const std::vector<double> &x) {
        return (y_predicted - y_actual);
    }

    double predict_one(const std::vector<double>& x) const
    {
        std::cout << "Predicting for one sample...\n";
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
    double bias{};
    double learning_rate;
    int max_epochs;
};

// ----------------------------------------------------
// Main driver
// ----------------------------------------------------
int main()
{
    std::cout << "Starting SGD demo...\n";

    std::vector<std::vector<double>> X = {
        {1.0}, {2.0}, {3.0}, {4.0}, {5.0}
    };
    std::vector<double> y = {3.0, 5.0, 7.0, 9.0, 11.0};

    SGD model(0.01, 100000);
    model.fit(X, y);
    model.print_weights();

    auto preds = model.predict(X);
    std::cout << "Predictions:\n";
    for (auto p : preds) std::cout << p << " ";
    std::cout << "\n";

    std::cout << "Done.\n";
    return 0;
}
