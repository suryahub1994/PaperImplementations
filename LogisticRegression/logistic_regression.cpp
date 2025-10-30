#include <iostream>
#include <vector>
#include <string>
#include <math.h>

class LogisticRegression
{
public:
    LogisticRegression(double lr = 0.01, int epochs = 1000)
        : learning_rate(lr), max_epochs(epochs)
    {
        std::cout << "LogisticRegression initialized with lr = "
                  << learning_rate << ", epochs = " << max_epochs << "\n";
    }

    void fit(const std::vector<std::vector<double>>& X,
             const std::vector<int>& y)
    {
        weights = std::vector<double>(X[0].size(), 0.0);
        int count = 0;
        while(count < max_epochs)
        {
            std::vector<double> predict_class = predict_proba(X);
            std::vector<double> nudges = get_dw(predict_class, X, y);
            for (int i = 0 ; i < nudges.size() ; i++)
            {
                weights[i] -= learning_rate*nudges[i];
            }
            bias -= learning_rate*get_db(predict_class, X, y);
            count++;
        }
    }

    double get_db(const std::vector<double> predict_class, const std::vector<std::vector<double>>& X,const std::vector<int>& y) const
    {
        double value = 0;
        for(int i = 0 ; i< y.size() ; ++i)
        {
            value += (predict_class[i]-y[i]);
        }
        return value/1.0*predict_class.size();
    }

    std::vector<double> get_dw(const std::vector<double> predict_class, const std::vector<std::vector<double>>& X,const std::vector<int>& y) const
    {
        std::vector<double> nudges(X[0].size(), 0.0);
        for(int i = 0 ; i< y.size() ; ++i)
        {
            for (int j = 0 ; j < X[0].size(); ++j)
            {
                nudges[j] += (predict_class[i]-y[i])*X[i][j];
            }
        }
        for (int j = 0 ; j < X[0].size(); j++)
        {
            nudges[j] = nudges[j]/(1*y.size());
        }
        return nudges;
    }

    double get_y_value(const std::vector<double> &x) const
    {
        double y = 0;
        for (int i = 0 ; i < x.size() ; ++i)
        {
            y += (x[i]*weights[i]);
        }
        y += bias;
        return y;
    }

    std::vector<double> predict_proba(const std::vector<std::vector<double> >& X) const
    {
        std::vector<double> probabilities;
        for (std::vector<double> x: X)
        {
            auto y_value = get_y_value(x);
            probabilities.push_back(sigmoid(y_value));
        }
        return probabilities;
    }

    std::vector<int> predict(const std::vector<std::vector<double>>& X) const
    {
        const double threshold = 0.5;
        std::cout << "Predicting binary outputs (0 or 1)...\n";
        std::vector<double> probabilities = predict_proba(X);
        std::vector<int> classes;
        for (auto probability : probabilities)
        {
            if(probability >= threshold)
                classes.push_back(1);
            else
                classes.push_back(0);
        }
        return classes;
    }

    void print_weights() const
    {
        std::cout << "Weights: ";
        for (auto w : weights) std::cout << w << " ";
        std::cout << "\nBias: " << bias << "\n";
    }

private:
    double sigmoid(double z) const
    {
        return 1.0/(1.0+exp(-1*z));
    }

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
    std::cout << "Starting Logistic Regression demo...\n";

    // Example input (binary classification)
    std::vector<std::vector<double>> X = {
        {0, 0, 0},
        {0, 1, 0},
        {1, 0, 1},
        {1, 1, 1},
        {0, 0, 1},
        {0, 1, 1}  
    };
    std::vector<int> y = {0, 0, 0, 1, 0, 0};

    LogisticRegression model(0.1, 1000);
    model.fit(X, y);
    model.print_weights();

    auto preds = model.predict(X);
    std::cout << "Predictions:\n";
    for (auto p : preds) std::cout << p << " ";
    std::cout << "\n";

    std::cout << "Done.\n";
    return 0;
}
