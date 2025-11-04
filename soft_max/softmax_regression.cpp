#include <iostream>
#include <vector>
#include <string>
#include <math.h>

class SoftmaxRegression
{
public:
    // Constructor
    SoftmaxRegression(double learning_rate = 0.01,
                      int epochs = 100,
                      int num_classes = 3): learning_rate(learning_rate), max_epochs(epochs), num_classes(num_classes)
    {
    
    }

    std::vector<double> get_y_vector(int x)
    {
        std::vector<double> y_vector(num_classes, 0.0);
        for(int i = 0 ; i < num_classes; i++)
        {
            if(i == x)
            {
                y_vector[i] = 1.0;
            }
        }
        return y_vector;
    }

    // Train modelu
    void fit(const std::vector<std::vector<double>>& X,
             const std::vector<int>& y)
    {
        weights = std::vector<std::vector<double>> (num_classes, std::vector<double>(X[0].size(), 0.0));
        bias = std::vector<double>(num_classes, 0.0);
        int count = 0;
        while(count < max_epochs)
        {
            for (int i = 0 ; i < X.size() ; ++i)
            {
                const std::vector<double> values = X[i];
                const std::vector<double> softmax_values = get_softmax_values_raw(values);
                const std::vector<double> vectorized_y = get_y_vector(y[i]);
                for(int j = 0 ; j < num_classes; j++)
                {
                    bias[j] -= (learning_rate*(softmax_values[j] - vectorized_y[j]));
                    for (int k = 0 ; k < X[0].size() ; ++k)
                    {
                        weights[j][k] = weights[j][k]- (learning_rate*(softmax_values[j] - vectorized_y[j])*X[i][k]);                    
                    }
                }
                
            }
            count++;
        }
        
    }


    double get_linear_values(const std::vector<double> &weights, const std::vector<double>& x) const
    {
        double ans = 0;
        for (int i = 0 ; i < weights.size() ; ++i)
        {
            ans += (weights[i]*x[i]);
        }
        return ans;
    }

    std::vector<double> get_softmax_values_raw(const std::vector<double>& x) const {
        std::vector<double> softmax_values;
        std::vector<double> linear_values;
        for (int i = 0 ; i < num_classes; i++)
        {
            auto linearized_result = get_linear_values(weights[i], x);
            linear_values.push_back(linearized_result);
        }
        softmax_values = get_softmax_values(linear_values);
        return softmax_values;
    }

    std::vector<double> get_softmax_values(const std::vector<double> &linearized_values) const
    {
        std::vector<double> exp_ans = linearized_values;
        double exp_sum = 0.0;
        for (auto value : linearized_values)
        {
            exp_sum += exp(value);
        }
        for(int i = 0 ; i < exp_ans.size() ; ++i)
        {
            auto curr_val = exp(linearized_values[i]);
            curr_val = curr_val/exp_sum;
            exp_ans[i] = curr_val;
        }

        return exp_ans;
    }


    // Predict one sample
    int predict_one(const std::vector<double>& x) const
    {
        std::vector<double> softmax_values;
        std::vector<double> linear_values;
        for (int i = 0 ; i < num_classes; i++)
        {
            auto linearized_result = get_linear_values(weights[i], x);
            linear_values.push_back(linearized_result);
        }
        softmax_values = get_softmax_values(linear_values);
        int class_ans = -1;
        auto greatest_softmax_value = 0.0;
        for (int i = 0 ; i < softmax_values.size(); ++i)
        {
            auto softmax_value = softmax_values[i];
            if(greatest_softmax_value < softmax_value)
            {
                class_ans = i;
                greatest_softmax_value = softmax_value;
            }

        }
        return class_ans;
    }

    // Predict for all samples
    std::vector<int> predict(const std::vector<std::vector<double>>& X) const
    {
        std::vector<int> predictions;
        for (int i = 0 ; i < X.size(); i++)
        {
            std::vector<double> x_i = X[i];
            predictions.push_back(predict_one(x_i));

        }
    
        return predictions;
    }

    // Print model weights and bias
    void print_weights() const
    {
        for (int i = 0; i < weights.size() ; i++)
        {
            std::cout<<"The class is "<<i<<std::endl;
            std::cout<<"The weights are: "<<std::endl;
            for (int j = 0 ; j < weights[i].size() ; j++)
            {
                std::cout<<weights[i][j]<<" ";
            }
            std::cout<<std::endl;
            std::cout<<"Bias is:"<<bias[i]<<std::endl;
        }
    }

private:
    // Parameters
    std::vector<std::vector<double>> weights; // [num_classes][num_features]
    std::vector<double> bias;                 // [num_classes]

    // Hyperparameters
    double learning_rate;
    int max_epochs;
    int num_classes;
};

int main()
{
    std::cout << "Starting Softmax Regression demo..." << std::endl;

    // Example dataset (3 classes, 2 features)
    std::vector<std::vector<double>> X = {
        {1.0, 5.1}, {1.0, 4.9}, {0.9, 5.1},   // class 0
        {5.0, 4.9}, {5.0, 5.2}, {5.1, 5.3},   // class 1
        {10, 5.1}, {9.9, 4.8}, {9.8, 5.0},    // class 2
    };

    std::vector<int> y = {
        0, 0, 0, 1, 1, 1, 2, 2, 2
    };

    // Initialize model
    SoftmaxRegression model(0.01, 500000, 3);

    // Train model
    model.fit(X, y);

    // Print learned weights
    model.print_weights();

    // Predict on the same data
    auto predictions = model.predict(X);

    std::cout << "Predictions:\n";
    for (auto p : predictions)
        std::cout << p << " ";
    std::cout << std::endl;

    std::cout << "Softmax Regression demo complete." << std::endl;
    return 0;
}
