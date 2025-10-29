#include <iostream>
#include <vector>
#include <string>

class LinearRegression
{
public:
    LinearRegression(double lr = 0.01, int epochs = 1000)
        : learning_rate(lr), max_epochs(epochs)
    {
    }

    void fit(const std::vector<std::vector<double>>& X,
             const std::vector<double>& y)
    {
        weights = std::vector<double>(X[0].size(), 0);
        bias = 0;
        int count = 0;
        while (count < max_epochs)
        {
            std::vector<double> y_results = predict(X);
            auto dw = get_dw(y, y_results, X);
            auto db = get_db(y, y_results);
            for (int j = 0; j < weights.size(); ++j)
                weights[j] -= learning_rate * dw[j];
            bias -= learning_rate * db;
            std::cout<<"For iteration: "<<count<<" "<<weights[0]<<" "<<bias<<std::endl;
            count++;
        }
    }

        std::vector<double> get_dw(const std::vector<double>& y,
                            const std::vector<double>& y_results,
                            const std::vector<std::vector<double>>& X)
    {
        double size = y.size() * 1.0;
        std::vector<double> direction_mag_vector(X[0].size(), 0.0);
        for (int i = 0; i < X.size(); i++)
        {
            for (int j = 0; j < X[0].size(); j++)
            {
                direction_mag_vector[j] += (y_results[i] - y[i]) * X[i][j];
            }
        }
        for (int i = 0; i < direction_mag_vector.size(); i++)
        {
            direction_mag_vector[i] /= size;
        }
        return direction_mag_vector;
    }

    double get_db(const std::vector<double>& y,
                const std::vector<double>& y_results)
    {
        double sum_of_difference = 0.0;
        double size = y.size() * 1.0;
        for (int i = 0; i < y.size(); i++)
        {
            sum_of_difference += (y_results[i] - y[i]);
        }
        return sum_of_difference / size;
    }


    std::vector<double> predict(const std::vector<std::vector<double>>& X) const
    {
        std::vector<double> predictions;
        for(auto sample : X)
        {
            double ans = 0;
            for (int i = 0 ; i < sample.size() ; i++)
            {
                if(i < weights.size())
                {
                    ans += sample[i]*weights[i];
                }
            }
            ans += bias;
            predictions.push_back(ans);
        }
        return predictions;
    }

    void print_weights() const
    {
        for (auto weight : weights)
        {
            std::cout<<weight<<" ";
        }
        std::cout<<std::endl;
    }

private:
    std::vector<double> weights;
    double bias{};
    double learning_rate;
    int max_epochs;
};

int main()
{
    std::cout << "Starting Linear Regression demo...\n";
    std::vector<std::vector<double>> X = {
        {1.0}, {2.0}, {3.0}, {4.0}, {5.0}
    };
    std::vector<double> y = {3.0, 5.0, 7.0, 9.0, 11.0};

    LinearRegression model(0.01, 15000);
    model.fit(X, y);
    model.print_weights();
    auto preds = model.predict(X);

    std::cout << "Predictions:\n";
    for (auto p : preds) std::cout << p << " ";
    std::cout << "\n";

    std::cout << "Done.\n";
    return 0;
}
