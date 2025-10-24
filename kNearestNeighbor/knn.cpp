#include <ostream>
#include <queue>
#include <vector>
#include <map>
#include <string>
#include <iostream>
#include <cmath>

using Point = std::vector<double>;

double euclidean_distance(const Point &p1, const Point &p2)
{
    double distance = 0;
    for (int i = 0; i < p1.size(); ++i)
    {
        distance += pow(p1[i] - p2[i], 2);
    }
    return pow(distance, 0.5);
}

struct PointDistance
{
    Point point;
    double distance;
    int idx;

    bool operator<(const PointDistance &p2) const
    {
        return this->distance < p2.distance;
    }
};

std::string knn_predict(
    const std::vector<Point> &X_train,
    const std::vector<std::string> &y_train,
    const Point &X_test,
    int k)
{
    std::priority_queue<PointDistance> distanceHeap;

    for (int i = 0; i < X_train.size(); i++)
    {
        double distanceFromXTest = euclidean_distance(X_train[i], X_test);
        PointDistance p1;
        p1.distance = distanceFromXTest;
        p1.point = X_train[i];
        p1.idx = i;
        distanceHeap.push(p1);

        if (distanceHeap.size() > k)
        {
            distanceHeap.pop();
        }
    }

    std::map<std::string, int> voteOfPoints;
    int currentCount = 0;
    std::string category = "";
    std::cout<<"-------------------------------------"<<std::endl;
    std::cout<<"The point in consideration: "<< X_test[0] <<" "<<X_test[1]<<std::endl;
    while (!distanceHeap.empty())
    {
        PointDistance point_distance = distanceHeap.top();
        std::cout<<"The Neighbor is : "<< point_distance.point[0] <<" "<<point_distance.point[1] <<" "<<y_train[point_distance.idx]<<std::endl;
        distanceHeap.pop();

        voteOfPoints[y_train[point_distance.idx]] += 1;

        if (currentCount < voteOfPoints[y_train[point_distance.idx]])
        {
            category = y_train[point_distance.idx];
            currentCount = voteOfPoints[y_train[point_distance.idx]];
        }
    }
     std::cout<<"-------------------------------------"<<std::endl;

    return category;
}

int main()
{
    std::vector<Point> X_train = {
        {1.0, 2.0}, {2.0, 3.0}, {3.0, 3.0}, {6.0, 5.0}, {7.0, 8.0}};
    std::vector<std::string> y_train = {"A", "A", "A", "B", "B"};

    Point X_test = {5.0, 5.0};
    int k = 3;

    std::string prediction = knn_predict(X_train, y_train, X_test, k);
    std::cout << "Predicted label: " << prediction << std::endl;

    Point X_test_1 = {0.0 , 0.0};
    std::string prediction_1 = knn_predict(X_train, y_train, X_test_1, k);
    std::cout << "Predicted label: " << prediction_1 << std::endl;


    Point X_test_2 = {9.0 , 9.0};
    std::string prediction_2 = knn_predict(X_train, y_train, X_test_2, k);
    std::cout << "Predicted label: " << prediction_2 << std::endl;
    return 0;
}// I think this implementation is correct
