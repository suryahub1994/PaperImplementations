#include <algorithm>
#include <vector>
#include <iostream>
#include <unordered_map>
#include <cmath>
#include <cstdlib>
#include <ctime>

class KMeans {
public:
    KMeans(int k_input, int max_iters_input = 100)
        : k(k_input), max_iters(max_iters_input) {
        srand((unsigned)time(nullptr));
    }

    double dist(const std::vector<double>& X, const std::vector<double>& centroid) const {
        double distanceSum = 0.0;
        for (int i = 0; i < X.size(); i++)
            distanceSum += (centroid[i] - X[i]) * (centroid[i] - X[i]);
        return std::sqrt(distanceSum);
    }

    std::vector<double> getMeanCentroid(const std::vector<std::vector<double>>& partition, int dims) const {
        if (partition.empty())
            return std::vector<double>(dims, 0.0);

        std::vector<double> centroid(dims, 0.0);
        for (const auto& point : partition)
            for (int i = 0; i < dims; ++i)
                centroid[i] += point[i];

        for (int i = 0; i < dims; ++i)
            centroid[i] /= partition.size();

        return centroid;
    }

    std::vector<std::vector<double>> getNewCentroids(
        const std::vector<std::vector<std::vector<double>>>& partitions, int dims) const
    {
        std::vector<std::vector<double>> newCentroids;
        newCentroids.reserve(partitions.size());
        for (const auto& partition : partitions)
            newCentroids.push_back(getMeanCentroid(partition, dims));
        return newCentroids;
    }

        void fit(const std::vector<std::vector<double>>& X) {
        int n = X.size();
        int dims = X[0].size();

        std::vector<double> minv(dims, 1e9), maxv(dims, -1e9);
        for (const auto& row : X)
            for (int j = 0; j < dims; ++j) {
                minv[j] = std::min(minv[j], row[j]);
                maxv[j] = std::max(maxv[j], row[j]);
            }
        centroids.assign(k, std::vector<double>(dims));
        for (int i = 0; i < k; ++i)
            for (int j = 0; j < dims; ++j)
                centroids[i][j] = minv[j] + (maxv[j] - minv[j]) * ((double)rand() / RAND_MAX);

        int count = 0;
        while (count < max_iters) {
            std::unordered_map<int, int> pointToCluster;
            std::unordered_map<int, double> minDistanceMap;

            for (int i = 0; i < n; i++) {
                for (int j = 0; j < k; j++) {
                    double distance = dist(X[i], centroids[j]);
                    if (!minDistanceMap.count(i)) minDistanceMap[i] = 1e18;
                    if (distance < minDistanceMap[i]) {
                        minDistanceMap[i] = distance;
                        pointToCluster[i] = j;
                    }
                }
            }

            std::vector<std::vector<std::vector<double>>> partitions(k);
            for (const auto& kv : pointToCluster)
                partitions[kv.second].push_back(X[kv.first]);

            auto newCentroids = getNewCentroids(partitions, dims);

            for (int i = 0; i < k; ++i) {
                std::cout << "Centroid " << i << ": ";
                for (auto val : newCentroids[i])
                    std::cout << val << " ";
                std::cout << "\n";
            }

            if (centroids == newCentroids) break;
            centroids = std::move(newCentroids);
            count++;
        }
    }

    std::vector<int> predict(const std::vector<std::vector<double>>& X) const {
        std::vector<int> predictions;
        for (const auto& point : X) {
            int cluster_idx = 0;
            double minDist = 1e18;
            for (int j = 0; j < centroids.size(); j++) {
                double distance = dist(point, centroids[j]);
                if (distance < minDist) {
                    minDist = distance;
                    cluster_idx = j;
                }
            }
            predictions.push_back(cluster_idx);
        }
        return predictions;
    }

    const std::vector<std::vector<double>>& get_centroids() const { return centroids; }

private:
    int k;
    int max_iters;
    std::vector<std::vector<double>> centroids;
};
