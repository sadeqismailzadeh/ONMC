#include <iostream>
#include <vector>
#include <random>

// WalkerAlias class for generating random numbers from a discrete distribution
class WalkerAlias {
public:
    // Constructor takes as input a vector of weights representing the discrete distribution
    WalkerAlias(const std::vector<double>& weights) {
        int n = weights.size();
        std::vector<double> p(n);
        double sum = 0;
        // Compute the sum of the weights
        for (int i = 0; i < n; i++) {
            sum += weights[i];
        }
        // Normalize the weights to obtain probabilities
        for (int i = 0; i < n; i++) {
            p[i] = weights[i] * n / sum;
        }
        std::vector<int> small, large;
        // Split the probabilities into small and large
        for (int i = 0; i < n; i++) {
            if (p[i] < 1) {
                small.push_back(i);
            } else {
                large.push_back(i);
            }
        }
        // Construct the alias table
        while (!small.empty() && !large.empty()) {
            int l = small.back();
            small.pop_back();
            int g = large.back();
            large.pop_back();
            prob[l] = p[l];
            alias[l] = g;
            p[g] = p[g] + p[l] - 1;
            if (p[g] < 1) {
                small.push_back(g);
            } else {
                large.push_back(g);
            }
        }
        while (!large.empty()) {
            int g = large.back();
            large.pop_back();
            prob[g] = 1;
        }
        while (!small.empty()) {
            int l = small.back();
            small.pop_back();
            prob[l] = 1;
        }
    }

    // Sample function returns a random number from the distribution using the Walker's alias algorithm
    int sample() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0, 1);
        int n = prob.size();
        int i = std::floor(dis(gen) * n);
        double p = dis(gen);
        if (p <= prob[i]) {
            return i;
        } else {
            return alias[i];
        }
    }

private:
    std::vector<double> prob; // probability table
    std::vector<int> alias; // alias table
};

int main() {
    std::vector<double> weights = {0.2, 0.3, 0.1, 0.4}; // weights representing the discrete distribution
    WalkerAlias wa(weights); // create a WalkerAlias object
    int n = 10; // number of samples to generate
    for (int i = 0; i < n; i++) {
        std::cout << wa.sample() << ' '; // generate a random number from the distribution
    }
    std::cout << '\n';
}
