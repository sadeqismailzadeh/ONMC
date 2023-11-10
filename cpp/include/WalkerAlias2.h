//
// Created by Sadeq Ismailzadeh on ۰۸/۰۱/۲۰۲۳.
//

#ifndef SRC_WALKERALIAS2_H
#define SRC_WALKERALIAS2_H

#include <random>
#include <vector>
#include <algorithm>


class AliasSampler
// This class implements the Walker-Vose Alias Sampling method.
//
// The initializing weights do not have to be normalized.
// The algorithm is described
// [here](https://web.archive.org/web/20131029203736/http://web.eecs.utk.edu/~vose/Publications/random.pdf)
// The naming of variables follows the Wikipedia [article](https://en.wikipedia.org/wiki/Alias_method) (As of 2022-10-31).
{
    typedef double real;
public:
//    AliasSampler() = delete;
    AliasSampler() = default;

    explicit AliasSampler(const std::vector<real>& weights)
            : K(weights.size())
    {
        // [...] If Ui = 1, the corresponding value Ki will never be consulted and is unimportant,
        //      but a value of Ki = i is sensible. [...]

        std::iota(K.begin(), K.end(), 0);

        p.reserve(weights.size());
        std::transform(
                weights.begin(), weights.end(),
                std::back_inserter(p),
                [result = std::reduce(weights.begin(), weights.end())]
                        (real w) -> real
                {
                    return w / result;
                }
        );

        U.reserve(weights.size());
        std::transform(
                p.begin(), p.end(),
                std::back_inserter(U),
                [&, this]
                        (real x) -> real
                {
                    return p.size() * x;
                }
        );

        // [...] As the lookup procedure is slightly faster if y < Ui (because Ki does not need to be consulted),
        //      one goal during table generation is to maximize the sum of the Ui.
        //      Doing this optimally turns out to be NP hard,  but a greedy algorithm comes reasonably close: rob from the richest and give to the poorest.
        //      That is, at each step choose the largest Ui and the smallest Uj.
        //      Because this requires sorting the Ui, it requires O(n log n) time. [...] (See the Wikipedia article)
        // For this reason we partition into small and large indices and use them in a sorted fashion.

        std::vector<std::size_t> indices(U.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(),
                  [&, this]
                          (int a, int b) -> bool
                  {
                      return U[a] < U[b];
                  });

        // I could use std::partition for partitioning into smaller and larger, **but**
        // AFAIK this would not make use of the fact that the array is already sorted.
        std::vector<std::size_t> smaller, larger;

        for (std::size_t i = 0; U[indices[i]] < 1; ++i) {
            smaller.push_back(indices[i]);
        };

        for (std::size_t i = U.size() - 1; U[indices[i]] >= 1; --i) {
            larger.push_back(indices[i]);
        };

        while (!smaller.empty() && !larger.empty()) {
            std::size_t s = pop(smaller);
            std::size_t l = pop(larger);
            K[s] = l;
            U[l] = U[l] - (1. - U[s]);
            if (U[l] < 1) {
                smaller.push_back(l);
            } else {
                larger.push_back(l);
            };
        };

        // [...] If one category empties before the other, the remaining entries may
        //      have U_i set to 1 with negligible error. [...] (See the Wikipedia article)
        while (!smaller.empty()) {
            std::size_t s = pop(smaller);
            U[s] = 1.;
        };

        while (!larger.empty()) {
            std::size_t l = pop(larger);
            U[l] = 1.;
        };
    };

    const auto& probabilities() const
    {
        return p;
    };


    template <typename RNE>
    auto operator()(RNE& gen) const
    // Return a random number according to the given probabilities
    // at initialization using a Random Number Engine.
    {
        auto x = get_r(gen);
        auto i = static_cast<size_t>(p.size() * x);
        auto y = p.size() * x - i;

        return y < U[i] ? i : K[i];
    };

    auto min() const
    {
        return 0;
    };

    auto max() const
    {
        return p.size() - 1;
    };

    template <typename T>
    T pop(std::vector<T>& V)
    // Remove and return the last element of a vector.
    {
        auto val = V.back();
        V.pop_back();
        return val;
    }


    template <typename RNE>
    auto get_r(RNE& gen)
    // Return a real number from [0, 1) using a Random Number Engine
    {
        static std::uniform_real_distribution<real> dist(0.0, 1.0);
        return dist(gen);
    };

private:
    std::vector<real> p{};
    std::vector<std::size_t> K{};
    std::vector<real> U{};
//    std::vector<std::size_t> indices_{};
};
#endif //SRC_WALKERALIAS2_H
