/**
    @author: Simone Boglio
    @mail: simone.boglio@mail.polimi.it
*/

#ifndef P3ALPHA_RP3BETA_H_
#define P3ALPHA_RP3BETA_H_

#include <algorithm>
#include <vector>
#include <utility>
#include <functional>

namespace similarity {

/** Functor that stores the Top K (Value/Index) pairs
 passed to it in and stores in its results member
 */
template <typename Index, typename Value>
struct TopK {
    explicit TopK(size_t K) : K(K) {}

    void operator()(Index index, Value score) {
        if ((results.size() < K) || (score > results[0].first)) {
            if (results.size() >= K) {
                std::pop_heap(results.begin(), results.end(), heap_order);
                results.pop_back();
            }

            results.push_back(std::make_pair(score, index));
            std::push_heap(results.begin(), results.end(), heap_order);
        }
    }

    size_t K;
    std::vector<std::pair<Value, Index> > results;
    std::greater<std::pair<Value, Index> > heap_order;
};

/** A utility class to multiply rows of a sparse matrix
 Implements the sparse matrix multiplication algorithm
 described in the paper 'Sparse Matrix Multiplication Package (SMMP)'
 http://www.i2m.univ-amu.fr/~bradji/multp_sparse.pdf
*/
template <typename Index, typename Value>
class SparseMatrixMultiplier {
 public:
    explicit SparseMatrixMultiplier(Index user_count, 
                                    Value * detPop,
                                    Value alpha,
                                    Value mode,
                                    Value shrink, Value threshold)
        :
        sums(user_count, 0),
        nonzeros(user_count, -1),
        detPop(detPop),
        alpha(alpha),
        mode(mode),
        shrink(shrink), threshold(threshold),
        head(-2), length(0) {
    }

    /** Adds value to the item at index */
    void add(Index index, Value value) {
        sums[index] += value;

        if (nonzeros[index] == -1) {
            nonzeros[index] = head;
            head = index;
            length += 1;
        }
    }

    void setRow(Index index) {
        row = index;
    }

    /** Calls a function once per non-zero entry, also clears state for next run*/
    template <typename Function>
    void foreach(Function & f) {  // NOLINT(*)

        for (int i = 0; i < length; ++i) {
            Value val;
            Index index = head;
            if (mode==0)
                val = std::pow(sums[index],alpha) / (detPop[index]+shrink);
            else //mode==1
                val = std::pow(sums[index],alpha) / (detPop[row]+shrink);
            if (val >= threshold)
                f(index, val);
            // clear up memory and advance linked list
            head = nonzeros[head];
            sums[index] = 0;
            nonzeros[index] = -1;
        }
        length = 0;
        head = -2;
    }

    Index nnz() const { return length; }

 protected:
    std::vector<Value> sums;
    std::vector<Index> nonzeros;
    Value *detPop;
    Value alpha;
    Value mode;
    Value shrink, threshold;
    Index row;
    Index head, length;
};
}  // namespace similarity
#endif  // P3APLHA_RP3BETA
