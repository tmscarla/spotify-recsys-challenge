/**
    @author: Simone Boglio
    @mail: simone.boglio@mail.polimi.it
*/

#ifndef SPLUS_H_
#define SPLUS_H_

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
                                    Value * detXlessY, Value * detYlessX, 
                                    Value * detX, Value * detY,
                                    Value n,
                                    Value l1, Value l2,
                                    Value t1, Value t2,
                                    Value c1, Value c2,
                                    Value shrink, Value threshold)
        :
        sums(user_count, 0),
        nonzeros(user_count, -1),
        detXlessY(detXlessY), detYlessX(detYlessX),
        detX(detX), detY(detY),
        n(n),
        l1(l1), l2(l2), 
        t1(t1), t2(t2), 
        c1(c1), c2(c2),
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
            Value detT=0, detC=0, val;
            Index index = head;
            Value xy = sums[index];
            if (n!=0){
                if(l1!=0)
                    detT = l1*(t1*(detXlessY[row]-xy)+ t2*(detYlessX[index]-xy) + xy);
                if(l2!=0)
                    detC = l2*(detX[row] * detY[index]);
                val = xy/(detT + detC + shrink);
            }
            else{
                if(shrink!=0)
                    val = xy/(1+shrink);
                else
                    val = xy;
            }
                
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
    Value *detXlessY, *detYlessX;
    Value *detX, *detY;
    Value n;
    Value l1, l2;
    Value t1, t2;
    Value c1, c2;
    Value shrink, threshold;
    Index row;
    Index head, length;
};
}  // namespace similarity
#endif  // SPLUS
