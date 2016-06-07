#ifndef Tensor_HH
#define Tensor_HH

#include <sstream>
#include <vector>
#include <functional>
#include <memory>
#include <stdexcept>
#include <cmath>
#include <cstdlib>
#include <iterator>
#include <vector>
#include <algorithm>

// ****************************************************************
// *************************** Tensor *****************************
// ****************************************************************

template <typename Container>
std::string join(const Container& vec, const char* delim)
{
    std::stringstream ss;
    std::ostream_iterator<int> out_it (ss,delim);
    std::copy( vec.begin(), vec.end(), out_it );
    return ss.str();
}

template <typename Type>
class Tensor
{
public:
    typedef std::vector<Type> Data;
    typedef std::vector<std::size_t> Shape;

    friend class Accessor;
    struct Accessor
    {
        static       Data&  data (      Tensor<Type>& a) { return *(a.data_); }
        static const Data&  data (const Tensor<Type>& a) { return *(a.data_); }
        static       Shape& shape(      Tensor<Type>& a) { return a.shape_; }
        static const Shape& shape(const Tensor<Type>& a) { return a.shape_; }
    };

private:
    std::shared_ptr<Data> data_;
    Shape                 shape_;

    std::size_t offsetOf(const std::initializer_list<std::size_t>& indexes) const
    {
        if (shape_.size() != indexes.size())
        {
            std::stringstream ss;
            ss << "Tensor accessed with incorrect number of indexes"
               << " Shape: "   << join(shape_, "x")
               << " indexes: " << join(indexes, ",");
            throw std::runtime_error(ss.str());
        }

        std::size_t offset = 0;
        std::size_t rank   = 0;
        std::size_t rank_scale = 1;
        for (std::size_t idx : indexes)
        {
            if (shape_[rank] < idx)
            {
                std::stringstream ss;
                ss << "Tensor index out of range "
                   << " at rank:" << rank
                   << " Shape: "   << join(shape_, "x")
                   << " indexes: " << join(indexes, ",");
                throw std::runtime_error(ss.str());
            }

            if (rank == 0)
            {
                offset     = idx;
                rank_scale = shape_[rank];
            }
            else
            {
                offset += rank_scale * idx;
                rank_scale *= shape_[rank];
            }

            ++rank;
        }
        return offset;
    }



public:
    // friend class TensorUtils<Type>;
    //public:

    Tensor()
    {}

    Tensor(const Shape& shape) :
        data_(new Data),
        shape_(shape)
    {
        data_->resize(size());
    }

    template <std::size_t N>
    Tensor(Type (&raw)[N],
           const Shape& shape) :
        data_(new Data),
        shape_(shape)
    {
        std::size_t theSize = size();
        if (theSize != N )
        {
            std::stringstream ss;
            ss << "Tensor shape wrong for supplied data"
               << " N: " << N
               << " shape: " << join(shape_, "x")
               << " hence size:" << theSize;
            throw std::runtime_error(ss.str());
        }

        data_->resize(N);
        data_->assign(raw,raw+N);
    }

    Tensor(const std::initializer_list<Type>& init,
           const Shape& shape):
        data_(new Data),
        shape_(shape)
    {
        std::size_t theSize = size();
        if ((init.end()-init.begin()) != theSize )
        {
            std::stringstream ss;
            ss << "Tensor shape wrong for supplied data"
               << " (end-begin): " << (init.end()-init.begin())
               << " shape: " << join(shape_, "x")
               << " hence size:" << theSize;
            throw std::runtime_error(ss.str());
        }

        data_->resize(size());
        data_->assign(init.begin(),init.end());
    }

    Tensor(Type* begin, Type* end,
           const Shape& shape):
        data_(new Data),
        shape_(shape)
    {
        std::size_t theSize = size();
        if ((end-begin) != theSize )
        {
            std::stringstream ss;
            ss << "Tensor shape wrong for supplied data"
               << " (end-begin): " << (end-begin)
               << " shape: " << join(shape_, "x")
               << " hence size:" << theSize;
        }

        data_->resize(size());
        data_->assign(begin,end);
    }

    Tensor(std::shared_ptr<Data> data,
           const Shape& shape):
        data_(data),
        shape_(shape)
    {}


    std::size_t size() const
    {
        std::size_t theSize = 1;
        for (Shape::const_iterator sit = shape_.begin();
             sit != shape_.end();
             ++sit)
        {
            theSize *= *sit;
        }
        return theSize;
    }

    Type& at(const std::initializer_list<std::size_t>& indexes)
    {
        return (*data_)[offsetOf(indexes)];
    }

    Type at(const std::initializer_list<std::size_t>& indexes) const
    {
        return (*data_)[offsetOf(indexes)];
    }

    Type operator[](std::size_t i)
    {
        return (*data_)[i];
    }
};

// ****************************************************************
// ************************ Tensor UTILS **************************
// ****************************************************************

template <typename Type>
struct TensorUtils : public Tensor<Type>::Accessor
{
    typedef typename Tensor<Type>::Data::iterator       iterator;
    typedef typename Tensor<Type>::Data::const_iterator const_iterator;

    // *sigh*.. why gcc 4.9.. the compiler had better inline this crud..
    static       typename Tensor<Type>::Data&  data (      Tensor<Type>& a) { return Tensor<Type>::Accessor::data(a);  }
    static const typename Tensor<Type>::Data&  data (const Tensor<Type>& a) { return Tensor<Type>::Accessor::data(a);  }
    static       typename Tensor<Type>::Shape& shape(      Tensor<Type>& a) { return Tensor<Type>::Accessor::shape(a); }
    static const typename Tensor<Type>::Shape& shape(const Tensor<Type>& a) { return Tensor<Type>::Accessor::shape(a); }


    static void print(std::ostream& os, const Tensor<Type>& a)
    {
        for (std::size_t y = 0; y < shape(a)[1]; ++y)
        {
            for (std::size_t x = 0; x < shape(a)[0]; ++x)
            {
                os << a.at({x,y}) << " ";
            }
            os << "\n";
        }
    }

    static Tensor<Type> dot(const Tensor<Type>& a,
                            const Tensor<Type>& b)
    {
        // https://people.rit.edu/pnveme/EMEM851n/constitutive/tensors_rect.html

        // t1 = sum_y(sum_x( e_y e_x v_yx ))
        // t2 = sum_z(e_z v2_z)
        // t3 = t1 . t2
        //    = sum_y(sum_x( e_y e_x v_yx )) . sum_z(e_z v2_z)
        //    = sum_y(sum_x(sum_z( e_y e_x v_yx . e_z v2_z)))
        //    = sum_y(sum_x(sum_z( e_y dirac_xz v_yx v2_z)))
        //    = sum_y( e_y sum_j ( v_yj v2_j))

        // Note change of axis Y is now in dim 0

        int lenA = shape(a).size();
        int lenB = shape(b).size();

        if (shape(a)[lenA-1] != shape(b)[0])
        {
            std::stringstream ss;
            ss << "Tensor shapes wrong for dot"
               << " a: " << join(shape(a),"x")
               << " b: " << join(shape(b),"x");
            throw std::runtime_error(ss.str());
        }

        // the new shape is the start of the left (remove the last dim)
        // with the end of the right (remove the first dim)
        std::vector<int> newShape;
        std::copy(shape(a).begin()  , shape(a).end()-1, newShape.end());
        std::copy(shape(b).begin()+1, shape(b).end()  , newShape.end());

        // ACS todo.. the general tensor form please..
        Tensor<Type> res({shape(b)[0], shape(a)[1]});

        SkipIterator ait

        for (std::size_t y = 0; y < shape(res)[1]; ++y)
        {
            for (std::size_t x = 0; x < shape(res)[0]; ++x)
            {
                Type sum = 0;
                for (std::size_t i = 0; i < shape(a)[0]; ++i)
                {
                    sum += a.at({i,y}) * b.at({x,i});
                }
                res.at({x,y}) = sum;
            }
        }

        return res;
    }

    static Tensor<Type> selrow(std::size_t row,
                               const Tensor<Type>& a)
    {
        // TODO reimpl as an iterator!
        Tensor<Type> r({shape(a)[0],1});

        for (std::size_t x = 0; x < shape(a)[0]; ++x)
        {
            r.at({x,0}) = a.at({x,row});
        }

        return r;
    }

    static Tensor<Type> selcol(std::size_t col,
                               const Tensor<Type>& a)
    {
        Tensor<Type> r({1,shape(a)[1]});

        for (std::size_t y = 0; y < shape(a)[1]; ++y)
        {
            r.at({0,y}) = a.at({col,y});
        }

        return r;
    }

    static Tensor<Type> transpose(const Tensor<Type>& a)
    {
        Tensor<Type> r({shape(a)[1],shape(a)[0]});

        for (std::size_t y = 0; y < shape(a)[1]; ++y)
        {
            for (std::size_t x = 0; x < shape(a)[0]; ++x)
            {
                r.at({y,x}) = a.at({x,y});
            }
        }

        return r;
    }

    static void unifunctor_inplace(std::function<Type (Type)> func,
                                   Tensor<Type>& a)

    {
        // TODO this feels llike there should be an std:algo for it
        //  maybe generate ??

        for(iterator it = data(a).begin();
            it != data(a).end();
            ++it)
        {
            *it = func(*it);
        }
    }

    static Tensor<Type> unifunctor(std::function<Type (Type)> func,
                                   const Tensor<Type>& a)

    {
        // TODO this feels llike there should be an std:algo for it
        //  maybe generate ??

        Tensor<Type> r(shape(a));

        const_iterator ait = data(a).begin();
        for(iterator rit = data(r).begin();
            rit != data(r).end();
            ++rit)
        {
            *rit = func(*ait);
            ++ait;
        }

        return r;
    }

    static Tensor<Type> bifunctor(std::function<Type (Type,Type)> func,
                                  const Tensor<Type>& a,
                                  const Tensor<Type>& b)

    {
        if (shape(a)[0] != shape(b)[0] or
            shape(a)[1] != shape(b)[1])
        {
            std::stringstream ss;
            ss << "Tensor shapes mismatch for bifunctor"
               << " a: " << join(shape(a), "x")
               << " b: " << join(shape(b), "x");
            throw std::runtime_error(ss.str());
        }

        // TODO this feels llike there should be an std:algo for it
        //  maybe generate ??
        Tensor<Type> r(shape(a));

        const_iterator ait = data(a).begin();
        const_iterator bit = data(b).begin();
        iterator       rit = data(r).begin();

        while (rit != data(r).end())
        {
            *rit = func(*ait, *bit);
            ++ait;
            ++bit;
            ++rit;
        }

        return r;
    }

    static void bifunctor_inplace(std::function<Type (Type,Type)> func,
                                  Tensor<Type>& a,
                                  const Tensor<Type>& b)

    {
        if (shape(a) != shape(b))
        {
            std::stringstream ss;
            ss << "Tensor shapes mismatch for bifunctor"
               << " a: " << join(shape(a), "x")
               << " b: " << join(shape(b), "x");
            throw std::runtime_error(ss.str());
        }

        // TODO this feels llike there should be an std:algo for it
        //  maybe generate ??
        Tensor<Type> r(shape(a));

        const_iterator bit = data(b).begin();
        for(iterator ait = data(a).begin();
            ait != data(a).end();
            ++ait)
        {
            *ait = func(*ait, *bit);
            ++bit;
        }
    }

    static Tensor<Type> bifunctor_row(std::function<Type (Type,Type)> func,
                                      const Tensor<Type>& a,
                                      const Tensor<Type>& b)
    {
        if (shape(a)[0] != shape(b)[0] or
            shape(b)[1] != 1)
        {
            std::stringstream ss;
            ss << "Tensor shapes mismatch for bifunctor_row"
               << " a: " << join(shape(a), "x")
               << " b: " << join(shape(b), "x");
            throw std::runtime_error(ss.str());
        }

        // TODO this feels llike there should be an std:algo for it
        //  maybe generate ??
        Tensor<Type> r(shape(a));

        const_iterator ait = data(a).begin();
        const_iterator bit = data(b).begin();
        iterator       rit = data(r).begin();

        while (rit != data(r).end())
        {
            if (bit == data(b).end()) bit = data(b).begin();

            *rit = func(*ait, *bit);
            ++ait;
            ++bit;
            ++rit;
        }

        return r;
    }

    static Tensor<Type> bifunctor_scaler(std::function<Type (Type,Type)> func,
                                        const Type a,
                                        const Tensor<Type>& b)

    {
        // TODO this feels llike there should be an std:algo for it
        //  maybe generate ??
        Tensor<Type> r(shape(b)[0], shape(b)[1]);

        const_iterator bit = data(b).begin();
        iterator       rit = data(r).begin();

        while (rit != data(r).end())
        {
            *rit = func(a,*bit);
            ++bit;
            ++rit;
        }

        return r;
    }

    static Tensor<Type> bifunctor_scaler(std::function<Type (Type,Type)> func,
                                        const Tensor<Type>& a,
                                        const Type b)
    {
        // TODO this feels llike there should be an std:algo for it
        //  maybe generate ??
        Tensor<Type> r(shape(a)[0], shape(a)[1]);

        const_iterator ait = data(a).begin();
        iterator       rit = data(r).begin();

        while (rit != data(r).end())
        {
            *rit = func(*ait,b);
            ++ait;
            ++rit;
        }

        return r;
    }

    static bool bifunctor_compare_all(std::function<bool (Type,Type)> func,
                                      const Tensor<Type>& a,
                                      const Tensor<Type>& b)
    {
        if (shape(a) != shape(b))
        {
            return false;
        }

        const_iterator ait = data(a).begin();
        const_iterator bit = data(b).begin();

        bool r = true;
        while (ait != data(a).end())
        {
            r &= func(*ait,*bit);
            ++ait;
            ++bit;
        }

        return r;
    }
    struct Helpers
    {
        static bool equal(Type a, Type b) { return a==b; }

        static Type add(Type a, Type b) { return a+b; }
        static Type sub(Type a, Type b) { return a-b; }
        static Type mul(Type a, Type b) { return a*b; }
        static Type div(Type a, Type b) { return a/b; }

        static Type relu(Type a) { return (a < 0) ? 0 : a; }
        static Type tanh(Type a) { return std::tanh(a); }
        static Type rand(Type a) { return static_cast<Type>(std::rand() % 100)/100.0 - 0.5; }

        static Type ones(Type a) { return 1.0; }
        static Type xor_f(Type a, Type b) { return (a*b < 0) ? -1.0 : 1.0; }
    };
};

// ****************************************************************
// ************************ Tensor OPERATORS **********************
// ****************************************************************

template <typename Type>
std::ostream& operator<<(std::ostream& os, const Tensor<Type>& a)
{
    TensorUtils<Type>::print(os, a);
    return os;
}

template <typename Type>
Tensor<Type> operator+(const Tensor<Type>& a,
                       const Tensor<Type>& b)
{
    return TensorUtils<Type>::bifunctor(&TensorUtils<Type>::Helpers::add,a,b);
}

template <typename Type>
Tensor<Type> operator-(const Tensor<Type>& a,
                       const Tensor<Type>& b)
{
    return TensorUtils<Type>::bifunctor(&TensorUtils<Type>::Helpers::sub,a,b);
}

template <typename Type>
Tensor<Type> operator*(const Tensor<Type>& a,
                       const Tensor<Type>& b)
{
    return TensorUtils<Type>::dot(a,b);
}

template <typename Type>
Tensor<Type> operator*(Type                a,
                       const Tensor<Type>& b)
{
    return TensorUtils<Type>::bifunctor_scaler(&TensorUtils<Type>::Helpers::mul,a,b);
}

template <typename Type>
Tensor<Type> operator*(const Tensor<Type>& a,
                       Type                b)
{
    return TensorUtils<Type>::bifunctor_scaler(&TensorUtils<Type>::Helpers::mul,a,b);
}

template <typename Type>
Tensor<Type> operator/(const Tensor<Type>& a,
                       const Tensor<Type>& b)
{
    return TensorUtils<Type>::bifunctor(&TensorUtils<Type>::Helpers::div,a,b);
}

template <typename Type>
Tensor<Type> operator+=(Tensor<Type>&       a,
                        const Tensor<Type>& b)
{
    TensorUtils<Type>::bifunctor_inplace(&TensorUtils<Type>::Helpers::add,a,b);
    return a;
}

template <typename Type>
bool operator==(const Tensor<Type>& a,
                const Tensor<Type>& b)
{
    return TensorUtils<Type>::bifunctor_compare_all(&TensorUtils<Type>::Helpers::equal,a,b);
}

template <typename Type>
Tensor<Type> product(const Tensor<Type>& a,
                     const Tensor<Type>& b)
{
    return TensorUtils<Type>::bifunctor(&TensorUtils<Type>::Helpers::mul,a,b);
}

template <typename Type>
Tensor<Type> rowadd(const Tensor<Type>& a,
                    const Tensor<Type>& b)
{
    return TensorUtils<Type>::bifunctor_row(&TensorUtils<Type>::Helpers::add,a,b);
}

template <typename Type>
Tensor<Type> transpose(const Tensor<Type>& a)
{
    return TensorUtils<Type>::transpose(a);
}

template <typename Type>
Tensor<Type> tanh(const Tensor<Type>& a)
{
    return TensorUtils<Type>::unifunctor(&TensorUtils<Type>::Helpers::tanh,a);
}

template <typename Type>
Tensor<Type> tanh_derivate(const Tensor<Type>& a)
{
    // dtanh/dx = 1 - (tanh(x)) ^ 2
    Tensor<Type> th = tanh(a);

    return TensorUtils<Type>::bifunctor_scaler(&TensorUtils<Type>::Helpers::sub, 1, product(th,th));
}

template <typename Type>
void rand(Tensor<Type>& a)
{
    TensorUtils<Type>::unifunctor_inplace(&TensorUtils<Type>::Helpers::rand,a);
}

template <typename Type>
Tensor<Type> pow(const Tensor<Type>& a, int power)
{
    return TensorUtils<Type>::unifunctor([power](Type a) { return std::pow(a,power); },
                                         a);
}

template <typename Type>
void ones(Tensor<Type>& a)
{
    TensorUtils<Type>::unifunctor_inplace(&TensorUtils<Type>::Helpers::ones,a);
}


#endif
