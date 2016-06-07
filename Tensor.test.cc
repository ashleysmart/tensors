#include "Tensor.hh"

#include <iostream>

#define EXPECT_EQ(A,B)                                                  \
    {                                                                   \
    if (not ((A) == (B)))                                               \
        std::cout << "FAILED:" << #A                                    \
                  << " (" << A << ") != "                               \
                  << #B                                                 \
                  << " (" << B << ")\n";                                \
}

#define EXPECT_THROW(A, WHAT)                                           \
    {                                                                   \
        try                                                             \
        {                                                               \
            A;                                                          \
            std::cout << "FAILED: " << #A << " didnt throw!\n";         \
        }                                                               \
        catch (std::exception& e)                                       \
        {                                                               \
            if (e.what() != std::string(WHAT))                          \
                std::cout << "FAILED: throw was: \"" << e.what()        \
                          << "\" expecting: \"" << WHAT                 \
                          << "\"\n";                                    \
        }                                                               \
    }

void basicTest()
{
    Tensor<int> a({1,2,3,4;    , {2, 2});
    Tensor<int> b({2,3,4,5}    , {2, 2});
    Tensor<int> c({2,3,4,5,6,7}, {3, 2});

    std::cout << a << "\n";
    std::cout << b << "\n";
    std::cout << c << "\n";

    Tensor<int> expAB({3,5,7,9}, {2, 2});
    EXPECT_EQ(expAB, (a+b));
    EXPECT_THROW((a + c), "Tensor shapes mismatch for bifunctor a: 2x2x b: 3x2x");

    Tensor<int> expAxB({12,15,18,26,33,40}, {3, 2});
    EXPECT_EQ(expAxB, TensorUtils<int>::dot(a,c));
    EXPECT_THROW(TensorUtils<int>::dot(c,a), "Tensor shapes wrong for dot product a: 3x2x b: 2x2x");
}

int main()
{
    basicTest();
}
