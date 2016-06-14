#include "Tensor.hh"

#include "test.hh"

void basicTest()
{

    Tensor<int> a({2, 2},
                  {1,2,
                   3,4});

    Tensor<int> b({2, 2},
                  {2,3,
                   4,5});
    Tensor<int> c({3, 2},
                  {2,3,
                   4,5,
                   6,7});
    Tensor<int> d({4,2,3},
                  {0,1,2,    3,4,5,
                   6,7,8,    9,10,11,
                   12,13,14, 15,16,17,
                   18,19,20, 21,22,23});

    std::cout << a << "\n";
    std::cout << b << "\n";
    std::cout << c << "\n";
    std::cout << d << "\n";

    Tensor<int> expAB({2,2},
                      {3,5,
                       7,9});
    EXPECT_EQ(expAB, (a+b));
    EXPECT_THROW((a + c), "Tensor shapes mismatch for bifunctor a: 2x2x b: 3x2x");

    Tensor<int> expCxA({3,2},{11,16,19,28,27,40});
    EXPECT_EQ(expCxA, TensorUtils<int>::dot(c,a));
    EXPECT_THROW(TensorUtils<int>::dot(a,c), "Tensor shapes wrong for dot a: 2x2x b: 3x2x");

    Tensor<int> expDxC({4,2,2},
                       {16,19,    52,64,
                        88,109,   124,154,
                        160,199,  196,244,
                        232,289,  268,334});
    EXPECT_EQ(expDxC, TensorUtils<int>::dot(d,c));
}

int main()
{
    try
    {
        basicTest();
    }
    catch (std::exception& e)
    {
        std::cout << "Opps... " << e.what() << "\n";
    }
}
