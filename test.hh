#include <iostream>
#include <sstream>

#define EXPECT_EQ(A,B)                                                  \
{                                                                       \
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

// yes yes.. bad idea to test againest printed format...
struct StreamedCheck
{
    bool              verbose_;
    std::string       what_;
    std::stringstream ss_;

    StreamedCheck(std::string what, bool verbose) :
        verbose_(verbose),
        what_(what)
    {}

    ~StreamedCheck()
    {
         if (ss_.str() != what_)
            std::cout << "FAILED:"
                      << "expected (" << what_ << ") != ("
                      << ss_.str()
                      << ")\n";
         else
             std::cout << " " << ss_.str() << "\n";


    }

    template <typename T>
    StreamedCheck& operator<<(const T& t)
    {
        ss_ << t;
    }
};

#define EXPECT_STREAMED_AS(WHAT)        \
    StreamedCheck(WHAT, true)
