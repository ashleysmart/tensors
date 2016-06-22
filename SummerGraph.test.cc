#include "SummerGraph.hh"

#include "test.hh"

void testExpressions()
{
    Handle uvec(new Var("a"));
    Handle uvec2(new Var("b"));

    EXPECT_STREAMED_AS("a") << uvec;

    Handle mult(new Mult(uvec, uvec2));

    EXPECT_STREAMED_AS("a*b") << mult;
}

int main()
{
    testExpressions();

    Handle m = Make::tensor("m",
                            {"i","j"});

    EXPECT_STREAMED_AS("sum_i(sum_j(U_i*U_j*m_ij))") << m;

    Handle x = Make::tensor("x",
                            {"k"});

    EXPECT_STREAMED_AS("sum_k(U_k*x_k)") << x;

    Handle l = Make::dot(x,m);
    EXPECT_STREAMED_AS("sum_k(U_k*x_k).sum_i(sum_j(U_i*U_j*m_ij))") << l;

    std::cout << "\n begin transforms....\n";

    // search tree and move summers to front
    // SumLifter lift;
    TransformAll<LiftSum> lift;
    l = lift.process(l);
    EXPECT_STREAMED_AS("sum_k(sum_i(sum_j(U_k*x_k.U_i*U_j*m_ij)))") << l;

    LocateLastSum lls;
    Handle lastSum = lls.find(l);
    EXPECT_STREAMED_AS("sum_j(U_k*x_k.U_i*U_j*m_ij)") << lastSum;

    // move mult/dot sequances to right side orientation
    TransformAll<RotateDotsMultsToRight> allDotsRotate;
    l = allDotsRotate.process(l);
    // note following looks iffy becase () are not printed
    EXPECT_STREAMED_AS("sum_k(sum_i(sum_j(U_k*x_k.U_i*U_j*m_ij)))") << l;

    // mode dots towards applicable unit vetors
    TransformAll<AttachDotsToUnitVectors> moveDotsToVectors;
    l = moveDotsToVectors.process(l);
    EXPECT_STREAMED_AS("sum_k(sum_i(sum_j(U_k.x_k*U_i*U_j*m_ij)))") << l;

    TransformAll<LiftUnitVectorUp> liftUnitVecUp;
    l = liftUnitVecUp.process(l);
    EXPECT_STREAMED_AS("sum_k(sum_i(sum_j(U_k.U_i*U_j*x_k*m_ij)))") << l;

}
