#include "GraphMath.hh"

void graphTest()
{
    std::shared_ptr<Node> x = Make::var(0, "x");
    std::shared_ptr<Node> m = Make::var(1, "m");
    std::shared_ptr<Node> c = Make::var(2, "c");

    std::shared_ptr<Node> s = Make::add(c,c);                             // s = c + c
    std::shared_ptr<Node> p = Make::add(x,c);                             // p = x + c
    std::shared_ptr<Node> r = Make::dot(x,m);                             // r = m*x + c
    std::shared_ptr<Node> y = Make::add(Make::dot(x,m),c);                // y = m*x + c
    std::shared_ptr<Node> l = Make::power(x,2);                           // l = x^2
    std::shared_ptr<Node> z = Make::add(Make::dot(Make::power(x,2),m),c); // z = m*x^2 + c
    std::shared_ptr<Node> q = Make::power(Make::add(Make::dot(x,m),c),2); // q = = y^2 = (m*x + c)^2

    std::shared_ptr<Node> dC_dX = Make::derivativeOf(c, x);
    std::shared_ptr<Node> dS_dX = Make::derivativeOf(s, x);
    std::shared_ptr<Node> dX_dX = Make::derivativeOf(x, x);
    std::shared_ptr<Node> dP_dX = Make::derivativeOf(p, x);
    std::shared_ptr<Node> dR_dX = Make::derivativeOf(r, x);
    std::shared_ptr<Node> dY_dX = Make::derivativeOf(y, x);
    std::shared_ptr<Node> dL_dX = Make::derivativeOf(l, x);
    std::shared_ptr<Node> dZ_dX = Make::derivativeOf(z, x);
    // std::shared_ptr<Node> dQ_dX = Make::derivativeOf(q, x);

    Tensor<double> tx({2}, {1,2});
    Tensor<double> tm({2,2}, {1,2,3,4});
    Tensor<double> tc({2}  , {5,6}    );

    Tensors params;
    params.push_back(tx);
    params.push_back(tm);
    params.push_back(tc);

    EXPECT_EQ(tx,x->op(params));
    EXPECT_EQ(tm,m->op(params));
    EXPECT_EQ(tc,c->op(params));

    EXPECT_EQ(Tensor<double>({2},{ 10,  12}),s->op(params));
    EXPECT_EQ(Tensor<double>({2},{  6,   8}),p->op(params));
    EXPECT_EQ(Tensor<double>({2},{  7,  10}),r->op(params));
    EXPECT_EQ(Tensor<double>({2},{ 12,  16}),y->op(params));
    EXPECT_EQ(Tensor<double>({2},{ 18,  24}),z->op(params));
    EXPECT_EQ(Tensor<double>({2},{144, 256}),q->op(params));

    std::cout << "\n x=" << x->renderOps();
    std::cout << "\n"    << x->op(params);
    std::cout << "\n m=" << m->renderOps();
    std::cout << "\n"    << m->op(params);
    std::cout << "\n c=" << c->renderOps();
    std::cout << "\n"    << c->op(params);
    std::cout << "\n";

    std::cout << "\n s=" << s->renderOps();
    std::cout << "\n"    << s->op(params);
    std::cout << "\n p=" << p->renderOps();
    std::cout << "\n"    << p->op(params);
    std::cout << "\n r=" << r->renderOps();
    std::cout << "\n"    << r->op(params);
    std::cout << "\n y=" << y->renderOps();
    std::cout << "\n"    << y->op(params);
    std::cout << "\n z=" << z->renderOps();
    std::cout << "\n"    << z->op(params);
    std::cout << "\n q=" << q->renderOps();
    std::cout << "\n"    << q->op(params);
    std::cout << "\n";

    // p = x + c
    // dp1/dx1 = 1 dp2/dx1 = 0
    // dp1/dx2 = 0 dp2/dx2 = 1
    //
    //Hence
    // dp/dx = [ 1 0 ]
    //         [ 0 1 ]

    // dp/dx = sum_i(U_i d/dx_i) sum_j(U_j (x_j + c_j))
    //       = sum_i(sum_j( U_i U_j d/dx_i(x_j + c_j)))
    //     Given
    //       d/dx_i(x_j + c_j) = 0 when i != j
    //       d/dx_i(x_j + c_j) = 1 when i == j
    //
    // if i==j
    // dp/dx = sum_i(sum_i( U_i U_i d/dx_i(x_i + c_i)))
    //       = sum_i( U_i U_i 1 )
    //       = sum_i( U_i U_i 1 )
    // if i!=j
    // dp/dx = sum_i(sum_j( U_i U_j d/dx_i(x_j + c_j)))
    //       = sum_i(sum_j( U_i U_j 0))
    //       = 0
    //
    //Hence
    // dp/dx = [ 1 0 ]
    //         [ 0 1 ]

    // EXPECT_EQ(Tensor<double>({4},  {0,0,0,0}),   dC_dX->op(params));
    // EXPECT_EQ(Tensor<double>({4},  {0,0,0,0}),   dS_dX->op(params));
    // EXPECT_EQ(Tensor<double>({4},  {1,0,0,1}),   dX_dX->op(params));
    // EXPECT_EQ(Tensor<double>({4},  {1,0,0,1}),   dP_dX->op(params));
    // EXPECT_EQ(Tensor<double>({2,2},{1,2,3,4}),   dR_dX->op(params));
    // EXPECT_EQ(Tensor<double>({2,2},{1,2,3,4}),   dY_dX->op(params));
    // EXPECT_EQ(Tensor<double>({2},  { 10,  12}),  dZ_dX->op(params));
    // EXPECT_EQ(Tensor<double>({2},  {  6,   8}),  dQ_dX->op(params));

    std::cout << "\n dc/dx:" << dC_dX->renderOps();
    std::cout << "\n dc/dx:" << dS_dX->renderOps();
    std::cout << "\n dx/dx:" << dX_dX->renderOps();
    std::cout << "\n dp/dx:" << dP_dX->renderOps();
    std::cout << "\n dr/dx:" << dR_dX->renderOps();
    std::cout << "\n dy/dx:" << dY_dX->renderOps();
    std::cout << "\n dl/dx:" << dL_dX->renderOps();
    std::cout << "\n dz/dx:" << dZ_dX->renderOps();
    //std::cout << "\n dq/dx:" << dQ_dX->renderOps();
    std::cout << "\n";

    std::cout << "\n dc/dx:" << dC_dX->renderOps();
    std::cout << "\n"        << dC_dX->op(params);
    std::cout << "\n dc/dx:" << dS_dX->renderOps();
    std::cout << "\n"        << dS_dX->op(params);
    std::cout << "\n dx/dx:" << dX_dX->renderOps();
    std::cout << "\n"        << dX_dX->op(params);
    std::cout << "\n dp/dx:" << dP_dX->renderOps();
    std::cout << "\n"        << dP_dX->op(params);
    std::cout << "\n dr/dx:" << dR_dX->renderOps();
    std::cout << "\n"        << dR_dX->op(params);
    std::cout << "\n dy/dx:" << dY_dX->renderOps();
    std::cout << "\n"        << dY_dX->op(params);
    std::cout << "\n dl/dx:" << dL_dX->renderOps();
    std::cout << "\n"        << dL_dX->op(params);
    std::cout << "\n dz/dx:" << dZ_dX->renderOps();
    std::cout << "\n"        << dZ_dX->op(params);
    // std::cout << "\n dq/dx:" << dQ_dX->renderOps();
    // std::cout << "\n"        << dQ_dX->op(params);
    std::cout << "\n";
}

int main()
{
    try
    {
        graphTest();
    }
    catch(std::exception& e)
    {
        std::cout << "Opps.." << e.what() << "\n";
    }
}
