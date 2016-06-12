#include "GraphMath.hh"

void graphTest()
{
    std::shared_ptr<Node> x = Make::var<0>("x");
    std::shared_ptr<Node> m = Make::var<1>("m");
    std::shared_ptr<Node> c = Make::var<2>("c");

    std::shared_ptr<Node> s = Make::add(c,c);                             // s = c + c
    std::shared_ptr<Node> p = Make::add(x,c);                             // p = x + c
    std::shared_ptr<Node> r = Make::dot(x,m);                             // r = m*x + c
    std::shared_ptr<Node> y = Make::add(Make::dot(x,m),c);                // y = m*x + c
    std::shared_ptr<Node> z = Make::add(Make::dot(Make::power(x,2),m),c); // z = m*x^2 + c
    std::shared_ptr<Node> q = Make::power(Make::add(Make::dot(x,m),c),2); // q = = y^2 = (m*x + c)^2

    std::shared_ptr<Node> dC_dX = Make::derivativeOf(c, x);
    std::shared_ptr<Node> dS_dX = Make::derivativeOf(s, x);
    std::shared_ptr<Node> dX_dX = Make::derivativeOf(x, x);
    std::shared_ptr<Node> dP_dX = Make::derivativeOf(p, x);
    std::shared_ptr<Node> dR_dX = Make::derivativeOf(r, x);
    std::shared_ptr<Node> dY_dX = Make::derivativeOf(y, x);
    std::shared_ptr<Node> dZ_dX = Make::derivativeOf(z, x);
    std::shared_ptr<Node> dQ_dX = Make::derivativeOf(q, x);

    Tensor<double> tx({2}, {1,2});
    Tensor<double> tm({2,2}, {1,2,3,4});
    Tensor<double> tc({2}  , {5,6}    );

    Tensors params;
    params.push_back(tx);
    params.push_back(tm);
    params.push_back(tc);

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

    EXPECT_EQ(Tensor<double>({2},  {0,0}),     dC_dX->op(params));
    EXPECT_EQ(Tensor<double>({2},  {0,0}),     dS_dX->op(params));
    EXPECT_EQ(Tensor<double>({2},  {1,1}),     dX_dX->op(params));
    EXPECT_EQ(Tensor<double>({2},  {1,1}),     dP_dX->op(params));
    EXPECT_EQ(Tensor<double>({2,2},{1,2,3,4}), dR_dX->op(params));
    EXPECT_EQ(Tensor<double>({2,2},{1,2,3,4}), dY_dX->op(params));
    EXPECT_EQ(Tensor<double>({2},{ 10,  12}),  dZ_dX->op(params));
    EXPECT_EQ(Tensor<double>({2},{  6,   8}),  dQ_dX->op(params));


    std::cout << "\n dc/dx:" << dC_dX->renderOps();
    std::cout << "\n dc/dx:" << dS_dX->renderOps();
    std::cout << "\n dx/dx:" << dX_dX->renderOps();
    std::cout << "\n dp/dx:" << dP_dX->renderOps();
    std::cout << "\n dr/dx:" << dR_dX->renderOps();
    std::cout << "\n dy/dx:" << dY_dX->renderOps();
    std::cout << "\n dz/dx:" << dZ_dX->renderOps();
    std::cout << "\n dq/dx:" << dQ_dX->renderOps();
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
    std::cout << "\n dz/dx:" << dZ_dX->renderOps();
    std::cout << "\n"        << dZ_dX->op(params);
    std::cout << "\n dq/dx:" << dQ_dX->renderOps();
    std::cout << "\n"        << dQ_dX->op(params);
    std::cout << "\n";
}

int main()
{
    graphTest();
}
