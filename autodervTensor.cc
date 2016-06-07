#include <tuple>
#include <iostream>
#include <cmath>

#include "Tensor.hh"

template <int VALUE>
struct Const
{
    template< typename... Types >
    static Tensor<double> op(std::tuple<Types...>& params )
    {
        return Tensor<double>({VALUE}, {1});
    }
};

template <int ID>
struct Var
{
    template< typename... Types >
    static Tensor<double> op(std::tuple<Types...>& params )
    {
        return std::get<ID>(params);
    }
};

template <typename L, typename R>
struct OpAdd
{
    template< typename... Types >
    static Tensor<double> op(std::tuple<Types...>& params)
    {
        return L::op(params) + R::op(params);
    }
};

template <typename L, typename R>
struct OpMult
{
    template< typename... Types >
    static Tensor<double> op(std::tuple<Types...>& params)
    {
        return L::op(params) * R::op(params);
    }
};

template <typename L, int POW>
struct OpPower
{
    template< typename... Types >
    static Tensor<double> op(std::tuple<Types...>& params)
    {
        return pow(L::op(params), POW);
    }
};

// ###############################################
// ################# DERIVATES ###################
// ###############################################

template <typename Num, typename Denum>
struct DerivativeOf {};

template <int V, int ID_D>
struct DerivativeOf<Const<V>, Var<ID_D> >
{
    typedef Const<0> Type;
};

template <int ID_N, int ID_D>
struct DerivativeOf<Var<ID_N>, Var<ID_D> >
{
    typedef Const<0> Type;
};

template <int ID_N>
struct DerivativeOf<Var<ID_N>, Var<ID_N> >
{
    typedef Const<1> Type;
};

template <typename L, typename R, typename D>
struct DerivativeOf<OpAdd<L,R>, D >
{
    typedef OpAdd<typename DerivativeOf<L,D>::Type,
                  typename DerivativeOf<R,D>::Type> Type;
};

template <typename L, typename R, typename D>
struct DerivativeOf<OpMult<L,R>, D >
{
    typedef OpAdd<OpMult<typename DerivativeOf<L, D>::Type, R>,
                  OpMult<L, typename DerivativeOf<R, D>::Type> > Type;
};

template <typename L, int POW>
struct DerivativeOf<OpPower<L,POW>, L >
{
    typedef OpMult<Const<POW>, OpPower<L, POW-1> > Type;
};

template <typename L, int POW, typename D>
struct DerivativeOf<OpPower<L,POW>, D >
{
    typedef OpMult<OpMult<Const<POW>, OpPower<L, POW-1> >,
                   typename DerivativeOf<L, D>::Type>  Type;
};

int main()
{
    typedef Var<0> X;
    typedef Var<1> M;
    typedef Var<2> C;
    typedef OpAdd<X,C> P;                      // p = x + c
    typedef OpAdd<OpMult<M,X>,C> Y;            // y = m*x + c
    typedef OpAdd<OpMult<M,OpPower<X,2>>,C> Z; // z = m*x^2 + c
    typedef OpPower<OpAdd<OpMult<M,X>,C>,2> Q; // q = = y^2 = (m*x + c)^2

    typedef DerivativeOf<M, X>::Type  dM_dX;   // dm/dx = 0
    typedef DerivativeOf<X, X>::Type  dX_dX;   // dx/dx = 1
    typedef DerivativeOf<P, X>::Type  dP_dX;   // dP/dx = 1
    typedef DerivativeOf<Y, X>::Type  dY_dX;   // dY/Dx = m
    typedef DerivativeOf<Z, X>::Type  dZ_dX;   // dZ/dx = 2x
    typedef DerivativeOf<Q, X>::Type  dQ_dX;   // dQ/dx = dq/dy + dy/dx = 2*(m*x + c)*m

    X x;
    M m;
    C c;
    P p;
    Y y;
    Z z;
    Q q;

    dX_dX  dx_dx;
    dM_dX  dm_dx;
    dP_dX  dp_dx;
    dY_dX  dy_dx;
    dZ_dX  dz_dx;
    dQ_dX  dq_dx;

    Tensor<double> tx({1,2},     {2});
    Tensor<double> tm({1,2,3,4}, {2,2});
    Tensor<double> tc({5,6},     {2});
    
    std::tuple<Tensor<double>&,
               Tensor<double>&,
               Tensor<double>&> params = std::tie(tx,tm,tc);

    std::cout << " x:" << x.op(params)
              << " m:" << m.op(params)
              << " c:" << c.op(params)
              << " p:" << p.op(params)
              << " y:" << y.op(params)
              << " z:" << z.op(params)
              << " q:" << q.op(params)
              << "\n";

    std::cout << " dm/dx:"  << dm_dx.op(params)
              << " dx/dx:"  << dx_dx.op(params)
              << " dp/dx:"  << dp_dx.op(params)
              << " dy/dx:"  << dy_dx.op(params)
              << " dz/dx:"  << dz_dx.op(params)
              << " dq/dx:"  << dq_dx.op(params)
              << "\n";
}
