#include "Tensor.hh"

#include <tuple>
#include <cmath>
#include <memory>
#include <stdexcept>

#include "test.hh"

typedef std::vector<Tensor<double> > Tensors;

struct Node
{
    virtual ~Node() {}

    virtual void render(std::ostream& os) const = 0;
    virtual Tensor<double> op(const Tensors& params) const = 0;

    virtual std::shared_ptr<Node> derive(const std::shared_ptr<Node>& denum) = 0;

    virtual bool equals(const Node& other)
    {
        return false;
    }

    bool operator==(const Node& other)
    {
        return equals(other);
    }

    std::string renderOps() const
    {
        std::stringstream ss;
        render(ss);
        return ss.str();
    }
};

struct ScalarConst : Node
{
    double value_;

    ScalarConst(double value) :
        value_(value)
    {}

    void render(std::ostream& os) const
    {
        os << value_;
    }

    Tensor<double> op(const Tensors& params ) const
    {
        return Tensor<double>({1}, {value_});
    }

    std::shared_ptr<Node> derive(const std::shared_ptr<Node>& denum)
    {
        return std::shared_ptr<Node>(new ScalarConst(0));
    }

};

struct Const : Node
{
    Tensor<double> value_;

    Const(Tensor<double> value) :
        value_(value)
    {}

    void render(std::ostream& os) const
    {
        os << value_;
    }

    Tensor<double> op(const Tensors& params ) const
    {
        return value_;
    }

    std::shared_ptr<Node> derive(const std::shared_ptr<Node>& denum)
    {
        return std::shared_ptr<Node>(new ScalarConst(0));
    }
};

struct VarDerivedConst : Node
{
    double value_;
    int    outerID_;
    int    innerID_;

    VarDerivedConst(double value,
                    int outerID,
                    int innerID) :
        value_(value),
        outerID_(outerID),
        innerID_(innerID)
    {}

    void render(std::ostream& os) const
    {
        os << "{" << value_ << ":" << outerID_ << "x" << innerID_ << "}";
    }

    Tensor<double> op(const Tensors& params ) const
    {
        //*sigh* because you cant pass a *member* to lamdbas, you need to pass this which is not *const* correct
        double val = value_;
        const Tensor<double>& outer = params[outerID_];
        const Tensor<double>& inner = params[innerID_];

        Tensor<double>::Shape rShape;
        std::copy(TensorUtils<double>::shape(outer).begin(), TensorUtils<double>::shape(outer).end(), std::back_inserter(rShape));
        std::copy(TensorUtils<double>::shape(inner).begin(), TensorUtils<double>::shape(inner).end(), std::back_inserter(rShape));

        // this is inefficent.. the constructor news memory... then the unifunc allocates a second
        Tensor<double> r(rShape);
        return unifunc<double>(r, [val](double){ return val; });
    }

    std::shared_ptr<Node> derive(const std::shared_ptr<Node>& denum)
    {
        return std::shared_ptr<Node>(new ScalarConst(0));
    }
};

struct Var : Node
{
    int    srcID_;
    std::string label_;

    Var(int srcID,
        std::string label) :
        srcID_(srcID),
        label_(label)
    {}

    void render(std::ostream& os) const
    {
        os << label_;
    }

    Tensor<double> op(const Tensors& params ) const
    {
        return params[srcID_];
    }

    std::shared_ptr<Node> derive(const std::shared_ptr<Node>& denum)
    {
        // incorrect the tensor maths shows this as
        // dc_abc../dx_ijk.. = sum_i(sum_j(....( U_i U_j U_k ... d/dx_ijk... ))) sum_a(sum_b(....( U_a U_b U_c ... c_abc... )))
        //                   = sum_i(sum_j(....( sum_a(sum_b(....( U_i U_j U_k... U_a U_b U_c... d/dx_ijk...(c_abc... ) ))) ... )))
        //                   = sum_i(sum_j(....( sum_a(sum_b(....( U_i U_j U_k... U_a U_b U_c... d/dx_ijk...(c_abc... ) ))) ... )))
        //                     |--------------- sums ------------|----- dimensional vector -----|----- operation -----|--- sums --|
        int denumID = -1;

        if (const Var* ptr = dynamic_cast<Var*>(denum.get()))
            denumID = ptr->srcID_;
        else
            throw std::runtime_error("derivate with non-var base not supported yet... mosty cause i have no idea how to get the shape..");

        if (*denum == *this)
            return std::shared_ptr<Node>(new VarDerivedConst(1, denumID, srcID_));
        return std::shared_ptr<Node>(new VarDerivedConst(0, denumID, srcID_));
    }

    virtual bool equals(const Node& other)
    {
        if (const Var* ptr = dynamic_cast<const Var*>(&other))
            if (ptr->srcID_ == srcID_)
                return true;
        return false;
    }
};


struct OpAdd : Node
{
    std::shared_ptr<Node> left_;
    std::shared_ptr<Node> right_;

    OpAdd(const std::shared_ptr<Node>& left,
          const std::shared_ptr<Node>& right) :
        left_(left),
        right_(right)
    {}

    void render(std::ostream& os) const
    {
        os << "(";
        left_->render(os);
        os << "+";
        right_->render(os);
        os << ")";
    }

    Tensor<double> op(const Tensors& params) const
    {
        return left_->op(params) + right_->op(params);
    }

    std::shared_ptr<Node> derive(const std::shared_ptr<Node>& denum)
    {
        return std::shared_ptr<Node>(new OpAdd(left_->derive(denum),
                                               right_->derive(denum)));
    }
};

struct OpDot : Node
{
    std::shared_ptr<Node> left_;
    std::shared_ptr<Node> right_;

    OpDot(const std::shared_ptr<Node>& left,
          const std::shared_ptr<Node>& right) :
        left_(left),
        right_(right)
    {}

    void render(std::ostream& os) const
    {
        os << "(";
        left_->render(os);
        os << ".";
        right_->render(os);
        os << ")";
    }

    Tensor<double> op(const Tensors& params) const
    {
        return left_->op(params) * right_->op(params);
    }

    std::shared_ptr<Node> derive(const std::shared_ptr<Node>& denum)
    {
        std::shared_ptr<Node> leftDeriv      =
            std::shared_ptr<Node>(new OpDot(left_->derive(denum),
                                            right_));
        std::shared_ptr<Node> rightDeriv =
            std::shared_ptr<Node>(new OpDot(left_,
                                            right_->derive(denum)));

        return std::shared_ptr<Node>(new OpAdd(leftDeriv,
                                               rightDeriv));

    }
};

struct OpScalarMult : Node
{
    std::shared_ptr<Node> left_;
    std::shared_ptr<Node> right_;

    OpScalarMult(const std::shared_ptr<Node>& left,
                 const std::shared_ptr<Node>& right) :
        left_(left),
        right_(right)
    {}

    void render(std::ostream& os) const
    {
        os << "(";
        left_->render(os);
        os << "*";
        right_->render(os);
        os << ")";
    }

    Tensor<double> op(const Tensors& params) const
    {
        return left_->op(params)[0] * right_->op(params);
    }

    std::shared_ptr<Node> derive(const std::shared_ptr<Node>& denum)
    {
        std::shared_ptr<Node> leftDeriv      =
            std::shared_ptr<Node>(new OpScalarMult(left_->derive(denum),
                                                   right_));
        std::shared_ptr<Node> rightDeriv =
            std::shared_ptr<Node>(new OpScalarMult(left_,
                                                   right_->derive(denum)));

        return std::shared_ptr<Node>(new OpAdd(leftDeriv,
                                               rightDeriv));

    }
};

struct OpPower : Node
{
    std::shared_ptr<Node> left_;
    double pow_;

    OpPower(const std::shared_ptr<Node>& left,
            double power) :
        left_(left),
        pow_(power)
    {}

    void render(std::ostream& os) const
    {
        os << "(";
        left_->render(os);
        os << ")^" << pow_;
    }

    Tensor<double> op(const Tensors& params) const
    {
        return pow(left_->op(params), pow_);
    }

    std::shared_ptr<Node> derive(const std::shared_ptr<Node>& denum)
    {
        std::shared_ptr<Node> pwrDown =
            std::shared_ptr<Node>(new OpPower(left_,
                                              pow_));

        std::shared_ptr<Node> outerDeriv =
            std::shared_ptr<Node>(new OpScalarMult(pwrDown,
                                                   left_));

        return std::shared_ptr<Node>(new OpDot(outerDeriv,
                                               left_->derive(denum)));
    }
};

// just simplify the typeing
struct Make
{
    static std::shared_ptr<Node> var(const int srcID,
                                     const std::string& ident)
    {
        std::shared_ptr<Node> node(new Var(srcID, ident));
        return node;
    }

    static std::shared_ptr<Node> add(const std::shared_ptr<Node>& left,
                                     const std::shared_ptr<Node>& right)
    {
        std::shared_ptr<Node> node(new OpAdd(left,right));
        return node;
    }

    static std::shared_ptr<Node> dot(const std::shared_ptr<Node>& left,
                                     const std::shared_ptr<Node>& right)
    {
        std::shared_ptr<Node> node(new OpDot(left,right));
        return node;
    }

    static std::shared_ptr<Node> scalarMult(const std::shared_ptr<Node>& left,
                                            const std::shared_ptr<Node>& right)
    {
        std::shared_ptr<Node> node(new OpScalarMult(left,right));
        return node;
    }

    static std::shared_ptr<Node> power(const std::shared_ptr<Node>& left,
                                       double power)
    {
        std::shared_ptr<Node> node(new OpPower(left,power));
        return node;
    }

    static std::shared_ptr<Node> derivativeOf(const std::shared_ptr<Node>& num,
                                              const std::shared_ptr<Node>& denum)
    {
        return num->derive(denum);
    }
};
