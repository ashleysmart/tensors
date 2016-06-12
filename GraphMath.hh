#include "Tensor.hh"

#include <tuple>
#include <cmath>
#include <memory>
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
    int    srcId_;
    double value_;

    VarDerivedConst(int srcID,
                double value) :
        srcId_(srcID),
        value_(value)
    {}

    void render(std::ostream& os) const
    {
        os << "{" << value_ << "}";
    }

    Tensor<double> op(const Tensors& params ) const
    {
        const Tensor<double>& src = params[srcId_];
        return unifunc(src, [value_](double ){ return value_; });
    }

    std::shared_ptr<Node> derive(const std::shared_ptr<Node>& denum)
    {
        return std::shared_ptr<Node>(new ScalarConst(0));
    }
};

template <int ID>
struct Var : Node
{
    std::string label_;

    Var(std::string label) :
        label_(label)
    {}

    void render(std::ostream& os) const
    {
        os << label_;
    }

    Tensor<double> op(const Tensors& params ) const
    {
        return params[ID];
    }

    std::shared_ptr<Node> derive(const std::shared_ptr<Node>& denum)
    {
        if (*denum == *this)
            return std::shared_ptr<Node>(new VarDerivedConst(1,ID));
        return std::shared_ptr<Node>(new VarDerivedConst(0,ID));
    }

    virtual bool equals(const Node& other)
    {
        if (const Var<ID>* ptr = dynamic_cast<const Var<ID>*>(&other))
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
    template <int ID>
    static std::shared_ptr<Node> var(const std::string& ident)
    {
        std::shared_ptr<Node> node(new Var<ID>(ident));
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
