#include <iostream>
#include <memory>
#include <vector>

// TODO
//  - complete optimisation of graph
//  - generate execution method that can run actual computations using the resultng graph
//  - compute derivative graph and run
//  - add Tensor memory pool system to reduce over head of "new" without needing to inpace too much
//  - determine worth of the design
//  - stop hacking and convert to production level code.. (more than half this stuff shouldnt be in the .hh)
//  - add secondary execution method as thrush implementation and check cuda operating speed

// design notes
// - at the moment i am delibratly keeping the tree inbalanced instead of forming redux trees.. makes some of the code easier and this is presently proof of concept investigation
// - i could potentally grow the Mult Add and other binary ops to > 2 children nodes then just for loop over the kids..

// ################################################
// ################################################
// ################################################

typedef std::vector<std::size_t> TensorShape;

class AbstractDispatcher;

struct Node
{
    typedef std::shared_ptr<Node> Handle;
    typedef std::vector<Handle>   Nodes;

    Nodes children_;

    Node() :
        children_()
    {}

    Node(const std::initializer_list<Handle> kids) :
        children_(kids)
    {}

    virtual void visit(AbstractDispatcher& dispatcher) = 0;
};

typedef Node::Handle Handle;
typedef Node::Nodes  Nodes;

struct Var;
struct UnitVec;
struct Element;
struct Summer;
struct Mult;
struct Dot;

class AbstractDispatcher
{
    // typeify the node and recall to the type specifc handler
public:
    virtual void handle(Var&      node) = 0;
    virtual void handle(UnitVec&  node) = 0;
    virtual void handle(Element&  node) = 0;
    virtual void handle(Summer&   node) = 0;
    virtual void handle(Mult&     node) = 0;
    virtual void handle(Dot&      node) = 0;
};

template <typename Owner>
struct Dispatcher : public AbstractDispatcher
{
    // extra layer of indrection allows me to just template away handle impls
    // i dont care about and keeps the dispatcher code bundled better
    // TODO meta functor.. this will allow varying responses from Owner
    Owner& owner_;

public:
    Dispatcher(Owner& owner) :
        owner_(owner)
    {}

    void handle(Var&      node) { owner_.handle(node); }
    void handle(UnitVec&  node) { owner_.handle(node); }
    void handle(Element&  node) { owner_.handle(node); }
    void handle(Summer&   node) { owner_.handle(node); }
    void handle(Mult&     node) { owner_.handle(node); }
    void handle(Dot&      node) { owner_.handle(node); }
};

template <typename Owner, typename State>
struct StatefulDispatcher : public AbstractDispatcher
{
    // one more xtra layer of indrection from the normal dispatch so that
    // I can  pass a state or other parameter set to the typified call back
    // TODO meta functor.. this will allow varying responses from Owner
    Owner& owner_;
    State& state_;

public:
    StatefulDispatcher(Owner& owner,
                       State& state) :
        owner_(owner),
        state_(state)
    {}

    void handle(Var&      node) { owner_.handle(state_, node); }
    void handle(UnitVec&  node) { owner_.handle(state_, node); }
    void handle(Element&  node) { owner_.handle(state_, node); }
    void handle(Summer&   node) { owner_.handle(state_, node); }
    void handle(Mult&     node) { owner_.handle(state_, node); }
    void handle(Dot&      node) { owner_.handle(state_, node); }
};

// ################################################
// ################################################
// ################################################

struct Var : Node
{
    std::string name_;

    Var(std::string name) :
        Node(),
        name_(name)
    {}

    void visit(AbstractDispatcher& dispatcher) { dispatcher.handle(*this); }
};

struct Summer : Node
{
    Summer(const Handle& idx,
           const Handle& exp) :
        Node({idx, exp})
    {}

    void visit(AbstractDispatcher& dispatcher) { dispatcher.handle(*this); }
};

struct UnitVec : Node
{
    UnitVec(const Handle& idx) :
        Node({idx})
    {}

    void visit(AbstractDispatcher& dispatcher) { dispatcher.handle(*this); }
};

struct Element : Node
{
    std::string name_;

    Element(const std::string name) :
        Node(),
        name_(name)
    {}

    Element(const std::string name,
            const std::initializer_list<Handle> indexes) :
        Node(indexes),
        name_(name)
    {}

    void visit(AbstractDispatcher& dispatcher) { dispatcher.handle(*this); }
};

struct Mult : Node
{
    Mult(const Handle& left,
         const Handle& right) :
        Node({left, right})
    {}

    void visit(AbstractDispatcher& dispatcher) { dispatcher.handle(*this); }
};

struct Dot : Node
{
    Dot(const Handle& left,
        const Handle& right) :
        Node({left, right})
    {}

    void visit(AbstractDispatcher& dispatcher) { dispatcher.handle(*this); }
};

// ################################################
// ################################################
// ################################################

struct Make
{
    static Handle tensor(std::string name,
                         const std::initializer_list<std::string> shape)
    {
        // construct element
        Element* element = new Element(name);
        Handle exp(element);

        for (const std::string& id : shape)
        {
            Handle idx(new Var(id));
            element->children_.push_back(idx);
        }

        // now build summer and unit vars to form tensor
        for (Nodes::const_reverse_iterator iit = element->children_.rbegin();
             iit !=  element->children_.rend();
             ++iit)
        {
            const Handle& idx = *iit;
            Handle uvec(new UnitVec(idx));
            Handle mult(new Mult(uvec, exp));

            exp = mult;
        }

        for (Nodes::const_reverse_iterator iit = element->children_.rbegin();
             iit !=  element->children_.rend();
             ++iit)
        {
            const Handle& idx = *iit;
            Handle summer(new Summer(idx,exp));

            exp = summer;
        }

        return exp;
    }

    static Handle dot(const Handle& left,
                      const Handle& right)
    {
        return Handle(new Dot({left, right}));
    }

};

// ################################################
// ################################################
// ################################################

class Render
{
    std::ostream& os_;
    Dispatcher<Render> dispatch_;

    friend Dispatcher<Render>;

    void handle(Var&      node)
    {
        os_ << node.name_;
    }

    void handle(UnitVec&  node)
    {
        os_ << "U_";
        node.children_[0]->visit(dispatch_);
    }

    void handle(Element&  node)
    {
        os_ << node.name_ << "_";
        for (Nodes::const_iterator iit = node.children_.begin();
             iit != node.children_.end();
             ++iit)
        {
            (*iit)->visit(dispatch_);
        }
    }

    void handle(Summer&   node)
    {
        os_ << "sum_";
        node.children_[0]->visit(dispatch_);
        os_ << "(";
        node.children_[1]->visit(dispatch_);
        os_ << ")";
    }

    void handle(Mult&     node)
    {
        node.children_[0]->visit(dispatch_);
        os_ << "*";
        node.children_[1]->visit(dispatch_);
    }

    void handle(Dot&      node)
    {
        node.children_[0]->visit(dispatch_);
        os_ << ".";
        node.children_[1]->visit(dispatch_);
    }

public:
    Render(std::ostream& os) :
        os_(os),
        dispatch_(*this)
    {}

    void render(Handle exp)
    {
        exp->visit(dispatch_);
    }
};

std::ostream& operator<<(std::ostream& os,
                         const Handle& a)
{
    Render render(os);
    render.render(a);
    return os;
}

// ################################################
// ################################################
// ################################################

template <typename NodeType>
class Has
{
    // determine if the expression handed in has the type of node listed
public:
    Has() {}

    bool find(const Handle& node)
    {
        if (const NodeType* ptr = dynamic_cast<NodeType*>(node.get()))
            return true;

        for (Nodes::const_iterator iit = node->children_.begin();
             iit != node->children_.end();
             ++iit)
        {
            if (find(*iit))
                return true;
        }
        return false;
    }
};
// ################################################
// ################################################
// ################################################

// TODO rework as a dynamic cast(remove dispatch) and templated type (ie same as Has<>)
class LocateLastSum
{
    // given an expression that starts with summers find the last one in the chain of them

    Handle lastSummer_;
    Handle current_;
    Dispatcher<LocateLastSum> dispatch_;

    friend Dispatcher<LocateLastSum>;

    template<typename Specific>
    void handle(Specific& node)
    {
        // ok we have a no summer..
        // whatever is in my handles is the result so we just stop..
    }

    void handle(Summer& node)
    {
        // ok still a summer so update state and then see what my kid is
        lastSummer_ = current_;
        current_    = node.children_[1];

        current_->visit(dispatch_);
    }

public:

    LocateLastSum() :
        lastSummer_(),
        current_(),
        dispatch_(*this)
    {}

    Handle find(Handle exp)
    {
        current_ = exp;
        current_->visit(dispatch_);

        return lastSummer_;
    }
};

// ################################################
// ################################################
// ################################################

template <typename Operation>
class TransformAll
{
    // outer walker method to search the tree for transformation points..
    struct State
    {
        Handle parent_;
        Handle current_;
        int    currentsIdxInParent_;
    };

    friend StatefulDispatcher<TransformAll, State>;

    bool      found_;
    Handle    parent_;
    Handle    current_;
    int       currentsIdxInParent_;
    Operation op_;

    template<typename Specific>
    bool check(State& state, Specific& node)
    {
        // check child
        if (op_.isApplicable(node))
        {
            parent_              = state.parent_;
            current_             = state.current_;
            currentsIdxInParent_ = state.currentsIdxInParent_;
            found_               = true;
            return true;
        }
        return false;
    }

    void next(State& state)
    {
        for (int childIdx = 0;
             childIdx < state.current_->children_.size();
             ++childIdx)
        {
            State nextState;
            nextState.parent_  = state.current_;
            nextState.current_ = state.current_->children_[childIdx];
            nextState.currentsIdxInParent_ = childIdx;

            StatefulDispatcher<TransformAll, State> dispatch(*this,nextState);
            nextState.current_->visit(dispatch);
        }
    }

    template<typename Specific>
    void handle(State& state, Specific& node)
    {
        if (check(state, node))
            return;
        next(state);
    }

public:
    // lift all the sums above other ops
    TransformAll() :
        op_(),
        found_(false),
        parent_(),
        current_()
    {}

    Handle process(Handle exp)
    {
        found_ = true;
        while (found_)
        {
            // Reset  vars
            parent_.reset();
            current_.reset();
            currentsIdxInParent_ = -1;
            found_ = false;

            // search for transformable item
            {
                State state;
                state.parent_.reset();
                state.current_ = exp;
                state.currentsIdxInParent_ = -1;

                StatefulDispatcher<TransformAll, State> dispatch(*this,state);
                exp->visit(dispatch);
            }

            // TODO it maybe possible to restart searching from the parent of the
            // found node instead of the top of the tree.. check and optimise
            if (found_)
            {
                if (parent_.get() == NULL)
                {
                    exp = op_.transform(exp);
                }
                else
                {
                    Handle newCurrent = op_.transform(current_);
                    if (newCurrent != current_)
                        parent_->children_[currentsIdxInParent_] = newCurrent;
                }
            }
        }

        return exp;
    }
};

// ################################################
// ################################################
// ################################################

class LiftSum
{
    // lift all the sums above other ops
    //  starts as
    //               C
    //              / \
    //             S
    //             |
    //             K
    //  converts to
    //               S
    //               |
    //               C
    //              / \
    //             K

public:
    LiftSum()
    {}

    Handle transform(Handle node)
    {
        // note we only change once! as it might be chained on that side
        for (int childIdx = 0;
             childIdx < node->children_.size();
             ++childIdx)
        {
            if (const Summer* ptr = dynamic_cast<Summer*>(node->children_[childIdx].get()))
            {
                // ok found it
                Handle summer = node->children_[childIdx];

                // so rotate parent to child
                node->children_[childIdx] = summer->children_[1];
                summer->children_[1] = node;

                return summer;
            }
        }

        // odd transform failed..
        return node;
    }

    template<typename Specific>
    bool isApplicable(Specific& node)  { return false; }

    bool isApplicable(Mult&     node)
    {
        if (const Summer* ptr = dynamic_cast<Summer*>(node.children_[0].get()))
            return true;
        if (const Summer* ptr = dynamic_cast<Summer*>(node.children_[1].get()))
            return true;
        return false;
    }

    bool isApplicable(Dot&      node)
    {
        if (const Summer* ptr = dynamic_cast<Summer*>(node.children_[0].get()))
            return true;
        if (const Summer* ptr = dynamic_cast<Summer*>(node.children_[1].get()))
            return true;
        return false;
    }
};

// ################################################
// ################################################
// ################################################

class RotateDotsMultsToRight
{
    //  starts as: (U_k*x_k).(U_i*U_j*m_ij)
    //         .
    //       /   \
    //      *     *
    //     / \   / \
    //   Uk  Xk Ui  *
    //             / \
    //            Uj  Xij
    //  converts to: U_k*x_k.U_i*U_j*m_ij
    //      *
    //     / \
    //    Uk  .
    //       / \
    //      Xk  *
    //          / \
    //         Ui  *
    //            / \
    //           Uj  Xij

public:
    RotateDotsMultsToRight()
    {}

    Handle transform(Handle current)
    {
        Handle newTop = current->children_[0];

        current->children_[0] = newTop->children_[1];
        newTop->children_[1] = current;

        return newTop;
    }

    // for outer TransformAll to decide
    template<typename Specific>
    bool isApplicable(Specific& node)  { return false; }

    bool isApplicable(Dot&      node)
    {
        if (const Dot* ptr = dynamic_cast<Dot*>(node.children_[0].get()))
            return true;
        if (const Mult* ptr = dynamic_cast<Mult*>(node.children_[0].get()))
            return true;
        return false;
    }

    bool isApplicable(Mult&     node)
    {
        if (const Dot* ptr = dynamic_cast<Dot*>(node.children_[0].get()))
            return true;
        if (const Mult* ptr = dynamic_cast<Mult*>(node.children_[0].get()))
            return true;
        return false;
    }
};

class AttachDotsToUnitVectors
{
    // pre-requiste: the graph is right side orientation

    // given a dot or mult op... check its left side for rotatio{
    //  starts as: U_k*x_k.U_i*U_j*m_ij
    //      *
    //     / \
    //    Uk  .
    //       / \
    //      Xk  *
    //          / \
    //         Ui  *
    //            / \
    //           Uj  Xij
    //  converts to: U_k.x_k*U_i*U_j*m_ij
    //      .
    //     / \
    //    Uk  *
    //       / \
    //      Xk  *
    //         / \
    //        Ui  *
    //           / \
    //          Uj  Xij
public:
    AttachDotsToUnitVectors() {}

    Handle transform(Handle mult)
    {
        // hmm not safe for a self run...
        Handle dot = mult->children_[1];
        Handle nonVectorDotKids = dot->children_[0];

        dot->children_[0]  = mult->children_[0];
        mult->children_[0] = nonVectorDotKids;
        mult->children_[1] = dot->children_[1];
        dot->children_[1]  = mult;

        return dot;
    }

    // for outer TransformAll to decide
    template<typename Specific>
    bool isApplicable(Specific& node)  { return false; }
    bool isApplicable(Mult&     node)
    {
        Handle rightSide = node.children_[1];
        if (const Dot* ptr = dynamic_cast<Dot*>(rightSide.get()))
        {
            Has<UnitVec> hasVectors;
            if (not hasVectors.find(rightSide->children_[0]))
                return true;
        }
        return false;
    }
};

class LiftUnitVectorUp
{
    // stating with: U_k.x_k*U_i*U_j*m_ij
    //      .
    //     / \
    //    Uk  *
    //       / \
    //      Xk  *
    //         / \
    //        Ui  *
    //           / \
    //          Uj  Xij
    // convert to: U_k.U_i*U_j*x_k*m_ij
    //      .
    //     / \
    //    Uk  *
    //       / \
    //      Ui  *
    //         / \
    //        Uj  *
    //           / \
    //         Xk   Xij
public:
    LiftUnitVectorUp() {}

    Handle transform(Handle mult)
    {
        // hmm not safe for a self run...
        Handle mult2nd = mult->children_[1];
        Handle nonVectorLeftKids = mult->children_[0];

        mult->children_[0]  = mult2nd->children_[0];
        mult2nd->children_[0] = nonVectorLeftKids;

        return mult;
    }

    // for outer TransformAll to decide
    template<typename Specific>
    bool isApplicable(Specific& node)  { return false; }

    bool isApplicable(Mult&     node)
    {
        Handle rightSide = node.children_[1];
        if (const Mult* ptr = dynamic_cast<Mult*>(rightSide.get()))
        {
            if (const UnitVec* ptr = dynamic_cast<UnitVec*>(rightSide->children_[0].get()))
            {
                Has<UnitVec> hasVectors;
                if (not hasVectors.find(node.children_[0]))
                    return true;
            }
        }
        return false;
    }
};


//TODO simple optimiser
// 1. bring the Unit vectors together
// 2. then convert the Unit vectors seperated via dots to sigmas.
// 3. then locate summers tied to sigmas varables and reduce sigmas to 1
