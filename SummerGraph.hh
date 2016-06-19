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

class SumLifter
{
    //  starts as
    //                 P
    //                / \
    //               C
    //              / \
    //             S
    //             |
    //             K
    //  converts to
    //                 P
    //                / \
    //               S
    //               |
    //               C
    //              / \
    //             K

    struct State
    {
        Handle parent_;
        Handle current_;
    };

    friend StatefulDispatcher<SumLifter, State>;

    bool   found_;
    Handle parent_;
    Handle current_;
    Handle summer_;

    Handle lift(Handle exp)
    {
        // TODO this seems like a more general tree op.. better abstract it
        if (parent_.get() != NULL)
        {
            // now we need to rotate the summer up..
            // replace the "parents" link to "current" with "summer"
            for (Nodes::iterator nit = parent_->children_.begin();
                 nit != parent_->children_.end();
                 ++nit)
            {
                if (*nit == current_)
                {
                    *nit = summer_;
                    break;
                }
            }
        }

        // then replace the "currents" link to "summer" with summers kid
        for (Nodes::iterator nit = current_->children_.begin();
             nit != current_->children_.end();
             ++nit)
        {
            if (*nit == summer_)
            {
                *nit = summer_->children_[1];
                break;
            }
        }

        // finally replace "summers" link to its kid with "current"
        summer_->children_[1] = current_;

        if (parent_.get() == NULL)
        {
            // where at the top so replace the entire expersion
            exp = summer_;
        }

        return exp;
    }

    bool test(State& state, Handle& child)
    {
        // check child
        if (const Summer* ptr = dynamic_cast<Summer*>(child.get()))
        {
            parent_  = state.parent_;
            current_ = state.current_;
            summer_  = child;
            found_   = true;
            return true;
        }

        State nextState;
        nextState.parent_  = state.current_;
        nextState.current_ = child;

        StatefulDispatcher<SumLifter, State> dispatch(*this,nextState);
        child->visit(dispatch);
        return false;
    }

    template<typename Specific>
    void handle(State& state, Specific& node)
    {
        // ok we have a no summer.. or op we know as lift safe
        // whatever it is we have to stop..
    }

    void handle(State& state, Summer&   node)
    {
        // dont move summer ordering around but do move it kids on..
        State nextState;
        nextState.parent_  = state.current_;
        nextState.current_ = node.children_[1];

        StatefulDispatcher<SumLifter, State> dispatch(*this,nextState);
        node.children_[1]->visit(dispatch);
    }

    void handle(State& state, Mult&     node)
    {
        if (test(state, node.children_[0])) return;
        if (test(state, node.children_[1])) return;
    }

    void handle(State& state, Dot&      node)
    {
        if (test(state, node.children_[0])) return;
        if (test(state, node.children_[1])) return;
    }

public:
    // lift all the sums above other ops
    SumLifter() :
        found_(false),
        parent_(),
        current_(),
        summer_()
    {}

    Handle process(Handle exp)
    {
        found_ = true;
        while (found_)
        {
            // Reset  vars
            parent_.reset();
            current_.reset();
            summer_.reset();
            found_ = false;

            // search for sum to lift
            State state;
            state.current_ = exp;

            StatefulDispatcher<SumLifter, State> dispatch(*this,state);
            exp->visit(dispatch);

            if (found_)
            {
                exp = lift(exp);
            }
        }

        return exp;
    }
};

// ################################################
// ################################################
// ################################################

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
    Handle current_;

    Dispatcher<RotateDotsMultsToRight> dispatch_;

    friend Dispatcher<RotateDotsMultsToRight>;

    template<typename Specific>
    void handle(Specific& node)
    {
        // left side op is something we cant handle
        // whatever it is just stop..
    }

    void handle(Dot& node)
    {
        // ok left is also a dot..
        // well then thats the new top node..
        // rotate it up as the head
        doRotate();
    }

    void handle(Mult& node)
    {
        // ok left is also a dot..
        // well then thats the new top node..
        // rotate it up as the head
        doRotate();
    }

    Handle doRotate(Handle exp)
    {
        Handle newTop = current->children_[0];

        current_->children_[0] = newTop->children_[1];
        newTop->children_[1] = current_;

        current_ = newTop_;
    }


public:
    RotateDotsMultsToRight() :
        dispatch(*this)
    {}

    Handle transform(Handle curent)
    {
        current_ = current;
        // TODO .. outer core to walk the tree for candidates

        Dispatcher<RotateDotsMultsToRightr> dispatch(*this);
        current_->children_[0]->visit(dispatch);

        return current_;
    }

    // for outer Looper to decide
    template<typename Specific>
    bool isApplicable(Specific& node)  { return false; }
    bool isApplicable(Dot&      node)  { return true; }
    bool isApplicable(Mult&     node)  { return true; }
};

template <typename Operation>
class TransformAll
{
    // outer walker method for RotateDotsMultsToRightAll
    struct State
    {
        Handle parent_;
        Handle current_;
    };

    friend StatefulDispatcher<TransformAll, State>;

    bool   found_;
    Handle parent_;
    Handle current_;

    bool check(State& state)
    {
        // check child
        if (op.isApplicable(child))
        {
            parent_  = state.parent_;
            current_ = state.current_;
            found_   = true;
            return true;
        }
        return false;
    }

    void next(State& state)
    {
        for (Nodes::iterator nit = state.current_->children_.begin();
             nit = state.current_->children_.end();
             ++nit)
        {
            State nextState;
            nextState.parent_  = state.current_;
            nextState.current_ = child;

            StatefulDispatcher<SumLifter, State> dispatch(*this,nextState);
            child->visit(dispatch);
        }
    }

    template<typename Specific>
    void handle(State& state, Specific& node)
    {
        if (check(state))
            return;
        next(state);
    }

public:
    // lift all the sums above other ops
    TransformAll() :
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
            found_ = false;

            // search for sum to lift
            State state;
            state.current_ = exp;

            StatefulDispatcher<SumLifter, State> dispatch(*this,state);
            exp->visit(dispatch);

            if (found_)
            {
                exp = lift(exp);
            }
        }

        return exp;
    }

}

class AttachDotsToUnitVectors
{
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
    //          / \
    //         Ui  *
    //            / \
    //           Uj  Xij
};

class LiftUnitVectorToLeft
{
    // stating with: U_k.x_k*U_i*U_j*m_ij
    //      .
    //     / \
    //    Uk  *
    //       / \
    //      Xk  *
    //          / \
    //         Ui  *
    //            / \
    //           Uj  Xij
    // convert to: U_k.U_i*U_j*x_k*m_ij
    //      .
    //     / \
    //    Uk  *
    //       / \
    //      Ui  *
    //          / \
    //         Uj  *
    //            / \
    //          Xk   Xij
}

class UnitVectorGatherLeft
{
    // move all Unit vecotrs to the left side following summers
    // be careful as dot products can not be crossed by unit vectors

    // assumes that the summers have been moved to left most outher

    //  starts as: (U_k*x_k).(U_i*U_j*m_ij)
    //         .
    //       /   \
    //      *     *
    //     / \   / \
    //   Uk  Xk Ui  *
    //             / \
    //            Uj  Xij
    //  convrts to: U_k*x_k.U_i*U_j*m_ij
    //      *
    //     / \
    //    Uk  .
    //       / \
    //      Xk  *
    //         / \
    //        Ui  *
    //           / \
    //          Uj  Xij
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
    //  converts to: U_k.U_i*x_k*U_j*m_ij
    //     .
    //    / \
    //   Uk  *
    //      / \
    //     Ui  *
    //        / \
    //       Xk  *
    //          / \
    //         Uj  Xij
    //  converts to: U_k.U_i*U_j*x_k*m_ij
    //     .
    //    / \
    //   Uk  *
    //      / \
    //     Ui  *
    //        / \
    //       Uj  *
    //          / \
    //         Xk  Xij

public:

    UnitVectorGatherLeft()
    {}

     Handle process(Handle exp)
     {
         // first locate the last of the summer chains
         LocateLastSum lls;
         Handle lastSum = lls.find(l);

         Handle workingPoint;
         if (lastSum.get() == NULL)
         {
             workingPoint = exp;
         }
         else
         {
             workingPoint = lastSum->children_[1];
         }

         // ok so now we are at the working point in the graph..
         // we want to locate all unit vectors and bring them into
         // the left sides at the working point

         // rotate all mults/dots to right hand side
         RotateDotsMultsToRight rotateToRight;
         workingPoint = rotateToRight.rotate(workingPoint);

         AttachDotsToUnitVectors dotToUnits;
         workingPoint = dotToUnits.gather(workingPoint);

         LiftUnitVectorToLeft liftUnits;
         workingPoint = liftUnits.gather(workingPoint);

         // reconnect end working point to summers

         if (lastSum.get() == NULL)
         {
             exp = workingPoint;
         }
         else
         {
             lastSum->children_[1] = workingPoint;
         }

         return exp;
     }
};


//TODO simple optimiser
// 1. bring the Unit vectors together
// 2. then convert the Unit vectors seperated via dots to sigmas.
// 3. then locate summers tied to sigmas varables and reduce sigmas to 1
