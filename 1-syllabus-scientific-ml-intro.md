# Syllabus

https://github.com/mitmath/18337

Studying scientific machine learning: solving differential equations using a mix of traditional methods and machine learning

Looking for ways to slam together traditional methods and machine learning

Hoping to get publications out of this

Looking for places people use different words for the same things; can let us slam things together

Can work together on projects but writeups individual

* can i throw interpretability / adversarial training at some of the problems in this class?

Mostly aimed at scientific computing, but neural networks integrate (heh) very smoothly into those

# What is scientific machine learning?
Domain models with integrated machine learning -- take navier stokes, climate models, ..., and throw machine learning at them

In a mathematical sense, a model is just a nonlinear function

Almost never have an analytical solution to your problem / version of your model model

So, 2 directions:
    1. differential equation / mechanistic models: a structural representation of the model
        why? in a science experiment, you're always measuring a change in the system -- the variation of one thing with another
        take a system, perturb it a little, see what changes
        all scientific laws are "this changes like this"
        so: differential equation! a way of describing a nonlinear function by its structure
        can almost never solve a differential equation 
    2. machine learning: specify a set of parameters to a black box

mechanistic models understand the structure, so they can extrabolate beyond data
    are interpretable, can be extended, transferred, understood
    however, require that you understand mechanism
non-mechanistic models can give a very predictive model directly from data, but w/o interpretability or extrapolability of mechanistic models

neither is better than the other

goal: combine mechanistic + non-mechanistic models to get the best (* and worst??) of both worlds.

# idea: latest machine learning revolution can be interpreted this way!
the reason this works: you can interpret it as including prior information about how spatial structure in images works

# machine learning in a nutshell: learnable functions and the universal approximation theorem
UAT: neural networks can get \eps close to any \R^n -> \R^n function

    $$\forall f \in C^\inf, \exists \theta s.t. f_\theta(x) : \R^n -> \R^m$$

where \theta: parameters, \f_\theta is a "big enough" neural network, then
* what is C?

    $$||f(x) - f_\theta(x)|| < \eps \forall x \in \Omega$$

to describe NN, describe a layer: $\sigma(Wx + b)$
    W: parameter
    b: parameter

    \sigma: "activation function", lots of them; choosing more an art than a science, but can know e.g. whether smooth or not
    $$NN_\theta(x) = L(L(L( .. L(x) .. )))$$
    $$             = o_{i=0}^N L_i(x)$$

where o: function composition

several recent survey papers on UAT -- can approach convergence from many directions, different approaches

-> neural networks are just function expansions, fancy Taylor Series-like things that are good for computing and bad for analysis

# what's out there in scientific machine learning?
not much

# a few results

## hidden physics models
instead of learning a neural network to approximate *all* of your data; use some difeq and some nn

    $$du/dt = N(...) + NN_\theta(...)$$

where N: correct physics, NN: neural network

can use much smaller data (e.g. 30 data points) and networks

## physics-informed neural networks
look at:

    $$du/dt = NN(u)$$ 

use euler's method or runge-kutta:

    u0 = ...
    u1 = u0 + NN(u0)dt
    u2 = u1 + NN(u1)dt

becomes a recurrent neural network!


## solving 1000-dimensional PDEs
crazy, but:
...math stuff...

interpreting learned NNs to find analytical solutions?

interesting result: stochastic RNN
    take system, write simplest PDE solver they can find; end up with a NN with randomness injected all over the place

## Finding Koopman Embeddings w/ Autoencoders
find linear things with autoencoders

## DGM: a deep learning algorithm for solving partial differential equations
just solve the whole equation w/ deep learning

sometimes faster, sometimes slower

# reverse: deep learning impacted by scientific computing

## Deep neural networks motivated by partial differential equations
...

## Neural Ordinary Differential Equations
don't use an RNN, just use a difeq defined by a neural network and then use an ODE solver; gives you lots of nice stuff

ODEs generalize rnns

# Prof's recent work
Latent difeqs: modeling without models -> backprop through ODE solvers

Interpretability: what does it actually learn?
    approximations of small parts of phase space!

Don't throw away structure, use it!

# Optimizing serial code

## Mental model of memory

blah blah

* binding tiramisu from julia?



