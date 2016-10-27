---
layout: post
title:  "Learning through logic gates"
date:   2016-10-24 18:46:51 +0100
categories: neuralnetworks
---
I've got a background in software and data analysis, but for a while now I've wanted
to learn more about machine learning, specifically neural networks. The
first attempt at doing this followed [this fantastic guide](http://karpathy.github.io/neuralnets/)
which happened to described how to implement a basic neural network in javascript. Rather
than just read along, I thought I'd spend some time rewriting and extending the work
in python.

Rather than talk through the basics of neural networks, this post discusses this specific
implementation, where the neural network is comprised entirely of two components: gates
 and units. Its an implementation that's limited in use (for reasons discussed below), but
 incredibly useful as a learning tool for understanding some of the fundamentals of neural networks.

All code is on [my github page](https://www.github.com/okopac/pynet/gatebased).

# Basics of neuralnets

Neural networks have lots of different types, architectures and components, but fundamentally
they are trying to approximate a function.

> Given an input `x`, what is best approximation `f'(x)` for the output `f(x)`?

This is achieved by taking an input vector and running it through a network of transformations,
with the output of each transformation feeding back into the next. Importantly there are weights
between the edges of each transformation which explain how important the result of the previous
neuron is in dictating the input for the next.

These weights are the key part of the network, changing these parameters allows us to
fit any function we desire. The processing of training a neural network is really just
changing all the weights to minimise the error between prediction `f'(x)` and reality
`f(x)`. This is achieved through a process called backpropagation.

# Code Architecture

As described above, this implementation is built from two fundamental components, `Units`
 and `Gates`. Units are bound to gates as their inputs and outputs, and gates are bound
 together by using common units between input and output.

 <img src='http://g.gravizo.com/g?
  digraph G {
    iu1 [label="Input Unit 1"];
    iu2 [label="Input Unit 2"];
    iu3 [label="Input Unit 3"];
    o1 [label="Output Unit"];
    o1 [label="Output Unit"];
    g1 [label="Gate 1" shape=square]
    g2 [label="Gate 2" shape=square]
    iu1 -> g1;
    iu2 -> g1;
    g1 -> o1;
    o1 -> g2;
    iu3 -> g2;
  }
 '/>

We discuss these below, and then move on to how we can combine these gates and units into
more complex structures to create a range of neural networks.

## Unit

A `unit` is simply a variable. It can be fixed (like the input x) or variable (like
  a parameter of the network). In the case of prediction (forward propagation) we need
  only its `value`.

When training the model, we also need a `gradient`, to store the direction
  and magnitude of change we should apply in order to reduce the prediction error.

In our example, we also optionally name our units to help with understanding the underlying
 processes within the network.

{% highlight python %}

class Unit(object):

  def __init__(self, value = None, grad = None, name=None):
      super(Unit, self).__init__()
      self.value = value
      self.grad = grad
      self.name = name

{% endhighlight %}

## Gate

A `gate` is the computational heart of the network. It take a unit, performs some simple transformation (possibly combining it with other input units) and outputs a new unit.

When calculating the output of the model the gate is used in the forward direction. It uses
the values of the inputs bound to to, and updates the output value to reflect this transformation.

During training we also need a backward method to propagate the parameter updates back
through the model.

{% highlight python %}
class Gate(object):
    def __init__(self, ucreator):
        super(Gate, self).__init__()
        self.ucreator = ucreator
        self.inputs = []
        self.output_unit = self.ucreator.new_unit()

    def forward(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

{% endhighlight %}

I've implemented a couple of key gates to build basic networks, called `Combiner`, `Sigmoid`,
`TanH`.

The `Combiner` gate is the building block of the network, and is the only gate that has more
than one input. It essentially models the equation `g(x1, x2, ...) = a0 + a1*x1 + a2*x2 + ...`,
combining input units with a weighted sum, including a bias term `a0` at the start. These
gates are responsible for combining together units from previous gates in the network before passing
them into one of the second set of single input/output gates.

The `Sigmoid` and `TanH` gates are both fairly simple in comparison, in that they have a single
input and perform a simple mathematical transformation to create an output value. These transformations
are non-linear, as shown below. This non-linearity help the neural network represent more
complex functions. You'll find these referred to as activation gates, basically determining what
input signal is required to 'fire' the output.

<img src="{{baseurl}}/assets/sigtan.png" align="center"/>

### Aside: unit creator

Whilst writing this code I started to find I would loose track of where units were created.
This was most apperent in the training phase where I would want to iterate over all units
in the network, and apply the gradient (as derived via backpropagation).

To deal with this problem, I introduced a unit creator class. This class is the single
place in the code base where units are created. Doing this allows us to easily keep track
of units in use, and have simple methods for applying gradient's during training.

# Example networks

I implemented two different networks using this framework, an SVM and a feed forward network.
I won't dig into the training process here, but a couple of key features are:

* I'm creating classification networks, that predict one of two outputs either 1 or -1
* The actual output of the network is a number. Anything above 0 implies the result 1,
anything below implies -1.
* I'm propagating a fixed gradient back through the network, best described by the following code snippet.
{% highlight python %}

pull = 0
if label == 1 and output < 1:
    pull = 1.
elif label == -1 and output > -1:
    pull = -1.
elif not label in [1, -1] :
    raise Exception("Label must be 1 or -1 (%f)" % label)

{% endhighlight %}
* I'm using stochastic gradient descent (i.e. updating the parameters after each training example).

## SVM

Using these basic constructs we can create a simple SVM model, that is a network defined by
the equation `f(x, y) = ax + bx + c`. In our case this consists entirely of a single `Combiner`
gate.

<img src='http://g.gravizo.com/g?
 digraph G {
   x [label="x"];
   y [label="y"];
   a [label="a"];
   b [label="b"];
   c [label="c"];
   o [label="output"]
   g [label="ax + bx + c" shape=square]
   x -> g;
   y -> g;
   a -> g;
   b -> g;
   c -> g;
   g -> o;
 }
' align="center"/>

# FFN (Feed Forward Networks)

When I first learnt about neural networks, this was the kind of network that was in my head.
Its the most basic to understand, a network comprised of layers of gates,  where each layer of
gates if fully connected to the next.

Networks can be fully parameterised by the number of layers, the type of activation
function, and the number of inputs/outputs. In our example we will stick with having
just two inputs (`x`, `y`) and a single output. A network with 3 layers of size 3 would
look like the diagram below.

<img src='http://g.gravizo.com/g?
 digraph G {
   x->a0;x->a1;x->a2;x->a3;y->a0;y->a1;y->a2;y->a3;
   a0->b0;a0->b1;a0->b2;a0->b3;a1->b0;a1->b1;a1->b2;a1->b3;a2->b0;a2->b1;a2->b2;a2->b3;a3->b0;a3->b1;a3->b2;a3->b3;
   b0->c0;b0->c1;b0->c2;b0->c3;b1->c0;b1->c1;b1->c2;b1->c3;b2->c0;b2->c1;b2->c2;b2->c3;b3->c0;b3->c1;b3->c2;b3->c3;
   c0->out;c1->out;c2->out;c3->out;
 }
' align="center"/>

The diagram is a little more complicated in this in our example. We only have one gate
type `Combiner` that is able to process multiple inputs at each layer. As such each of the gates
draw above is actually a combination of a `Combiner` gate, followed by an `Activation` gate (which
  could be `Linear`, `Sigmoid` or `TanH`).

  <img src='http://g.gravizo.com/g?
   digraph G {
     x->comb_a0;x->comb_a1;x->comb_a2;x->comb_a3;y->comb_a0;y->comb_a1;y->comb_a2;y->comb_a3;
act_c0->out;act_c1->out;act_c2->out;act_c3->out;
act_a0->comb_b0;act_a0->comb_b1;act_a0->comb_b2;act_a0->comb_b3;comb_a0->act_a0;act_a1->comb_b0;act_a1->comb_b1;act_a1->comb_b2;act_a1->comb_b3;comb_a1->act_a1;act_a2->comb_b0;act_a2->comb_b1;act_a2->comb_b2;act_a2->comb_b3;comb_a2->act_a2;act_a3->comb_b0;act_a3->comb_b1;act_a3->comb_b2;act_a3->comb_b3;comb_a3->act_a3;
act_b0->comb_c0;act_b0->comb_c1;act_b0->comb_c2;act_b0->comb_c3;comb_b0->act_b0;act_b1->comb_c0;act_b1->comb_c1;act_b1->comb_c2;act_b1->comb_c3;comb_b1->act_b1;act_b2->comb_c0;act_b2->comb_c1;act_b2->comb_c2;act_b2->comb_c3;comb_b2->act_b2;act_b3->comb_c0;act_b3->comb_c1;act_b3->comb_c2;act_b3->comb_c3;comb_b3->act_b3;
comb_c0->act_c0;comb_c1->act_c1;comb_c2->act_c2;comb_c3->act_c3;
}
' align="center"/>

# Conclusions
Using just these components we can build complex networks that are capable of modeling
[complex structures](https://github.com/okopac/pynet/blob/master/gatebased/LayerNNExample.ipynb).
We can combine individual gates into layers.

Writing a neural network from scratch has helped to explain some of the fundamental concepts like
minimising loss, back propagation and gradient descent.

It's been a great introduction into the subject, but using single gates is not a great way to
scale out network design. Python is a language that has lots of matrix support, so my next step
will be to implement some NN using numpy, and try and tackle some more complex problems.

# References
[^1]: https://gist.github.com/okopac/957f275251fd4bb3aacc8eb36a7c841c
[jekyll-docs]: http://jekyllrb.com/docs/home
[jekyll-gh]:   https://github.com/jekyll/jekyll
[jekyll-talk]: https://talk.jekyllrb.com/
