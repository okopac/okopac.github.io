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
 and units. Its an implementation that's limited (for reasons discussed below) but simple
 to understand and work with.

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

As described above, this implementation is built from two fundamental components, `units`
 and `gates`. Units are bound to gates as thier inputs and outputs, and gates are bound
 together by using common units between input and output.

 DIAGRAM

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

During training we also need a backward method to propagate the paramter updates back
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

The `Sigmoid` and `TanH` gates are fairly simple in comparison, in that they have a single
input and perform a simple mathematical transformation to create an output value. These transformations
are non-linear, as shown below. This non-linearity help the neural network represent more
complex functions.

# Conclusions
Using just these components we can build complex networks that are capable of modeling
[complex structures](https://github.com/okopac/pynet/blob/master/gatebased/LayerNNExample.ipynb).
We can combine individual gates into layers.

Writing a neural network from scratch has helped to explain some of the fundamental concepts like
minimising loss, back propagation and gradient descent.

It's been a great introduction into the subject, but using single gates is not a great way to
scale out network design. Python is a language that has lots of matrix support, so my next step
will be to implement some NN using numpy, and try and tackle some more complex problems.

[jekyll-docs]: http://jekyllrb.com/docs/home
[jekyll-gh]:   https://github.com/jekyll/jekyll
[jekyll-talk]: https://talk.jekyllrb.com/
