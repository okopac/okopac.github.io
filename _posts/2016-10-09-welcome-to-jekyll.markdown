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

This post covers some of the basics that I have gained from this work. All code is
on [my github page](https://www.github.com/okopac/pynet/gatebased). Although the maths
is really interesting, I've tried to avoid getting into it below. There are many great
references which I've linked below.

# Basics of neuralnets

Neural networks have lots of different types, architectures and components, but fundamentally
they are trying to approximate a function.

> Given an input `x`, what is best approximation `f'(x)` for the output `f(x)`?

This is achieved by taking an input vector and running it through a network of transformations,
with the output of each transformation feeding back into the next. Importantly there are weights
between the edges of each transformation which explain how important the result of the previous
neuron is in dictating the input for the next. These weights are the key part of the network,
changing these parameters allows us to fit any function we desire.

# Minimising loss and back propagation

There are lots of great resources out these to learn about this subject. Sufficed to say
for this implementation I needed just the basics.

Minimising loss means 'how can I change my function approximation so it has the least errors
when compared to the real function'.

Back propagation is the process through which the error in our guess of the output is
learnt from. Basically we are asking the question, how should I change the parameters in my
network to make the error (or loss) lower. In reality this means differentiating our function and using the chain rule
to pass the desired change in output back through all the parameters we could possibly change.

# Code Architecture

Following on the from previously mentioned blog, I've implemented a neural network using the
basic principle of two components: gates and units.

A unit is simply a variable. It can be fixed (like the input x) or variable (like
  a parameter of the network). In the case of the former we need only its value. For the latter
  we also need the gradient (the direction in which we wish to move it).

{% highlight python %}

class Unit(object):

def __init__(self, value = None, grad = None, name=None):
    super(Unit, self).__init__()
    self.value = value
    self.grad = grad
    self.name = name

{% endhighlight %}

A gate is a transformation, it take a unit, performs some simple transformation (possibly
  combining it with other input units) and outputs a new unit. We need to know two things, how
  to perform the forward transformation and how to pass the gradient back through when
  performing back propagation.

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

Using just these components we can build complex networks that are capable of modeling
[complex structures](https://github.com/okopac/pynet/blob/master/gatebased/LayerNNExample.ipynb).
We can combine individual gates into layers.

# Conclusions

Writing a neural network from scratch has helped to explain some of the fundamental concepts like
minimising loss, back propagation and gradient descent.

It's been a great introduction into the subject, but using single gates is not a great way to
scale out network design. Python is a language that has lots of matrix support, so my next step
will be to implement some NN using numpy, and try and tackle some more complex problems.

[jekyll-docs]: http://jekyllrb.com/docs/home
[jekyll-gh]:   https://github.com/jekyll/jekyll
[jekyll-talk]: https://talk.jekyllrb.com/
