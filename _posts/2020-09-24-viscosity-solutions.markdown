---
layout: post
title: Making Sense of Continuous RL with Viscosity Solutions
# featured-img: optimal-couplings
categories: math reinforcement-learning
---
<link rel="stylesheet" href="/assets/css/mathstuff.css">

I'm writing this post to address a fairly shocking issue that arises when considering the plight of
RL algorithms in the limit of continuous time updates. This post is heavily inspired by [Remi
Munos](http://researchers.lille.inria.fr/munos/)' manuscript, [A Study of Reinforcement Learning in
the Continuous Case by the Means of Viscosity
Solutions](https://link.springer.com/article/10.1023/A:1007686309208), which itself builds on the
theory of *viscosity solutions*, developed by [Crandall and
Lions](https://www.ams.org/journals/tran/1983-277-01/S0002-9947-1983-0690039-8/S0002-9947-1983-0690039-8.pdf)
in 1983. While Munos' paper does a fantastic job of pointing out the flaws with familiar
reinforcement learning algorithms in the continuous time setting and how to go about improving them,
the notion of viscosity solutions is left essentially as a black box. On the other hand, Crandall
and Lions' seminal work on viscosity solutions thoroughly analyses the properties of viscosity
solutions, it does not consider their implications on reinforcement learning, and some mathematical
steps in the theory are skipped, making it difficult for me to follow at times with my background in
engineering and computer science. In this post, I'll try to introduce the concept of viscosity
solutions in a more approachable manner.

# Continuous Time RL and the HJB Equation

This post is concerned only with value-based RL methods. We'll consider an MDP with state space
$$\mathcal{X}$$, action space $$\mathcal{A}$$, and state transition kernel $$P(x_{t+1}, r_t\mid x_t, a_t)$$.
For a given policy $$\pi$$, we define

$$
P^\pi(x_{t+1}\mid x_t) = \int_{\mathbf{R}}P(x_{t+1}, r_t\mid
    x_t, a_t)\pi(a_t\mid s_t)dr_t
$$

In familiar (discrete-time) RL, our goal is to learn the *value function* $$V^\pi$$ for a given
policy $$\pi$$,

$$
V^\pi(x) = \underset{x_{t+1}\sim P^\pi(\cdot\mid
    x_t)}{\mathbf{E}}\left\{\sum_{t=0}^\infty\gamma^tR_t\mid x_0=x\right\} = R_t + \gamma\underset{x_{t+1}\sim P^\pi(\cdot\mid
    x_t)}{\mathbf{E}}\left\{V^\pi(x_{t+1})\right\}
$$

The natural extension of this to the continuous time setting is as follows,

$$
V^\pi(x) = \underset{x_{t+1}\sim P^\pi(\cdot\mid
    x_t)}{\mathbf{E}}\left\{\int_{t=0}^\infty\gamma^tR_tdt\mid x_0 = x\right\} = \mathbf{E}\left\{\int_0^\tau\gamma^tR_tdt + \gamma^\tau
  V^\pi(x_\tau)\right\}
$$

where $$\tau$$ can be any positive real number. We're concerned now about what happens as $$\tau\to
0$$. By writing the first order Taylor approximation of the equation above about $\tau = 0$ and
taking the limit as $\tau\to\ 0$, we arrive at what's known as the **Hamilton-Jacobi-Bellman (HJB)
Equation**,

$$
V^\pi(x)\ln\gamma = R + \underset{a\sim\pi(\cdot\mid x)}{\mathbf{E}}\bigg\langle \nabla_xV^\pi(x),
f(x, a)\bigg\rangle
$$

where $$f(x, a)\triangleq\frac{d}{dt}x$$.
