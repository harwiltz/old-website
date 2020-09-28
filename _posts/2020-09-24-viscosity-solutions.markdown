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

where $$f(x, a)\triangleq\frac{d}{dt}x$$. Note that a differentiable value function satisfies the
HJB equation. However, as we'll see, even very simple continuous time MDPs can admit
non-differentiable value functions.

# A Problematic MDP

Remi Munos proposed a very simple continuous-time MDP in the paper cited above that demonstrates
some issues that can arise if we naively adapt familiar RL algorithms to continuous time.

## The Munos MDP

The MDP has state space $$\mathcal{X}=[0,1]$$ and action space $$\mathcal{A}=\{\pm 1\}$$. We control
a particle with dynamics satisfying $$\dot{x}(t) = a(t)$$, where $$\dot{x}(t)$$ and $$a(t)$$ are the
velocity of the particle and the chosen action respectively at time $$t$$. For any $$x\in (0, 1)$$
the agent receives no reward, and at the endpoints the rewards are given by constants $$R_0$$ and
$$R_1$$ for $$x = 0$$ and $$x=1$$ respectively. We will set $$R_0=1$$ and $$R_1=2$$. We also use a
discount factor $$\gamma=0.3$$.

## The Optimal Policy

It's relatively easy to compute the optimal policy for the Munos MDP. Clearly there will be a
threshold point $$\overline{x}$$ such that the optimal policy $$\pi^{\star}$$ will satisfy

$$
\pi^{\star}(a\mid x) = -\mathbf{1}_{[x\leq\overline{x}]} + \mathbf{1}_{[x>\overline{x}]}
$$

## The Value Function

It's also fairly easy to derive the value function analytically. When $$x\leq\overline{x}$$, we have

$$V(x) = R_0\gamma^{x} = 0.3^{x}$$

Similarly, when $$x>\overline{x}$$, we have

$$V(x) = R_1\gamma^{1-x} = 2(0.3)^{1-x}$$

Moreover, note that $$\overline{x}$$ by definition satisfies

$$
R_0\gamma^{\overline{x}} = R_1\gamma^{1-\overline{x}}
$$

After some simple algebra, we find that $$\overline{x} = \frac{\log (R_1/R_0)}{2\log\gamma} +
\frac{1}{2}$$. The value function looks like

![](/assets/img/viscosity/viscosity-graph.png)

Notice that at $$\overline{x}$$ the value function is not differentiable. Does this mean that it
cannot solve the HJB equation?

You might also be curious about why the continuous-time setting introduces this issue -- after all,
the value function does not depend on time. However, the derivative of the value function with
respect to $$x$$ arise from the chain rule when computing $$dV/dt$$ in the derivation of the HJB
equation. Very subtle.

# Viscosity Solutions to the Rescue

As we saw, simple MDPs can have non-differentiable value functions, but the HJB equation requires
computation of the value function's gradient. If we try to settle for
almost-everywhere-differentiable functions, we lose uniqueness guarantees, so there can potentially
be a bunch of functions satisfying the HJB that don't actually compute the expected return. However,
it can be shown that the HJB equation admits unique *viscosity solutions*, which exhibit exactly
those properties that we'd like from the value function.

## What is a Viscosity Solution?

A viscosity solution is a function obeying some properties (don't even try to understand these yet)
that satisfy partial differential equations of the form

$$
H(x, v, Dv) = 0
$$

where we'll interpret $$x$$ as a state, $$v$$ as a value function, and $$Dv = \nabla_xv$$.
Furthermore, $$H$$ is an arbitrary function that is monotonically non-decreasing in $$v$$, known as
the Hamiltonian. In the case of the HJB equation, we see that

$$
H(x, V, DV) = -V(x)\ln\gamma + R + \bigg\langle\nabla_xV(x), f(x, \pi^{\star}(x))\bigg\rangle
$$

At a high level, viscosity solutions are continuous functions that are differentiable almost
everywhere.

## Intro to the Design of Viscosity Solutions

I'll conjecture that without ever having been exposed to the concept of viscosity solutions, simply
looking at the definition will be fairly demoralizing. According to Crandall and Lions in their
seminal paper on the topic, the following is what led to the development of the theory of viscosity
solutions. Let's start by assuming that $$V\in C^1(\mathcal{X})$$ (note that we of course don't want
to restrict viscosity solutions to functions of this class!). Consider another function $$\phi\in
C^1(\mathcal{X})$$, where $$\phi(x)V(x) = \max\phi V$$ at some state $$x$$, and $$\phi(x)V(X)>0$$.
Note that

$$
D(\phi V) = D\phi V + \phi DV
$$

This leads us to the following representation of $$DV(x)$$,

$$
DV(x) = -\frac{V(x)}{\phi(x)}D\phi(x)
$$

since, by definition, $$D(\phi V)(x) = 0$$.

Therefore, it suffices to equivalently solve

$$
H\left(x, V(x), -\frac{V(x)}{\phi(x)}D\phi(x)\right) = 0
$$

The ingenuity in Crandall and Lions' design of the notion of a viscosity solution revolves around
allowing stronger smoothness assumptions on $$\phi$$ than an $$V$$.

## The Definition of a Viscosity Solution

The properties of a viscosity solution are heavily inspired by the discussion above. Let
$$\mathcal{X}\subset\mathbf{R}^N$$ and $$\mathcal{D}(\mathcal{X})^{+}$$ denotes the continuously
differentiable, positive functions supported on compact subsets of $$\mathcal{X}$$. We also say

$$E_{+}(f)\triangleq\{x\in\mathcal{X}\mid f(x) = \max f > 0\}$$

and likewise

$$E_{-}(f)\triangleq \{x\in\mathcal{X}\mid f(x) = \min f \lt 0\}$$

Now we can define the viscosity solutions.

<div class="definition">
<b>Definition (Viscosity Subsolution):</b> A viscosity subsolution is a function in
\(\mathcal{C}(\mathcal{X})\) such that for every \(\phi\in\mathcal{D}(\mathcal{X})^{+}\) and
\(k\in\mathbf{R}\) we have

\begin{align}
&E_{+}(\phi\cdot(u-k))\neq\emptyset\implies\\&\quad\exists y\in E_{+}(\phi(u-k)): H\left(y, V(y), -\frac{V(y) -
    k}{\phi(y)}D\phi(y)\right)\leq 0
\end{align}

</div>

<div class="definition">
<b>Definition (Viscosity Supersolution):</b> A viscosity supersolution is a function in
\(\mathcal{C}(\mathcal{X})\) such that for every \(\phi\in\mathcal{D}(\mathcal{X})^{+}\) and
\(k\in\mathbf{R}\) we have

\begin{align}
&E_{-}(\phi\cdot(u-k))\neq\emptyset\implies\\&\quad\exists y\in E_{-}(\phi(u-k)): H\left(y, V(y), -\frac{V(y) -
    k}{\phi(y)}D\phi(y)\right)\geq 0
\end{align}

</div>

<div class="definition">
<b>Definition (Viscosity Solution):</b> A viscosity solution is a function that is both a viscosity
subsolution and a viscosity supersolution.
</div>

## The Value Function as the Unique Viscosity Solution

We'll now show a beautiful property of viscosity solutions, that being that viscosity solutions
$$H(x, V, DV) = 0$$ (and therefore viscosity solutions of the HJB equation) are unique. We'll go
over a proof from the seminal paper by Crandall and Lions.

<div class="theorem">
<b>Theorem (Uniqueness of Viscosity Solutions):</b> Let \(V\) be a bounded viscosity subsolution of
                                                 \(H(x, V, DV) = 0\) and let \(V'\) be a bounded
                                                 viscosity supersolution of \(H(x, V', DV') =
                                                 m(x)\), where \(m:\mathbf{R}^N\to\mathbf{R}\) is a
                                                 bounded continuous function. Moreover, let \(R_0 =
                                                 \max(\|V\|_{L^{\infty}(\mathbf{R}^N)},
                                                     \|V'\|_{L^{\infty}(\mathbf{R}^N)})\). We make
                                                     the following assumptions:
<ol>
  <li>
    For any \(R > 0\), \(H\) is uniformly continuous on \(\mathbf{R}^N\times[-R,R]\times B(0,R)\)
  </li>
  <li>
    For each \(R > 0\), there is a continuous non-decreasing function
    \(\gamma_R:[0,2R]\to\mathbf{R}\) with \(\gamma_R(0)=0\) and

    $$
    H(x, r, p) - H(x, s, p)\geq\gamma_R(r - s)
    $$

    for each \(x, p\) and \(r, s\in[-R,R]\) with \(r\geq s\).
  </li>
  <li>
    For each \(R_1,R_2>0\), the following holds:
    \begin{align}
    \limsup_{\varepsilon\downarrow 0}\{&|H(x, r, p) - H(y, r, p)| :\\&|p(x-y)|\leq R_1,
    |x-y|\leq\varepsilon, r\leq R_2\} = 0
    \end{align}
  </li>
</ol>
Then, we have

$$
\|\gamma((V-V')^{+})\|_{L^{\infty}(\mathbf{R}^N)}\leq\|m^{+}\|_{L^{\infty}(\mathbf{R}^N)}
$$

where \(f^+ = \max(f, 0)\).
</div>
