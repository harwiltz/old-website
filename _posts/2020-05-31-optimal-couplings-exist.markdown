---
layout: post
title: Optimal Couplings Exist
featured-img: optimal-couplings
categories: math
---
<link rel="stylesheet" href="/assets/css/mathstuff.css">

This post is a journey through the proof of a beautiful fact from the theory of optimal
transportation. First let's define what a coupling is.

Given two probability spaces $$(\mathcal{X},\mu), (\mathcal{Y},\nu)$$, a *coupling* of $$(\mu,\nu)$$
is simply a pair (shall we say couple?) of random variables $$(X, Y)$$ such that
$$X\sim\mu,Y\sim\nu$$. On the surface, this does not seem particularly inspiring, however we will
hopefully learn that couplings can dramatically simplify analysis. Constructing coupling arguments
has a somewhat artistic nature, since the simplification analysis often relies on choosing a clever
correlation between the couple of random variables.

Now let's define what it means for a coupling to be optimal. Suppose we have two metric spaces
$$\mathcal{X},\mathcal{Y}$$ and some cost function $$c$$ that assigns a cost of *transporting*
$$x\in\mathcal{X}$$ to $$y\in\mathcal{Y}$$. When $$\mathcal{X}=\mathcal{Y}$$, an interesting and
natural choice of the cost function is simply the distance function $$d$$ of the metric space,
perhaps raised to some power $$p\geq 1$$. For probability spaces
$$(\mathcal{X},\mu),(\mathcal{Y},\nu)$$, we say that a coupling $$(X, Y)$$ is optimal if it
minimizes the expected transportation cost:

$$
(X, Y) = \inf_{X'\sim\mu, Y'\sim\nu}\mathbf{E}c(x, y) = \inf_{X'\sim\mu, Y'\sim\nu}\int_{\mathcal{X}\times\mathcal{Y}}c(x,
    y)d\pi(x, y)
$$

where $$(X, Y)\sim\pi$$. What we want to show now is that such an optimal coupling *always exists*
under some assumptions of the probability spaces and the cost function.

Let's begin by stating out assumptions:

1. $$(\mathcal{X}, \mu), (\mathcal{Y},\nu)$$ are Polish probability spaces. In other words,
   $$\mathcal{X},\mathcal{Y}$$ are complete, separable metric spaces.
1. There exists upper semicontinuous functions $$a: \mathcal{X}\to\mathbf{R}\cup\{-\infty\}$$, $$b:
   \mathcal{Y}\to\mathbf{R}\cup\{-\infty\}$$ where $$a$$ is integrable w.r.t. $$\mu$$ and $$b$$ is
   integrable w.r.t. $$\nu$$, such that the cost function $$c$$ satisfies $$c(x, y)\geq a(x) +
   b(y)$$ for each $$x, y$$.
1. The cost function $$c:\mathcal{X}\times\mathcal{Y}\to\mathbf{R}\cup\{\infty\}$$ is lower semicontinuous.

To begin, we'd like to show that $$\Pi(\mu,\nu)$$, the space of couplings of $$(\mu,\nu)$$, is
compact. Prokhorov's theorem, which shows a relationship between compactness and tightness in Polish
spaces, will be really handy here. The theorem states the following:

<div class="theorem">
<b>Theorem (Prokhorov)</b>: If \(\mathcal{X}\) is a Polish space,then a set
\(\mathcal{P}\subset\mathscr{P}(\mathcal{X})\) is precompact (has compact closure) w.r.t. the weak topology if
and only if it is tight; that is, for any \(\mu\in\mathcal{P}\) and \(\varepsilon>0\), there is a
compact set \(K_\varepsilon\subset\mathcal{X}\) such that \(\mu(\mathcal{X}\setminus
      K_\varepsilon)\leq\varepsilon\).
</div>

Moving forward, we'd like to show that if $$\mathcal{P},\mathcal{Q}$$ are subsets of
$$\mathscr{P}(\mathcal{X}),\mathscr{P}(\mathcal{Y})$$ respectively, then the set $$\Pi(\mathcal{P},
\mathcal{Q})$$ of all couplings with marginals in $$\mathcal{P}$$ and $$\mathcal{Q}$$ is itself
tight in $$\mathscr{P}(\mathcal{X}\times\mathcal{Y})$$. Then, by Prokhorov's theorem, we will have
shown that $$\Pi(\mathcal{P},\mathcal{Q})$$ has a compact closure.

Fix some $$\varepsilon>0$$. Since $$\mathcal{P},\mathcal{Q}$$ are tight by assumption, there exist
sets $$K_{\varepsilon/2},L_{\varepsilon/2}$$ such that regardless of $$\mu\in\mathcal{P}$$ and
$$\nu\in\mathcal{Q}$$, we have $$\mu(\mathcal{X}\setminus K_{\varepsilon/2})\leq\varepsilon/2$$ and
similarly $$\nu(\mathcal{Y}\setminus L_{\varepsilon/2})\leq\varepsilon/2$$. Then, for any coupling
$$(X, Y)$$ of $$(\mu,\nu)$$, we have

$$
\Pr((X, Y)\not\in K_{\varepsilon/2}\times L_{\varepsilon/2})\leq\Pr(X\not\in K_{\varepsilon/2}) +
\Pr(Y\not\in L_{\varepsilon/2}) = \varepsilon/2 + \varepsilon/2 = \varepsilon
$$

Therefore, for any probability measure $$\pi\in\Pi(\mathcal{P}\times\mathcal{Q})$$ and
$$\varepsilon>0$$, we have $$\pi(\mathcal{X}\times\mathcal{Y}\setminus
K_{\varepsilon/2},L_{\varepsilon/2})\leq\varepsilon$$, so $$\Pi(\mathcal{P}, \mathcal{Q})$$ is tight
and therefore precompact, by Prokhorov's theorem.

In particular, $$\{\mu\}, \{\nu\}$$ are both clearly tight sets. Therefore, $$\Pi(\mu,\nu)$$, the
space of all couplings of $$(\mu, \nu)$$ is precompact. In fact, it can be shown that $$\Pi(\mu,
\nu)$$ is closed, so it is equal to its closure, and it is therefore compact.

Since $$\Pi(\mu,\nu)$$ is compact, every
sequence in $$\Pi(\mu,\nu)$$ has a convergent subsequence. Let $$(\pi_i)_{i\in\mathbf{N}}$$ denote a
convergent sequence in $$\Pi(\mu,\nu)$$ such that the "transport cost" $$\int cd\pi_i$$ converges to
its infimum value. Then we can assume that
$$\pi_i\overset{i\to\infty}{\longrightarrow}\pi\in\Pi(\mu,\nu)$$. To ensure that $$\pi$$ itself is
the law of an optimal coupling, we need to show that $$\int cd\pi = \liminf_{k\to\infty}\int
cd\pi_k$$.

Let there be some upper semicontinuous function $$h$$ that is integrable w.r.t. each $$\pi_i$$ and
$$\pi$$, such that $$c\geq h$$. Then $$\tilde{c} = c - h$$ is a non-negative, lower semicontinuous function that
is integral w.r.t. each $$\pi_i$$ and $$\pi$$ as well. Since $$\tilde{c}$$ is a lower semicontinuous
function to the nonnegtive real numbers, we construct a sequence of continuous functions
$$(\tilde{c}_k)_{k\in\mathbf{N}}$$ that converges to $$\tilde{c}$$, such that
$$\tilde{c}_{k+1}\geq\tilde{c}_k$$. Using the monotone convergence theorem, we have

$$
\int_{\mathcal{X}\times\mathcal{Y}}\tilde{c}(x, y)d\pi =
\int_{\mathcal{X}\times\mathcal{Y}}\lim_{k\to\infty}\tilde{c}_k(x, y)d\pi =
\lim_{k\to\infty}\int_{\mathcal{X}\times\mathcal{Y}}\tilde{c}_k(x, y)d\pi
$$

Using the monotone convergence theorem again, we have

$$
\lim_{k\to\infty}\int_{\mathcal{X}\times\mathcal{Y}}\tilde{c}_k(x, y)d\pi =
  \lim_{k\to\infty}\int_{\mathcal{X}\times\mathcal{Y}}\tilde{c}_k(x, y)\lim_{i\to\infty}d\pi_i
  =\lim_{k\to\infty}\lim_{i\to\infty}\int_{\mathcal{X}\times\mathcal{Y}}\tilde{c}_k(x, y)d\pi_i
$$

Ultimately, this leads us to the conclusion that

$$
\int_{\mathcal{X}\times\mathcal{Y}}cd\pi = \lim_{k\to\infty}\lim_{i\to\infty}\int_{\mathcal{X}\times\mathcal{Y}}\tilde{c}_kd\pi_i \leq \liminf_{i\to\infty}\int_{\mathcal{X}\times\mathcal{Y}}\tilde{c}d\pi_i
$$

So, since $$c(x, y)\geq a(x) + b(y)$$, letting $$h(x, y) = a(x) + b(y)$$ we see that $$\int
(c - h)d\pi\leq\liminf_{i\to\infty}\int(c-h)d\pi_i$$, so the coupling defined by $$\pi$$ minimizes
the transportation cost! Of course, this is relative to the cost function $$c-h$$ rather than $$c$$.
According to Villani, we can usually let $$a = b = 0$$ so that $$c - h = c$$.

## Summary (TL;DR)

Let's summarize what we just did.

1. We first showed that $$\Pi(\mu,\nu)$$, the space of couplings of $$(\mu,\nu)$$ is compact using
   Prokhorov's theorem.
1. Since $$\Pi(\mu,\nu)$$ is compact, we deduced that we can construct a convergent sequence of
   couplings with laws $$\pi_i$$ such that the transportation cost converges to its infimum, and we
   say $$\pi_i\longrightarrow\pi$$.
1. Using the monotone convergence theorem, we show that $$\int
   cd\pi\leq\liminf_{i\to\infty}\int cd\pi_i$$, which proves that the distribution $$\pi$$ with marginals
   $$\mu,\nu$$ minimizes the transportation cost.

# References

I learned the ideas behind this proof from the great textbook by Villani, ["Optimal Transport: Old and New"](https://www.semanticscholar.org/paper/Optimal-Transport%3A-Old-and-New-Villani/ca73ce2623aa93bf32b88f7fb998af9576aed20f) (Theorem 4.1).
