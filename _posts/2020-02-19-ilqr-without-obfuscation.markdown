---
layout: post
title: iLQR Without Obfuscation
featured-img: ilqr
categories: control
---
<link rel="stylesheet" href="/assets/css/me_photo_style.css">

# What's Wrong With the iLQR Literature?

The iLQR optimal control algorithm could look quite daunting to those that are
unfamiliar with it. I hope I can convince you that iLQR is actually fairly
simple, it just happens to be one of the most poorly documented algorithms out
there. I don't mean to say there's a shortage of documentation on this topic --
rather, every document that I've read that attempts to explain or describe iLQR
makes a severe mistake that obfuscates the algorithm tremendously. Maybe my
reading comprehension needs some work, but for what it's worth, this is my
attempt to explain how iLQR works from the ground up, without the distracting
flaw that I previously mentioned.

The issue that I believe obfuscates the iLQR algorithm in the literature that
I've read is the abundant slew of comparisons to LQR. Using LQR to motivate iLQR
is *almost* like using SAT to motivate integer factorization. Admittedly, now
that I understand iLQR I can see the connection, but I truly believe it's
beneficial to extrapolate that connection from an algorithm that you understand
than to try to force it to begin with. So, if you don't know what LQR is, don't
worry. If you do know what it is, pretend that you don't. My derivation of iLQR
won't make any connection to LQR at all.

# An Overview of iLQR

Before we begin, let's clear up what iLQR is actually responsible for. The
acronym stands for iterative linear quadratic regulator, which I won't talk
about anymore. Its purpose is to perform finite-horizon *trajectory
optimization* for nonlinear systems. Essentially, the algorithm is given the
initial state of a system with known (possibly nonlinear) dynamics, the target
state to reach at the end of a given time horizon, and an initial "nominal
trajectory", and iLQR iteratively improves this trajectory until it is optimized
with respect to some cost function. The algorithm is subject to the following
constraints:

* The system must be discretized in time
* The cost function will be approximated with second degree Taylor series
approximations

An abstract overview of the main steps in the algorithm is given below:

~~~python
def ilqr(x0, target, dynamics, state_trajectory, control_trajectory):
  converged = False
  while not converged:
    gains = BackwardUpdate(state_trajectory, control_trajectory, target)
    states, controls, loss = ForwardRollout(state_trajectory, control_trajectory, gains)
    if |controls - control_trajectory| < threshold:
      converged = True
    state_trajectory = states
    control_trajectory = controls
~~~

So basically we need to figure out what `ForwardRollout` and `BackwardUpdate`
do. Readers that are familiar with machine learning could think of these like
forward and backward passes in a neural network, if that helps. Essentially, the
backward update tells us how to change the trajectory to most effectively
minimize its cost, and the forward pass applies the updates and computes a new
trajectory.

# The Math

## Some Notation and Modeling

Let $$x_t\in\mathbf{C}^{n_x}$$ denote the state vector at time $$t$$,
and $$u_t\in\mathbf{C}^{n_u}$$ denote the control signal at time $$t$$.
We describe the dynamics of the system according to

$$
x_{t+1} = f(x_t,u_t)
$$

Moreover, we define a *cost-to-go* function as follows,

$$
V_{t:T}(x_t) = \min_{u_{t:T}}\sum_{k=t}^T\ell(x_k, u_k)
$$

where $$V_{t:T}$$ designates the cumulative cost in the trajectory starting at
time $$t$$ and ending at time $$T$$, and $$\ell$$ is a per-step cost function.
As we mentioned previously, iLQR approximates the cost function with a second
degree Taylor approximation. So, rather than requiring this approximation, we'll
define a second-order cost function as follows:

$$
\ell(x_t, u_t) = \frac{1}{2}x^{\top}Q\bar{x} + \frac{1}{2}u^\top R\bar{u}
$$

where $$Q\in\mathbf{R}^{n_x\times n_x}$$ and $$R\in\mathbf{R}^{n_u\times n_u}$$
decide how to penalize trajectories based on their states and controls. With
this quadratic parameterization of the cost function, we don't need to worry
about the Taylor series approximation.

## Putting the "Optimization" in Trajectory Optimization

In order to optimize the cost of trajectories, iLQR makes local linear
approximations of the dynamics of the system. To compute this, we analyze the
perturbation of the next state $$\delta x_{t+1}$$ due to a small perturbation in
state $$\delta x_t$$ and control $$\delta u_t$$:

$$
x_{t+1} + \delta x_{t+1} = f(x + \delta x_t, u + \delta u_t) = f(x_t, u_t) +
\frac{\partial f}{\partial x}\bigg\rvert_{x_t, u_t}(x - x_t) + \frac{\partial
  f}{\partial u}\bigg\rvert_{x_t,u_t}(u - u_t)
$$

where $$\delta x_t\triangleq x - x_t$$ and $$\delta u_t\triangleq u - u_t$$.
Here, the $$x,u$$ refer to states and controls in a *proposed* trajectory, while
$$x_t, u_t$$ refer to states and controls in the trajectory that is being
improved (referring to the pseudocode above, $$x, u$$ would be in `states` and
`controls`, while $$x_t, u_t$$ would be in `state_trajectory` and
`control_trajectory`). Also, noting that $$x_{t+1}\equiv f(x_t, u_t)$$, we have

$$
\delta x_{t+1} = A_t\delta x_t + B_t\delta u_t
$$

where $$A_t\in\mathbf{C}^{n_x\times n_x},B_t\in\mathbf{C}^{n_x,n_u}$$ are the
Jacobians of $$f$$ with respect to $$x$$ and $$u$$ respectively, evaluated at
$$(x_t, u_t)$$.

With that out of the way, we can think about how to optimize the cost-to-go of
our trajectory. Firstly, note the recursive property of the cost to go:

$$
V_{t:T}(x_t) = \min_{u_t}[\ell(x_t, u_t) + V_{t+1:T}(f(x_t, u_t))]
$$

This is called the Hamilton-Jacobi-Bellman equation, and it is beautiful
(reinforcement learning folk are obligated to agree). The minimum over controls
is taken here since the cost-to-go should represent the optimal cost of
completing the trajectory from state $$x_t$$. Keeping with reinforcement
learning style notation, we'll define $$Q_{t:T}(x_t, u_t) = \ell(x_t, u_t) +
V_{t+1:T}(f(x_t, u_t))$$, so $$V_{t:T}(x_t) = \min_{u_t}Q_{t:T}(x_t, u_t)$$. In
order to minimize the cost-to-go with respect to controls, we approximate the
loss function by a quadratic, as mentioned previously. Now we'll analyze how the
cost-to-go is perturbed due to a small change in $$x_t$$, using a second order
Taylor expansion (note, the Jacobians will be very simple due to the quadratic
costs):

$$
\begin{equation}
V_{t:T}(x_t) + \delta V_{t:T} = V_{t:T}(x_t + \delta x_t) = V_{t:T}(x_t) +
\frac{\partial V_{t:T}}{\partial x}\bigg\rvert_{x_t}\delta x + \frac{1}{2}\delta
x^\top\frac{\partial^2 V_{t:T}}{\partial x^2}\delta x_t
\end{equation}
$$

$$
\begin{equation}\tag{dV}\label{eq:dV}
\therefore \delta V_{t:T} \triangleq M_t^\top \delta x_t + \frac{1}{2}\delta
x_t^\top N_t\delta x_t
\end{equation}
$$

To compute the Jacobians of $$V_{t:T}$$, we must compute the Jacobians of
$$Q_{t:T}$$. In a similar fashion, we have

$$
Q_{t:T}(x_t, u_t) + \delta Q_{t:T} = Q(x_t + \delta x_t, u_t + \delta u_t)\\
\hspace{2cm}= Q_{t:T}(x_t, u_t) + \frac{\partial Q_{t:T}}{\partial x}\bigg\rvert_{x_t,u_t}\delta x_t + \frac{\partial Q_{t:T}}{\partial
    u}\bigg\rvert_{x_t,u_t}\delta u_t\\
\hspace{3cm} + \frac{1}{2}\delta x^\top\frac{\partial^2Q_{t:T}}{\partial
  x^2}\bigg\rvert_{x_t, u_t}\delta x + \frac{1}{2}\delta
  u_t^\top\frac{\partial^2Q_{t:T}}{\partial u^2}\bigg\rvert_{x_t,u_t}\delta
  u_t\\
\hspace{3.2cm} + \frac{1}{2}\delta u_t^\top\frac{\partial^2 Q_{t:T}}{\partial
  u\partial x}\bigg\rvert_{x_t, u_t}\delta x_t + \frac{1}{2}\delta
  x_t^\top\frac{\partial^2 Q_{t:T}}{\partial x\partial u}\bigg\rvert_{x_t,u_t}\delta u_t
$$

Phew! Believe me, that was not any more complicated than a second order Taylor
series expansion. Thankfully, we can write this in matrix form, where the
Jacobians will be replaced with cleaner symbols:

<div>
$$
\begin{equation}
\delta Q_{t:T} = \frac{1}{2}\begin{bmatrix} \delta x_t\\\delta
u_t\end{bmatrix}^\top\begin{bmatrix}Q^{(t)}_{xx} & Q^{(t)}_{xu}\\ Q^{(t)}_{ux} &
Q^{(t)}_{uu}\\\end{bmatrix}\begin{bmatrix}\delta x_t\\\delta u_t\\\end{bmatrix} +
\begin{bmatrix}Q^{(t)}_x\\Q^{(t)}_u\\\end{bmatrix}^\top\begin{bmatrix}\delta x_t\\\delta
u_t\\\end{bmatrix}\tag{dQ}\label{eq:dQ}
\end{equation}
$$
</div>

The matrices $$Q^{(t)}_{xx}, Q^{(t)}_{uu}, Q^{(t)}_{ux}, Q^{(t)}_{xu}, Q^{(t)}_x, Q^{(t)}_u$$ are derived below:

<div>
$$
\begin{align}
Q^{(t)}_x &\triangleq \frac{\partial Q_{t:T}}{\partial x}\bigg\rvert_{x_t, u_t} =
\frac{\partial\ell}{\partial x}\bigg\rvert_{x_t, u_t} + \frac{\partial
  V_{t+1:T}}{\partial x}\bigg\rvert_{x_t, u_t} = Qx_t + M_{t+1}^\top A_t\\
Q^{(t)}_u &\triangleq \frac{\partial Q_{t:T}}{\partial u}\bigg\rvert_{x_t, u_t} =
\frac{\partial\ell}{\partial u}\bigg\rvert_{x_t, u_t} + \frac{\partial
  V_{t+1:T}}{\partial u}\bigg\rvert_{x_t, u_t} = Ru_t + M_{t+1}^\top B_t\\
Q^{(t)}_{xx} &\triangleq \frac{\partial^2 Q_{t:T}}{\partial x^2}\bigg\rvert_{x_t,
  u_t} = \frac{\partial^2\ell}{\partial x^2}\bigg\rvert_{x_t,u_t} +
  \frac{\partial^2 V_{t+1:T}}{\partial x^2}\bigg\rvert_{x_t, u_t} = Q + A_t^\top
  N_{t+1}A_t\\
Q^{(t)}_{uu} &\triangleq \frac{\partial^2 Q_{t:T}}{\partial u^2}\bigg\rvert_{x_t,
  u_t} = \frac{\partial^2\ell}{\partial u^2}\bigg\rvert_{x_t,u_t} +
  \frac{\partial^2 V_{t+1:T}}{\partial u^2}\bigg\rvert_{x_t, u_t} = R + B_t^\top
  N_{t+1}B_t\\
Q^{(t)}_{xu} = Q^{(t)\top}_{ux} &\triangleq \frac{\partial^2Q_{t:T}}{\partial u\partial
  x}\bigg\rvert_{x_t, u_t} = \frac{\partial^2\ell}{\partial u\partial
    x}\bigg\rvert_{x_t, u_t} + \frac{\partial^2 V_{t+1:T}}{\partial u\partial x}
    = B_t^\top N_{t+1}A
\end{align}
$$
</div>

You may be concerned about the dependence of these matrices on the $$M$$ and $$N$$
matrices "from the future", but fortunately we can solve for these matrices
backwards in time (hence, the backwards update step). Note that at $$t=T$$ (the
end of the horizon), the cost to go is $$V_{T:T}(x_T) = \ell(x_T) = x_T^\top Q_f
x_T$$, where we optionally specify another cost matrix $$Q_f$$ to distinguish
"transient" penalties from the penalty on the final state. Therefore, by
\eqref{eq:dV}, we have
$$M_{T} = Q_f(x_T - x^*)$$ for target state $$x^*$$, and $$N_T = Q_f$$. With
this edge case, we can compute the Jacobian matrices backward from $$t=T$$ to
$$t=0$$.

To minimize the cost-to-go $$V_{t:T}(x_t)$$, we must solve for $$\delta u_t =
\arg\min_{\delta u_t}\delta Q_{t:T}$$. We proceed to do this using standard
calculus optimization techniques. Due to Bellman's principle of optimality, we
can simply optimize $$\delta u_t$$ individually for each timestep. Referring to
\eqref{eq:dQ}, have

<div>
$$
\begin{align}
\delta u_t &= \arg\min_{\delta u_t}\delta Q_{t:T}\\
0 &= \frac{\partial\delta Q_{t:T}}{\partial \delta u_t}\\
0 &= \frac{\partial}{\partial\delta u_t}\left[\frac{1}{2}\left(2\delta
    u_t^\top Q^{(t)}_{ux}\delta x_t + \delta u_t^\top Q^{(t)}_{uu}\delta u_t + Q^{(t)}_u\delta u_t\right) + Q^{(t)}_u\delta u_t\right]\\
0 &= Q^{(t)}_{ux}\delta x_t + Q^{(t)}_{uu}\delta u_t + Q_u^{(t)}\\
\delta u_t &= -(Q^{(t)}_{uu})^{-1}Q_{ux}^{(t)}\delta x_t -
(Q^{(t)}_{uu})^{-1}Q_u^{(t)}
\end{align}
$$
</div>

Notably, we have

$$
\begin{equation}
\delta u_t = K_t\delta x_t + d_t\tag{du}\label{eq:du}
\end{equation}
$$

where $$K_t = -(Q_{uu}^{(t)})^{-1}Q^{(t)}_{ux}$$ and $$d_t =
-(Q_{uu}^{(t)})^{-1}Q^{(t)}_u$$, so optimal changes to controls at each step are
affine transformations of the measured error in state (we will discuss how this
state error is calculated soon).

Recall that in order to compute these gains, we need the $$M$$ and $$N$$ matrices
for each timestep. First, substitute the $$\delta u_t$$ computed in
\eqref{eq:du} into \eqref{eq:dQ}:

<div>
$$
\begin{equation}
\delta Q_{t:T} = \frac{1}{2}\begin{bmatrix} \delta x_t\\K_t\delta x_t + d_t\\\end{bmatrix}^\top\begin{bmatrix}Q^{(t)}_{xx} & Q^{(t)}_{xu}\\ Q^{(t)}_{ux} &
Q^{(t)}_{uu}\\\end{bmatrix}\begin{bmatrix}\delta x_t\\K_t\delta x_t + d_t\\\end{bmatrix} +
\begin{bmatrix}Q^{(t)}_x\\Q^{(t)}_u\\\end{bmatrix}^\top\begin{bmatrix}\delta x_t\\
K_t\delta x_t + d_t\\\end{bmatrix}
\end{equation}
$$
</div>

Recall that $$M_t$$ is the term that is linear in $$\delta x_t$$ in \eqref{eq:dV}.
Since $$\frac{\partial V_{t:T}}{\partial\delta x_t} = \frac{\partial}{\partial\delta
  x_t}\min_{\delta u_t}Q_{t:T}(\delta x_t,\delta u_t) = \delta Q_{t:T}$$, $$M_T$$
  is obtained by extracting the linear terms in $$\delta x_t$$ from the matrix
  equation above:

$$
\begin{equation}
M_t = Q_{ux}^{(t)\top}d_t + K^\top_tQ^{(t)}_{uu}d_t + Q_x^{(t)} +
K^\top_tQ^{(t)}_u\tag{Mt}\label{eq:Mt}
\end{equation}
$$

Similarly, since $$N_t$$ is the term that is quadratic in $$\delta x_t$$, we have

$$
\begin{equation}
N_t = Q^{(t)}_{xx} + K_t^\top Q^{(t)}_{ux} + Q_{ux}^{(t)\top} K_t + K_t^\top
Q^{(t)}_{uu}K_t\tag{Nt}\label{eq:Nt}
\end{equation}
$$

By starting at $$t=T$$ and working backwards, we can compute all of the $$K_t$$
and $$d_t$$ vectors. This is the job of `BackwardUpdate` in the pseudocode
above.

## Generating Rollouts

With the `BackwardUpdate`, we know how to adjust controls given some error in
state. Now we discuss how this is implemented in iLQR -- this is the
`ForwardRollout` subroutine described in the pseudocode above.

Recall that `ForwardRollout` takes as input a trajectory of states, a trajectory
of controls, and a target state. The trajectory of states corresponds to the
previous rollout that was done (we'll talk about [how to initialize the first
rollout later](#the-initial-trajectory), and the trajectory of controls
corresponds to the controls used in the previous trajectory. Now,
`ForwardRollout` works as follows:

* Let $$\bar{x}_t$$ denote the previous state trajectory, $$\bar{u}_t$$ the
previous control trajectory
* Let $$x_t\leftarrow x_0$$, the initial state
* Let $$L=0$$, the loss
* For $$t$$ from $$0$$ to $$T-1$$

  1. Compute $$\delta x_t$$: $$\delta x_t\leftarrow x_t - \bar{x}_t$$
  1. Compute $$\delta u_t$$ using \eqref{eq:du}: $$\delta u_t\leftarrow K_t\delta x_t + d_t$$
  1. Update controls: $$u_t\leftarrow \bar{u}_t + \delta u_t$$
  1. Aggregate loss: $$L\leftarrow L + x_t^\top Qx_t + u_t^\top u_t$$
  1. Get next state from dynamics: $$x_{t+1}\leftarrow f(x_t, u_t)$$

* $$L\leftarrow L + x_T^\top Q_fx_T$$, the final state penalty
* Return $$(\{x_t\}_{t=0}^{T}, \{u_t\}_{t=0}^{T-1}, L)$$

And that's it! Now we've discussed all of the essential math to understand how
the iLQR algorithm works. We simply alternate between computing the gains to
improve the controls and improving the controls to generate new trajectories,
until the controls stop changing (this sounds suspiciously similar to Value
Iteration, but there are some key differences). I think it's finally time to see
a demo. I implemented an iLQR algorithm to swing up an inverted pendulum on a
cart. The system has a four-dimensional state space consisting of position,
velocity, pole angle, and angular velocity, and the control signal is a scalar
force applied to the cart. The swingup task is nonlinear, so linear algorithms
tend to struggle with this task.

<div class="center-image">
  <video autoplay="autoplay" loop="loop" width="400px" height="auto"
  class="center-image">
    <source src="/assets/img/ilqrswingup.webm" type="video/webm">
  </video>
</div>

Calm down, Boston Dynamics. I gotta finish grad school before I can help you
make Spot as good as this.

# Implementation Details and Tricks

While the optimization part of iLQR is derived above, some implementation
details were omitted. Moreover, in implementing this algorithm, I found that
some "tricks" were either essential or very helpful to yield successful
controllers. Read on to learn more about this.

## The Initial Trajectory

Before iLQR even begins, we must have some initial trajectory of states and
controls. It is natural to wonder how one concocts such an initial trajectory.
Unfortunately, in the general case, there is no single "right way" to do this,
unless somehow you know of a trajectory that is somewhat close to optimal.
Otherwise, to the best of my knowledge, we can use some heuristics to generate a
trajectory of controls and initialize the state trajectory according to the
dynamics following those controls. Here's some ideas for generating the control
trajectory that I played with:

* $$u_t = 0$$: This one was suggested by Li and Todorov (the inventors of iLQR).
Intuitively, this might make training a little slow if the starting state is in
a stable equilibrium.
* $$u_t = \alpha\sin(\omega t), \alpha,\omega\in\mathbf{R}$$: This is likely to
make more sense for scalar controls, but it ensures that a nice range of
controls will be attempted in the initial trajectory. I found this worked
slightly better than the "zero initialization" for the swingup.
* $$u_t\sim\mathcal{N}(0, \Sigma), \Sigma=\Sigma^\top\in\mathbf{R}^{n_u\times
  n_u}_{\geq 0}\succ 0$$: Zero-centered random controls. This one was the best during my
  experiments.

Since iLQR is an iterative nonlinear optimization algorithm, initialization can
definitely have an affect on the final controller. Playing with various
initialization schemas would be wise.

## Gradient Clipping

I cannot emphasize enough how **essential** this trick was. When tuning my iLQR
implementation, I noticed that the controller would fail to swing up the pole
because it wouldn't accelerate fast enough, which I attributed to penalizing
velocities and control magnitudes too much. But when I would try reducing these
penalties, iLQR would outright explode: velocities and angular velocities would
shoot up, and the loss would skyrocket, leading to a completely abysmal final
trajectory. After further inspection, I noticed that $$|\delta u_t|$$
was blowing up. To prevent this from happening, I implemented the following
adjustment

$$
\delta u_t = \text{sgn}(K_t\delta x_t + d_t)\min(K_t\delta x_t + d_t, \Delta)
$$

where $$\text{sgn}(x) = |x|/x$$, which essentially clips the control updates to
some magnitude $$\Delta$$. Note that while this transformation is defined for
scalar controls, an equivalent transformation for vector controls can be derived
without much trouble. Upon employing this trick, these divergence issues ceased
**immediately**. This allowed me to be much more flexible with my cost
functions, and ultimately allowed me to tune my iLQR to a competent solution in
a finite amount of time.

This trick *does* impose an additional hyperparameter, however in my experiments
I found that this hyperparameter was very easy to tune. Moreover, when
$$\Delta$$ is "too low", the only consequence is that iLQR won't make progress
as quickly, which normally isn't such a big deal.

Alternatively, line search methods or adjustable (perhaps even adaptive) $$\delta
u_t$$ step sizes could potential help solve this problem. However, these methods
involve hyperparameters of their own, and I still believe gradient clipping
could be necessary -- for instance, say you update $$u_t\leftarrow\bar{u}_t +
\alpha\delta u_t$$ for some small step size $$\alpha$$, but $$||\delta u_t||$$
is enormous. The small step size would dampen the effect of this, but without
forcing *very* small step sizes, this can't eliminate the divergence issue I was
experiencing. Combining gradient clipping with line search or adaptive step
sizes would likely be the most robust strategy, however gradient clipping on its
own was more than sufficient enough for my purposes.

## Scaling Cost Function Weights

This trick allowed me to design my cost functions with much more intuition,
which proved to be very helpful. Firstly, if you distinguish $$Q_f$$ from $$Q$$,
the influence of the penalties governed by $$Q$$ to the total trajectory cost
will devastatingly outweigh that of $$Q_f$$ for long time horizons. It might
even be wise to let $$Q = 0$$ if you mostly care about penalizing state at the
end of the trajectory. Otherwise, I found that scaling $$Q$$ by a factor of
$$1/T$$ helped design $$Q$$ and $$Q_f$$ quite a bit.

Moreover, if the state variables are measured on different scales, this should
be compensated for. Let's say you want to penalize deviations in angle twice as
much as deviations in velocity. Angles are confined to $$[0,2\pi]$$, while
velocities could be much larger, say in the range $$[-v_{\max},v_{\max}]$$.
Therefore it doesn't suffice to assign twice the weight to deviations in angle,
since a deviation of $$\pi$$ radians is most likely much more significant than a
deviation of $$\pi$$ in velocity. Let $$r_i$$ denote the range of state $$i$$,
which would be $$2\pi$$ for angles and $$2v_{\max}$$ for the velocity
example. To compensate for changes of scale, let $$\tilde{Q}$$ be the
"intuitive" cost matrix (that is, one where you assign twice the weight to a
state deviation you want to penalize twice as much), and compute $$Q$$ according
to

<div>
$$
Q = \begin{pmatrix}r_1^{-1} & 0 & \dots & 0\\0 & r_2^{-1} & \dots & 0\\0 & 0 &
\ddots & \vdots\\0 & 0 & \dots & r_n^{-1}\\\end{pmatrix}\tilde{Q}
$$
</div>

## Computing Jacobians the Lazy Way

In the derivation of the iLQR optimization math, we needed to compute a bunch of
Jacobians. This doesn't necessarily cause too much trouble, since given the
dynamics model we can simply write out the Taylor expansions on paper and then
code them up. I can think of two reasons why we might prefer a workaround for
this:

1. The state-control space is large, so deriving the Jacobians could be very
   time consuming and/or error-prone (or you're simply too lazy to derive the
   Jacobians)
1. You don't actually have an explicit dynamics model (for instance, perhaps you
   have a powerful simulator instead)

Fortunately, we can automatically approximate Jacobians using the method of
[finite differences](https://en.wikipedia.org/wiki/Finite_difference_method),
which is so easy to implement that it's almost irresistible. The method
essentially works by perturbing each variable to be differentiated by some small
$$\epsilon>0$$ in each direction and examining how that changes the output, like
so:

$$
\frac{\partial f}{\partial x_i}\bigg\rvert_{\mathbf{x}}\approx
\frac{f(x_1,\dots, x_i+\epsilon,\dots x_n) - f(x_1,\dots, x_i - \epsilon,\dots
    x_n)}{2\epsilon}
$$

If the function $$f$$ is "smooth enough", we can approximate the Jacobian
arbitrarily well by shrinking $$\epsilon$$. For example, approximating the
Jacobian with respect to states for a dynamics function $$f:\mathbf{R}^n\times
\mathbf{R}^{n_u}\to\mathbf{R}^n$$ can be achieved as follows:

~~~python
def finite_differences(dynamics, x, u, eps):
  n = x.shape[0]
  jacobian = np.zeros((n, n))
  for i in range(n):
    dxp = x.copy()
    dxp[i] += eps
    x_inc = dynamics(dxp, u)
    dxm = x.copy()
    dxm[i] -= eps
    x_dec = dynamics(dxm, u)
    jacobian[:, i] = (x_inc - x_dec) / (2 * eps)
  return jacobian
~~~
