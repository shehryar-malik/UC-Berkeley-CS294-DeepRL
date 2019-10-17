# CS294-112 HW 2: Policy Gradient

Dependencies:
 * Python **3.5**
 * Numpy version **1.14.5**
 * TensorFlow version **1.10.5**
 * MuJoCo version **1.50** and mujoco-py **1.50.1.56**
 * OpenAI Gym version **0.10.5**
 * seaborn
 * Box2D==**2.3.2**

Before doing anything, first replace `gym/envs/box2d/lunar_lander.py` with the provided `lunar_lander.py` file.

The only file that you need to look at is `train_pg_f18.py`, which you will implement.

See the [HW2 PDF](http://rail.eecs.berkeley.edu/deeprlcourse/static/homeworks/hw2.pdf) for further instructions.

# Solutions

**Problem 1**

**Part (a)**

From the law of iterated expectations it follows that:
$$
\begin{align}
& \mathbb{E}_{\tau \sim P_\theta(\tau)}\left[\nabla_\theta\log \Pi_\theta(a_t \vert s_t)(b(s_t))\right]\\
&= \mathbb{E}_{(s'_t,a'_t)\sim P_\theta(s_t',a_t')}\left[\mathbb{E}_{\tau \sim P_\theta(\tau \vert s_t=s'_t,a_t=a'_t )}\left[\nabla_\theta\log P_\theta(a_t \vert s_t)(b(s_t))\right]\right]\\
&= \int P_\theta(s'_t,a'_t)\int P_\theta(\tau\vert s_t=s'_t,a_t=a'_t)\nabla_\theta\log P_\theta(a_t \vert s_t)(b(s_t))d\tau d(s'_t,a'_t)
\end{align}
$$


where ' is used to differentiate between the state-action marginals in the outer and inner expectations. Note that:

$$
P(\tau \vert s_t=s'_t,a_t=a'_t) = \left\{
\begin{align}
&0 && s_t \neq s'_t \text{ or } a_t \neq a'_t\\
&P(\tau \vert s_t=s'_t,a_t=a'_t) && \text{otherwise}
\end{align}
\right.
$$


Consequently, we have:
$$
\begin{align}
& \int P_\theta(s'_t,a'_t)\int P_\theta(\tau/s_t,a_t\vert s'_t,a'_t)\nabla_\theta P_\theta(a'_t \vert s'_t)(b(s'_t))d(\tau/s_t,a_t)d(s'_t,a'_t)\\
&= \int P_\theta(s'_t,a'_t)\int P_\theta(\tau/s_t,a_t\vert s'_t,a'_t)d(\tau/s_t,a_t)\nabla_\theta\log P_\theta(a'_t \vert s'_t)(b(s'_t))d(s'_t,a'_t)\\
&= \int P_\theta(s'_t,a'_t)\nabla_\theta \log P_\theta(a'_t \vert s'_t)(b(s'_t))d(s'_t,a'_t)\\
&= \int P_\theta(s'_t)P_\theta(a'_t\vert s'_t)\nabla_\theta \log P_\theta(a'_t \vert s'_t)(b(s'_t))d(s'_t,a'_t)\\
&= \int P_\theta(s'_t)\nabla_\theta P_\theta(a'_t \vert s'_t)(b(s'_t))d(s'_t,a'_t)\\
&= \int P_\theta(s'_t)b(s'_t)\nabla_\theta\int P_\theta(a'_t \vert s'_t)da'_tds'_t\\
&= \int P_\theta(s'_t)b(s'_t)\nabla_\theta(1)ds'_t\\
&= 0\\
\end{align}
$$
**Part (b)**

(a) Because of Markov's assumption (i.e. the future is independent of the past given the present)

(b) From the law of iterated expectations it follows that:
$$
\begin{align}
& \mathbb{E}_{\tau \sim P_\theta(\tau)}\left[\nabla_\theta\log P_\theta(a_t \vert s_t)(b(s_t))\right]\\
&= \mathbb{E}_{(s'_{1:t},a'_{1:t-1}) \sim P_\theta(s'_{1:t},a'_{1:t-1})}\left[\mathbb{E}_{\tau \sim P_\theta(\tau\vert s'_{1:t},a'_{1:t-1})}\left[\nabla_\theta\log P_\theta(a_t \vert s_t)(b(s_t))\right]\right]
\end{align}
$$
Using the same logic as in part (a) we have:
$$
\begin{align}
& \int P_\theta(s'_{1:t},a'_{1:t-1})\int P_\theta(s_{t+1:T},a_{t:T}\vert s'_{1:t},a'_{1:t-1})d(s_{t+1:T},a_{t:T})\nabla_\theta\log P_\theta(a_t \vert s'_t)(b(s_t))d(s'_{1:t},a'_{1:t-1})\\
&=
\begin{split}
&\int P_\theta(s'_{1:t},a'_{1:t-1})\int P_\theta(s_{t+1:T},a_{t+1:T}\vert s'_{1:t},a'_{1:t-1},a_t)P_\theta(a_t\vert s'_{1:t},a'_{1:t-1})\nabla_\theta\log P_\theta(a_t \vert s'_t)(b(s_t))\\

&d(s_{t+1:T},a_{t:T})d(s'_{1:t},a'_{1:t-1})
\end{split}\\

&=
\begin{split}
& \int P_\theta(s'_{1:t},a'_{1:t-1})\int\int P_\theta(s_{t+1:T},a_{t+1:T}\vert s'_{1:t},a'_{1:t-1},a_t)P_\theta(a_t\vert s'_{1:t},a'_{1:t-1})\nabla_\theta\log P_\theta(a_t \vert s'_t)(b(s_t))\\

&d(s_{t+1:T},a_{t+1:T})d(a_t)d(s'_{1:t},a'_{1:t-1})\\
\end{split}\\

&= 
\begin{split}
&\int P_\theta(s'_{1:t},a'_{1:t-1})\int P_\theta(s_{t+1:T},a_{t+1:T}\vert s'_{1:t},a'_{1:t-1},a_t)d(s_{t+1:T},a_{t+1:T})\int P_\theta(a_t\vert s'_{1:t},a'_{1:t-1})\\

&\nabla_\theta\log P_\theta(a_t \vert s'_t)(b(s_t))d(a_t)d(s'_{1:t},a'_{1:t-1})\\
\end{split}\\
&= \int P_\theta(s'_{1:t},a'_{1:t-1})\int P_\theta(a_t\vert s'_t)\nabla_\theta\log P_\theta(a_t \vert s'_t)(b(s_t))d(a_t)d(s'_{1:t},a'_{1:t-1})\\

&= \int P_\theta(s'_{1:t},a'_{1:t-1})b(s_t)\int \nabla_\theta P_\theta(a_t \vert s'_t)d(a_t)d(s'_{1:t},a'_{1:t-1})\\

&= 0
\end{align}
$$
**Problem 4**

![p4_sb](plots/a2_p4_sb.png)

![p4_lb](plots/a2_p4_lb.png)

(a) Reward-to-go.

(b) Yes. In the case of a large batch size the average returns are a bit more stable (after convergence) when the advantages were centered compared to when they were not. However, there is no noticeable effect for small batch sizes.

(c) Yes. For larger batch sizes a smaller number of iterations were required for convergence. Also, the graphs for average returns were more smoother (less fluctuations) for larger batch sizes.

**Problem 5**

It can be seen in the graph below that several different combinations of batch sizes and learning rates achieve the optimum return. One such example is a batch size of 1000 and a learning rate of 0.010.![p5](plots/a2_p5.png)

**Problem 7**![p7](plots/a2_p7.png)



**Problem 8**

(a) The following graph shows the effects that changing the batch size and learning rates had on the average returns. It can be seen that larger batch sizes and higher learning rates generally lead to higher average returns.![p8a](plots/a2_p8a.png)  

(b) Choose $$b*=50000$$ and $$r*=0.020$$. The following graph shows the results with different settings of reward-to-go and baselines. ![p8b](plots/a2_p8b.png)
