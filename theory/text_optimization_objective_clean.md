# Text optimization objective

**Def 1 (data-generating process).** Base model $M$ with parameters $\theta$ gives $p_\theta(y\mid x,\pi)$, the probability of response $y$ given query $x$ and system prompt $\pi$. Construct the dataset $D=\{(x_i,y_i)\}_{i=1}^n$ by sampling $x_i\sim Q$ (a pool of queries) and $y_i\sim p_\theta(\cdot\mid x_i,\pi^\star)$ (responses under the canonical system prompt $\pi^\star$). Joint:
$$\mathcal{D}(x,y)=Q(x)\,p_\theta(y\mid x,\pi^\star).$$

**Def 2 (NLL objective).** Text optimization minimizes
$$\mathcal{L}(\pi)=\mathrm{NLL}(D;\pi)=-\frac{1}{|D|}\sum_{(x,y)\in D}\log p_\theta(y\mid x,\pi)\approx\mathbb{E}_{x\sim Q}\,\mathbb{E}_{y\sim p_\theta(\cdot\mid x,\pi^\star)}\big[-\log p_\theta(y\mid x,\pi)\big].$$

In practice we minimize the left-hand side; spiritually it is the right-hand side, which is what we prove things about.

**Remark.** Generally this setup is context distillation. When the canonical system prompt $\pi^\star$ and the set of queries are unrelated (e.g., loving cats vs. numbers), it replicates the classic prompted subliminal learning setup.

**Theorem 1.** $\pi^\star$ is the unique minimizer of $\mathcal{L}$.

*Part 1 ($\pi^\star$ minimizes $\mathcal{L}$).* For any prompt $\pi$, adding and subtracting $\log p_\theta(y\mid x,\pi^\star)$ inside the population objective gives
$$\begin{aligned}
\mathcal{L}(\pi) &= \mathbb{E}_{x\sim Q}\,\mathbb{E}_{y\sim p_\theta(\cdot\mid x,\pi^\star)}\big[-\log p_\theta(y\mid x,\pi)\big]\\
&= \mathbb{E}_{x,y}\Big[\log\tfrac{p_\theta(y\mid x,\pi^\star)}{p_\theta(y\mid x,\pi)}\Big] + \mathbb{E}_{x,y}\big[-\log p_\theta(y\mid x,\pi^\star)\big]\\
&= \mathbb{E}_{x\sim Q}\,D_{\mathrm{KL}}\!\big(p_\theta(\cdot\mid x,\pi^\star)\,\|\,p_\theta(\cdot\mid x,\pi)\big) + \mathcal{L}(\pi^\star)\;\ge\;\mathcal{L}(\pi^\star),
\end{aligned}$$
since KL $\ge 0$ ($\mathbb{E}_{x,y}$ over the same sampling as the first line). Equality iff $p_\theta(\cdot\mid x,\pi)=p_\theta(\cdot\mid x,\pi^\star)$ for $Q$-a.e. $x$. In other words, no prompt has lower loss than $\pi^\star$.

*Part 2 (uniqueness).* By Part 1 it suffices to show that for any $\pi\neq\pi^\star$ and any query $x$, $D_{\mathrm{KL}}\!\big(p_\theta(\cdot\mid x,\pi^\star)\,\|\,p_\theta(\cdot\mid x,\pi)\big)\neq 0$. We more or less get this for free from [Nikolaou et al., arXiv:2510.15511]: for almost every choice of weights $\theta$ (for almost all initializations and through a finite number of gradient steps), the language model is injective in its input, so distinct prompts produce distinct hidden states.

*(Nathan's ~90% sure of this.) Their proof is involved, but it should extend for free to output distributions: the unembed + softmax is also real-analytic, and one just updates their witness construction to additionally have the unembedding matrix preserve the hidden-state gap...*

**Remark.** One implication is that any text-optimization procedure that searches and verifies over candidate prompts $\pi$, where every prompt has a nonzero chance of being proposed, is guaranteed to converge to the optimal $\pi^\star$ when run for sufficiently long. This isn't that interesting, since it applies even to the naive approahces like enumerating over all possible system prompts.
