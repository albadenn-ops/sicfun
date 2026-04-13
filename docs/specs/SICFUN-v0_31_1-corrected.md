# SICFUN-v0.31.1 — Exploitability Closure, Local Safety, and World Algebra Revision

**Version:** 0.31.1  
**Date:** 2026-04-04  
**Status:** World algebra closure — supersedes v0.31  
**Lineage:** v0.26 → … → v0.30.4 surgical patch → v0.31 exploitability closure → v0.31.1 world algebra closure (this document)

**Scope.** This specification extends the architecture to a 13-tuple (incorporating the design-signal kernel as a constitutive component), preserves all v0.30.4 corrections, and the five v0.31 extensions (formal exploitability, AS-strong, local Bellman-safe law, world-index framework, world-aware risk decomposition) while resolving a structural type inconsistency in the world-index algebra:

1. **Three orthogonal world axes.** The system operates on three axes — learning channel, showdown activation, policy scope — not two. The two value decompositions (four-world grid and telescopic chain) live in different product spaces of these axes.
2. **Canonical \(\Omega^{\mathrm{act}}\) unified.** The learning-channel set is fixed to \(\{\mathrm{blind}, \mathrm{ref}, \mathrm{attrib}, \mathrm{design}\}\) (4 elements) everywhere. The prior inconsistency between 3-element and 4-element versions is eliminated.
3. **Showdown wired into kernel.** The full kernel (Definition 20) is now parameterized by \((\omega^{\mathrm{act}}, \omega^{\mathrm{sd}})\), with \(\omega^{\mathrm{sd}}\) explicitly controlling whether showdown composition fires. The telescopic decomposition has well-defined counterfactual semantics.
4. **Four-world grid correctly typed.** The four worlds are declared in their own product space \(\Omega^{\mathrm{grid}} = \{\mathrm{blind}, \mathrm{attrib}\} \times \{\Pi^{\mathrm{ol}}, \Pi^S\}\), not as a subset of \(\Omega^{\mathrm{chain}}\).

**Design contract.** All design-contract prohibitions from v0.31 and prior versions are inherited. Additionally:
- the symbol \(\delta_{\mathrm{adapt}}\) is renamed to \(\varepsilon_{\mathrm{adapt}}\) to avoid collision with the Dirac/floor usage of \(\delta\); backward references to \(\delta_{\mathrm{adapt}}\) should be read as \(\varepsilon_{\mathrm{adapt}}\);
- the symbol \(\delta_{\mathrm{retreat}}\) is canonically written as \(\delta_{\mathrm{cp\text{-}retreat}}\) everywhere; backward references to \(\delta_{\mathrm{retreat}}\) should be read as \(\delta_{\mathrm{cp\text{-}retreat}}\);
- no bare \(\omega\) without product-space qualification is permitted.

**Multiway invariant.** Inherited and strengthened. The exploitability and safety definitions are natively multiway through the joint rival profile \(\sigma^{-S} \in \Sigma^{-S}\).

---

## 0. Convention

Let \(t \in \mathbb{N}\) be the discrete temporal decision index. Rewards are bounded by \(R_{\max}\), and the discount factor satisfies \(\gamma \in (0,1)\).

We distinguish two policy layers:

\[
\Pi^{\mathrm{ol}} \subseteq \Pi^S,
\]

where:

- \(\Pi^S\) is SICFUN's full admissible policy class,
- \(\Pi^{\mathrm{ol}}\) is the open-loop subclass, whose elements do not condition on observational history.

Whenever a generic policy class \(\Pi\) appears in a value definition, it is understood that \(\Pi \subseteq \Pi^S\).

**Convention on \(\Pi^S\) closure type.** This specification does not force \(\Pi^S\) to be closed-loop unless a specific theorem or construction requires it. Theorems that require closed-loop policies state this as an explicit hypothesis.

**Three orthogonal world axes.** The system distinguishes:

| Axis | Symbol | Domain | Controls |
|---|---|---|---|
| Learning channel | \(\omega^{\mathrm{act}}\) | \(\Omega^{\mathrm{act}} := \{\mathrm{blind}, \mathrm{ref}, \mathrm{attrib}, \mathrm{design}\}\) | Which action-channel kernel updates the rival |
| Showdown activation | \(\omega^{\mathrm{sd}}\) | \(\Omega^{\mathrm{sd}} := \{0, 1\}\) | Whether showdown composition is applied in the full kernel |
| Policy scope | \(\omega^{\mathrm{pol}}\) | \(\Omega^{\mathrm{pol}} := \{\Pi^{\mathrm{ol}}, \Pi^S\}\) | Over which policy class the value optimum is taken |

Two product spaces are used:

- **Chain space** \(\Omega^{\mathrm{chain}} := \Omega^{\mathrm{act}} \times \Omega^{\mathrm{sd}}\), with \(|\Omega^{\mathrm{chain}}| = 8\). Used for parameterizing the full kernel (Definition 20) and for the telescopic edge/risk decomposition (§8.2A, §9D).

- **Grid space** \(\Omega^{\mathrm{grid}} := \{\mathrm{blind}, \mathrm{attrib}\} \times \{\Pi^{\mathrm{ol}}, \Pi^S\}\), with \(|\Omega^{\mathrm{grid}}| = 4\). Used for the four-world value decomposition (Definition 44, Theorem 4).

**Grid restriction rationale.** The grid projects onto only two of four learning channels because the four-world decomposition isolates the extremes of rival learning: \(\mathrm{blind}\) (no learning) and \(\mathrm{attrib}\) (full attributed learning). The intermediate channels \(\mathrm{ref}\) and \(\mathrm{design}\) contribute to the telescopic chain, not to the aggregate grid. This yields a minimal \(2 \times 2\) grid sufficient to decompose value into control, signaling, and interaction components. A Lemma justifying this restriction: for the grid decomposition (Theorem 4), only the endpoints of the signal dimension matter, since intermediate channels are linearly interpolated in the telescopic chain.

These two product spaces are **not** subsets of each other. They project onto different pairs of axes.

**Full product space (reserved notation).** The full product \(\Omega^{\mathrm{full}} := \Omega^{\mathrm{act}} \times \Omega^{\mathrm{sd}} \times \Omega^{\mathrm{pol}}\) has \(|\Omega^{\mathrm{full}}| = 4 \times 2 \times 2 = 16\). No theorem in the current specification requires simultaneous variation of all three axes, but the notation is reserved for future extensions.

**Notational disambiguation.** Throughout this document:
- \(\alpha_X, \alpha_A, \alpha_Y\) denote **abstraction maps** (A1′);
- \(\kappa_{\mathrm{temp}} \in (0,1]\) denotes the **power-posterior tempering exponent** (§4.1);
- \(\kappa_{\mathrm{cp}} \in (0,1)\) denotes the **changepoint detection threshold** (Definition 28);
- \(\beta_t^{i,\mathrm{exploit}} \in [0,1]\) denotes the **exploitation interpolation parameter** for rival \(i\) (Definition 15C);
- \(\delta_{\mathrm{cp\text{-}retreat}} \in (0,1]\) denotes the **changepoint retreat rate** (Definition 15C); this is the sole permissible use of \(\delta_{\mathrm{retreat}}\) — any bare \(\delta_{\mathrm{retreat}}\) is to be read as \(\delta_{\mathrm{cp\text{-}retreat}}\);
- \(\varepsilon_{\mathrm{adapt}} \ge 0\) denotes the **adaptation safety tolerance** (§9B).

No other uses of α, κ, β, or ω are permitted unless locally qualified.

---

## 1. System definition

SICFUN is the 13-tuple

\[
\mathfrak{S}
:=
\big(
\widetilde{\mathcal X},
\mathcal U,
\mathcal O,
T,
O,
\Pi^S,
\Pi^{\mathrm{ol}},
\{\Gamma^{\mathrm{act,attrib},i}\}_{i\in\mathcal R},
\{\Gamma^{\mathrm{act,ref},i}\}_{i\in\mathcal R},
\{\Gamma^{\mathrm{act,blind},i}\}_{i\in\mathcal R},
\{\Gamma^{\mathrm{act,design},i}\}_{i\in\mathcal R},
\{\Gamma^{\mathrm{sd},i}\}_{i\in\mathcal R},
r
\big),
\]

where \(\mathcal R\) is the set of active rivals and:

| Symbol | Object | Role |
|---|---|---|
| \(\widetilde{\mathcal X}\) | Augmented hidden state space | Domain of the operative belief |
| \(\mathcal U\) | Action–sizing space \(\mathcal A \times \Lambda\) | SICFUN's decision variables |
| \(\mathcal O\) | Observation space | Private and public signals |
| \(T\) | Transition kernel \(\widetilde{\mathcal X}\times\mathcal U \to \Delta(\widetilde{\mathcal X})\) | Hidden-state dynamics (incorporates A3′ type dynamics) |
| \(O\) | Observation kernel \(\widetilde{\mathcal X}\times\mathcal U \to \Delta(\mathcal O)\) | Signal generation |
| \(\Pi^S\) | Full policy class | Admissible policies of SICFUN |
| \(\Pi^{\mathrm{ol}}\subseteq \Pi^S\) | Open-loop subclass | Policies independent of observational history |
| \(\Gamma^{\mathrm{act,attrib},i}\) | Inferential action kernel for rival \(i\) | Rival \(i\)'s update under its attributed model of SICFUN |
| \(\Gamma^{\mathrm{act,ref},i}\) | Inferential action kernel for rival \(i\) | Rival \(i\)'s update under SICFUN's true baseline |
| \(\Gamma^{\mathrm{act,blind},i}\) | Inferential action kernel for rival \(i\) | Frozen rival: ignores action-signal learning |
| \(\Gamma^{\mathrm{act,design},i}\) | Design-signal kernel for rival \(i\) | Rival \(i\)'s update under action-marginal tempered likelihood |
| \(\Gamma^{\mathrm{sd},i}\) | Showdown kernel for rival \(i\) | Revelatory update on certified terminal disclosures |
| \(r\) | Reward function \(\widetilde{\mathcal X}\times\mathcal U\to\mathbb R\) | Bounded by \(R_{\max}\) |

**Topological note on \(\Lambda\).** The sizing space \(\Lambda\) appearing in \(\mathcal U = \mathcal A \times \Lambda\) is assumed to be a compact metrizable space unless otherwise stated. Compactness of \(\Lambda\) (together with finiteness or compactness of \(\mathcal A\)) ensures that suprema over policies in value definitions (Definition 32) are well-defined. When \(\Lambda\) is finite, compactness is immediate; when continuous, measurability and compactness must be verified by the implementer.

**Architectural extensions (non-constitutive).**

| Symbol | Object | Role | Section |
|---|---|---|---|
| \(\mathcal D\) | Changepoint detection module | Non-stationarity monitor for rival types | §5A |
| \(\mathcal B_\rho\) | Ambiguity set for distributionally robust planning | Wasserstein ball of radius \(\rho\) around operative belief | §6A |
| \(B^\star\) | Minimal safety certificate | Bellman-safe budget for adaptation safety | §9C |

**Constitutive extension for safety.**

| Symbol | Object | Role | Section |
|---|---|---|---|
| \(\Sigma^{-S}\) | Joint rival profile class | Multiway adversarial strategy space; constitutive for exploitability (§9A′), adaptation safety (§9B), and the local Bellman-safe law (§9C) | §9A′ |

**Note.** \(\Sigma^{-S}\) is reclassified from non-constitutive to constitutive because it appears in the definitions of the robust performance functional (Definition 52A), security value (Definition 52B), adaptation safety (Definition 57), and the safety Bellman operator (Definition 60).

---

## 2. Assumptions

### A1′. Abstraction with guarantees

There exist abstraction maps \(\alpha_X,\alpha_A,\alpha_Y\) and constants \(\varepsilon_R,\varepsilon_T,\varepsilon_O \ge 0\) such that:

\[
\sup_{x,u}\big|r(x,u)-\hat r(\alpha_X(x),\alpha_A(u))\big|
\le \varepsilon_R,
\]

\[
\sup_{x,u}\mathrm{TV}\!\big(
T(\cdot\mid x,u),
\alpha_X^{-1\sharp}\hat T(\cdot\mid \alpha_X(x),\alpha_A(u))
\big)
\le \varepsilon_T,
\]

\[
\sup_{x,u}\mathrm{TV}\!\big(
O(\cdot\mid x,u),
\alpha_Y^{-1\sharp}\hat O(\cdot\mid \alpha_X(x),\alpha_A(u))
\big)
\le \varepsilon_O.
\]

In fully observed MDPs:

\[
\|V-\hat V\|_\infty
\le
\frac{\varepsilon_R}{1-\gamma}
+
\frac{\gamma R_{\max}\varepsilon_T}{(1-\gamma)^2}.
\]

In POMDPs, \(\varepsilon_O\) enters through posterior distortion; value error must be controlled jointly through the induced belief-MDP. Specifically, if the belief-MDP induced by the abstracted observation kernel satisfies a coupling condition with the true belief-MDP, the total value error is bounded by

\[
\|V-\hat V\|_\infty
\le
\frac{\varepsilon_R}{1-\gamma}
+
\frac{\gamma R_{\max}(\varepsilon_T + \varepsilon_O)}{(1-\gamma)^2}.
\]

The additive contribution of \(\varepsilon_O\) follows from the triangle inequality on total-variation distance in the belief update. A tight bound requires explicit coupling analysis of the belief-MDP, which is delegated to the implementation layer.

### A2. Closed Markovianity

\[
\widetilde X_{t+1} \sim T(\cdot\mid \widetilde X_t, u_t).
\]

### A3. Rival latent type

For each rival \(i\in\mathcal R\), there exists \(\theta_t^{R,i}\in\Theta^{R,i}\) summarizing exploitable regularities.

### A3′. Rival type non-stationarity

\[
\theta_{t+1}^{R,i} \sim \mathcal{K}^i(\cdot \mid \theta_t^{R,i}, \zeta_t^i),
\]

where \(\zeta_t^i \in \{0,1\}\) is a latent changepoint indicator with prior hazard rate \(h^i \in [0,1)\), and \(\mathcal{K}^i\) is the type transition kernel.

**Integration with T.**

\[
T\big|_{\Theta^{R,i}}(\theta_{t+1}^{R,i} \mid \widetilde X_t, u_t) = \mathcal{K}^i(\theta_{t+1}^{R,i} \mid \theta_t^{R,i}, \zeta_t^i).
\]

**Stationarity recovery.** A3 is recovered when \(h^i = 0\) and \(\mathcal{K}^i\) is the identity kernel.

**Status of the run length.** \(r_t^i = R^i(\mathcal{H}_t) \in \mathbb{N}\) is a derived sufficient statistic, not a component of \(\widetilde{\mathcal X}\). Consequently, the changepoint detection module \(\mathcal{D}^i\) (Definition 26) operates on the observational history \(\mathcal{H}_t\), not on the augmented state \(\widetilde{X}_t\).

### A4′. Own statistical sufficiency (conditioned)

There exists finite \(\xi_t^S\in\Xi^S\) sufficient for SICFUN's accumulated evidence relative to A1′, A6, and a declared parametric family \(\mathfrak F^R\).

**Definition of \(\mathfrak F^R\).** The parametric family \(\mathfrak F^R := \{\mathcal{K}^i(\cdot \mid \theta, \zeta) : \theta \in \Theta^{R,i}, \, \zeta \in \{0,1\}, \, i \in \mathcal{R}\}\) is the class of rival-type transition kernels admitted by A3′. Sufficiency of \(\xi_t^S\) is relative to the product family \(\prod_{i \in \mathcal{R}} \mathfrak{F}^{R,i}\), meaning that \(\xi_t^S\) captures all information in \(\mathcal{H}_t\) relevant for distinguishing among members of \(\mathfrak{F}^R\) given A1′ and A6.

### A5. Bounded reward and discounting

\[
|r(\widetilde x,u)|\le R_{\max},\qquad \gamma\in(0,1).
\]

### A6. First-order interactive sufficiency

For each rival \(i\), future policy depends on public history only through \(m_t^{R,i}\in\mathcal M^{R,i}\). This is a first-order truncation.

### A6′. Detection-aware exploitation

There exists a detection predicate \(\mathrm{DetectModeling}^i : \mathcal{H}_t \to \{0,1\}\) for each rival \(i\).

**Minimal activation conditions.** An implementation of \(\mathrm{DetectModeling}^i\) must satisfy: (i) it is measurable with respect to the filtration generated by \(\mathcal{H}_t\); (ii) it returns 0 identically when rival \(i\)'s observed play is indistinguishable from baseline under A6; and (iii) if rival \(i\) conditions on a model of SICFUN's exploitation parameter \(\beta^{i,\mathrm{exploit}}\), then \(\mathrm{DetectModeling}^i = 1\) with probability tending to 1 as evidence accumulates. The specific statistical test is left to the implementation layer.

### A7. Well-defined full rival update

For each rival \(i\), there exists a full kernel

\[
\Gamma^{\mathrm{full},(\omega^{\mathrm{act}},\omega^{\mathrm{sd}}),\,i}
:
\mathcal M^{R,i}\times \mathcal Y \times \mathcal X^{\mathrm{pub}}
\to
\mathcal M^{R,i},
\]

where \((\omega^{\mathrm{act}}, \omega^{\mathrm{sd}}) \in \Omega^{\mathrm{chain}}\), such that

\[
m_{t+1}^{R,i}
=
\Gamma^{\mathrm{full},(\omega^{\mathrm{act}},\omega^{\mathrm{sd}}),\,i}(m_t^{R,i},Y_t,x_t^{\mathrm{pub}}).
\]

The learning-channel component \(\omega^{\mathrm{act}} \in \Omega^{\mathrm{act}}\) selects the action-channel kernel. The showdown component \(\omega^{\mathrm{sd}} \in \{0,1\}\) controls whether revelatory composition with \(\Gamma^{\mathrm{sd},i}\) is applied.

### A8. Strategically relevant repetition

There exists a constant \(\underline{p} > 0\) such that for all \(t\) and all states \(\widetilde x \in \widetilde{\mathcal X}\),

\[
\Pr\!\big(\text{at least one future interaction with rival } i \text{ occurs after } t \mid \widetilde X_t = \widetilde x\big) \ge \underline{p}.
\]

**Interpretation.** Future interaction occurs with non-negligible intensity. The constant \(\underline{p}\) provides the lower bound required for the exploitation and safety results: any theorem depending on A8 inherits a dependence on \(\underline{p}\). In the discounted setting with \(\gamma < 1\), \(\underline{p}\) need not be close to 1; it suffices that the probability of future interaction is bounded away from zero uniformly.

### A9. Spot-conditioned polarization (per-rival)

\[
\mathrm{Pol}_t^i(\lambda)
=
\mathrm{Pol}_t^i(\lambda \mid x_t^{\mathrm{pub}}, \pi^{0,S}, m_t^{R,i}).
\]

### A10. Adaptation safety baseline (revised)

There exists an approximate baseline strategy \(\bar\pi := \bar\pi^{0,S} \in \Pi^S\) with exploitability \(\varepsilon_{\mathrm{base}} \ge 0\):

\[
\mathrm{Exploit}_{\mathcal{B}_{\mathrm{dep}}}(\bar\pi) \le \varepsilon_{\mathrm{base}},
\]

where \(\mathrm{Exploit}\) is defined formally in §9A′.

SICFUN's exploitation must satisfy **adaptation safety** (Definition 57): for every joint rival profile \(\sigma^{-S} \in \Sigma^{-S}\) and every deployment belief \(\widetilde b \in \mathcal{B}_{\mathrm{dep}}\),

\[
J(\widetilde b; \pi, \sigma^{-S}) \ge J(\widetilde b; \bar\pi, \sigma^{-S}) - \varepsilon_{\mathrm{adapt}}.
\]

---

## 3. Primitive objects

### Definition 1. Private strategic classes of SICFUN

\[
\mathcal C^S
=
\mathcal C^V
\sqcup
\mathcal C^B
\sqcup
\mathcal C^M
\sqcup
\mathcal C^{SB}.
\]

**Semantic interpretation.**
- \(\mathcal{C}^V\) (value classes): classes in which SICFUN holds genuine private value and acts accordingly.
- \(\mathcal{C}^B\) (bluff classes): classes in which SICFUN does not hold private value commensurate with its action but acts aggressively to create the appearance of value.
- \(\mathcal{C}^M\) (mixed classes): classes that combine elements of value-holding and bluffing in a single strategic posture.
- \(\mathcal{C}^{SB}\) (structural-bluff classes): classes in which the bluff arises from SICFUN's structural position rather than from a deliberate deception — the aggressive action is warranted by the structural context even when private value is absent.

### Definition 2. Current private strategic class

\[
c_t^S \in \mathcal C^S.
\]

### Definition 3. Aggressive-wager predicate

\[
\mathrm{Agg}:\mathcal U\to\{0,1\}
\]

### Definition 4. Structural-bluff predicate

\[
\mathrm{StructuralBluff}(c,u)=1
\iff
\big(c\in \mathcal C^B\big)\land \mathrm{Agg}(u)=1.
\]

### Definition 5. Size-aware public action signal

\[
Y_t^{\mathrm{act}}=(a_t,\lambda_t,\tau_t),
\qquad
\lambda_t\in\Lambda,
\quad
\tau_t\in\mathcal T.
\]

### Definition 6. Total public signal

\[
Y_t=(Y_t^{\mathrm{act}},Y_t^{\mathrm{sd}})
\in
\mathcal Y
=
\mathcal Y^{\mathrm{act}}\times \mathcal Y^{\mathrm{sd}},
\]

where \(Y_t^{\mathrm{sd}}=\varnothing\) if no terminal revelation occurs at \(t\).

### Definition 7. Observation object and canonical identification

\[
Z_{t+1}\in\mathcal O.
\]

In the canonical SICFUN semantics,

\[
\mathcal O := \mathcal Y,
\qquad
Z_{t+1} := Y_t.
\]

**Convention.** The identification \(Z_{t+1}:=Y_t\) is a **temporal indexing convention**, not an ontological identity.

### Definition 8. Signal routing convention

- \(Y_t^{\mathrm{act}}\) enters the **inferential channel** through the two-layer tempered likelihood (Definition 15A).
- \(Y_t^{\mathrm{sd}}\) enters the **revelatory channel** through certified showdown updating.
- Full rival updating composes both channels in the declared order of §4, gated by \(\omega^{\mathrm{sd}}\).

### Definition 9. Real baseline of SICFUN

\[
\pi^{0,S}(a,\lambda\mid c,x_t^{\mathrm{pub}})
\]

### Definition 10. Attributed baseline (per-rival, state-conditioned)

\[
\widehat{\pi}^{0,S,i}(a,\lambda\mid c,x^{\mathrm{pub}},m^{R,i}).
\]

**Partial-policy typing.** For fixed \(x \in \mathcal{X}^{\mathrm{pub}}\) and \(m \in \mathcal{M}^{R,i}\), the attributed baseline induces a partial policy

\[
\widehat{\pi}^{0,S,i}_{x,m}(\cdot \mid \cdot) := \widehat{\pi}^{0,S,i}(\cdot \mid \cdot, x, m).
\]

### Definition 11. Reputational projection (derived, per-rival)

\[
\phi_t^{S,i}:=g^i(m_t^{R,i}).
\]

No global \(\phi^S\) exists unless explicitly introduced.

### Definition 12. Augmented hidden state space

\[
\widetilde{\mathcal X}
:=
\mathcal X^{\mathrm{pub}}
\times
\mathcal X^{\mathrm{priv}}
\times
\prod_{i\in\mathcal R}(\Theta^{R,i}\times\mathcal M^{R,i})
\times
\Xi^S.
\]

### Definition 13. Augmented hidden state

\[
\widetilde X_t
=
\big(
x_t^{\mathrm{pub}},
x_t^{\mathrm{priv}},
\{\theta_t^{R,i},m_t^{R,i}\}_{i\in\mathcal R},
\xi_t^S
\big).
\]

### Definition 14. Operative belief

\[
\widetilde b_t \in \Delta(\widetilde{\mathcal X}).
\]

---

## 4. Rival kernel constructor

### 4.1 Two-layer tempered regularization

### Definition 15. Tempering exponent and safety floor

Let \(\kappa_{\mathrm{temp}} \in (0,1]\) be the power-posterior tempering exponent, with recommended calibration \(\kappa_{\mathrm{temp}} \approx 0.85\text{–}0.95\). Let \(\delta_{\mathrm{floor}} \ge 0\) be the safety floor parameter. Let \(\eta \in \Delta(\mathcal Y^{\mathrm{act}})\) satisfy \(\eta(y) > 0\) for all \(y \in \mathcal Y^{\mathrm{act}}\).

The semantic choice is declared at model configuration time:

- **two-layer tempered semantics (default):** power-posterior tempering with additive safety floor;
- **pure power-posterior semantics:** \(\delta_{\mathrm{floor}} = 0\), totality conditional on at least one class having positive base probability;
- **legacy ε-smoothing semantics (backward compatibility):** use the separate legacy likelihood.

### Definition 15A. Two-layer tempered likelihood

\[
L_{\kappa,\delta}^i(y \mid c, \pi, x^{\mathrm{pub}}, m^{R,i})
:=
\Pr(y \mid c, \pi, x^{\mathrm{pub}}, m^{R,i})^{\kappa_{\mathrm{temp}}}
+
\delta_{\mathrm{floor}} \cdot \eta(y).
\]

**Interpretation.** \(L_{\kappa,\delta}^i\) is a **positive regularized score**, not necessarily a normalized probability.

**Default semantics convention.** Unless a theorem or definition explicitly selects a different semantic option, the two-layer tempered semantics (\(\kappa_{\mathrm{temp}} \in (0,1]\), \(\delta_{\mathrm{floor}} > 0\)) is the active default. Theorems 1 and 2 explicitly state their dependence on semantic parameters. All other results assume the default.

**Convergence note.** Because \(L_{\kappa,\delta}^i\) is a non-normalized score, the sequence of posteriors \(\{\mu_t^{R,i}\}\) produced by Definition 15B does not inherit the standard Bayesian consistency guarantees directly. Under the default semantics with \(\delta_{\mathrm{floor}} > 0\), the posterior is well-defined (Theorem 1) and converges to the smoothed Bayes posterior as \(\kappa_{\mathrm{temp}} \to 1\) (Theorem 2a). Long-run concentration around the true type requires that the tempered likelihood preserves identifiability — i.e., distinct classes \(c \neq c'\) produce distinct tempered scores for at least some \(y\). This condition is satisfied generically when the base likelihood separates classes and \(\kappa_{\mathrm{temp}} > 0\).

**Properties:**
1. **Unconditional totality.** If \(\delta_{\mathrm{floor}} > 0\), then \(L_{\kappa,\delta}^i(y \mid c, \ldots) \ge \delta_{\mathrm{floor}} \cdot \eta(y) > 0\).
2. **Likelihood ordering preservation.**
3. **Tempering benefit.** \(\kappa_{\mathrm{temp}} < 1\) attenuates extreme likelihood ratios.
4. **Calibration.** \(\kappa_{\mathrm{temp}} \in (0,1]\); clipped anchor \(\kappa_{\mathrm{temp}}^{\mathrm{anchor}} := \min\{1, 1/(2\varepsilon_{\mathrm{mis}}^2)\}\).

**Legacy recovery.** The v0.29.1 form \(L_{\mathrm{legacy}}^i(y \mid c, \pi, x^{\mathrm{pub}}, m^{R,i}; \varepsilon_{\mathrm{legacy}}) := (1 - \varepsilon_{\mathrm{legacy}})\Pr(y \mid c, \ldots) + \varepsilon_{\mathrm{legacy}} \cdot \eta(y)\) is available as a third configuration option.

### Definition 15B. Posterior-on-class update (two-layer tempered)

\[
\mu_{t+1}^{R,i,\pi,\kappa,\delta}(c)
=
\frac{
L_{\kappa,\delta}^i(Y_t^{\mathrm{act}} \mid c, \pi, x_t^{\mathrm{pub}}, m_t^{R,i}) \, \mu_t^{R,i}(c)
}{
\sum_{c'}
L_{\kappa,\delta}^i(Y_t^{\mathrm{act}} \mid c', \pi, x_t^{\mathrm{pub}}, m_t^{R,i}) \, \mu_t^{R,i}(c')
}.
\]

When \(\delta_{\mathrm{floor}} > 0\), the denominator is strictly positive. When \(\delta_{\mathrm{floor}} = 0\) and the denominator vanishes, the prior is preserved.

### Definition 15C. Exploitation interpolation parameter

For each rival \(i\), \(\beta_t^{i,\mathrm{exploit}} \in [0, 1]\).

**Formal interpolation mechanism.** The interpolated action kernel for rival \(i\) is:

\[
\Gamma_t^{\mathrm{act,interp},i}(m, y, x^{\mathrm{pub}})
:=
U^{R,i}\!\Big(
m, \;
(1 - \beta_t^{i,\mathrm{exploit}}) \cdot \mu^{R,i,\pi^{0,S},\kappa,\delta}_{t+1}
+
\beta_t^{i,\mathrm{exploit}} \cdot \mu^{R,i,\widehat{\pi}^{0,S,i},\kappa,\delta}_{t+1}
\Big).
\]

**Adaptation safety constraint.** Subject to the local safety law of §9C when available, or to the scalar bound \(\beta_t^{i,\mathrm{exploit}} \in S(\varepsilon_{\mathrm{adapt}})\) of Theorem 8.

**Detection-triggered retreat.** When \(\mathrm{DetectModeling}^i(\mathcal{H}_t) = 1\):

\[
\beta_{t+1}^{i,\mathrm{exploit}} \leftarrow \max\!\big(0, \, \beta_t^{i,\mathrm{exploit}} - \delta_{\mathrm{cp\text{-}retreat}}\big).
\]

**Changepoint-triggered retreat.** When CPD triggers for rival \(i\):

\[
P(r_t^i \le r_{\min} \mid Y_{1:t}) > \kappa_{\mathrm{cp}}
\;\implies\;
\beta_{t+1}^{i,\mathrm{exploit}} \leftarrow \max\!\big(0, \, \beta_t^{i,\mathrm{exploit}} - \delta_{\mathrm{cp\text{-}retreat}}\big).
\]

### 4.2 Kernel construction

### Definition 16. State-embedding updater

For each rival \(i\),

\[
U^{R,i}:
\mathcal M^{R,i}\times \Delta(\mathcal C^S)
\to
\mathcal M^{R,i}.
\]

**Policy-invariance convention.** Once the posterior argument is supplied, the updater is independent of the generating policy.

### Definition 17. BuildRivalKernel\(^i_{\kappa,\delta}\) (action channel only)

\[
\mathrm{BuildRivalKernel}^i_{\kappa,\delta}(\pi)(m,y,x^{\mathrm{pub}})
:=
U^{R,i}\!\big(m,\mu^{R,i,\pi,\kappa,\delta}(\cdot \mid m,y,x^{\mathrm{pub}})\big).
\]

### Definition 18. Inferential action kernels

\[
\Gamma^{\mathrm{act,ref},i}
:=
\mathrm{BuildRivalKernel}^i_{\kappa,\delta}(\pi^{0,S}),
\]

\[
\Gamma^{\mathrm{act,attrib},i}(m,y,x^{\mathrm{pub}})
:=
\mathrm{BuildRivalKernel}^i_{\kappa,\delta}
\big(
\widehat{\pi}^{0,S,i}_{x^{\mathrm{pub}},m}
\big)
(m,y,x^{\mathrm{pub}}),
\]

where \(\widehat{\pi}^{0,S,i}_{x^{\mathrm{pub}},m}\) is the partial policy of Definition 10.

\[
\Gamma^{\mathrm{act,blind},i}(m,y,x^{\mathrm{pub}})
:=
m.
\]

### Definition 19. Showdown kernel

\[
\Gamma^{\mathrm{sd},i}
:
\mathcal M^{R,i}\times \mathcal Y^{\mathrm{sd}}
\to
\mathcal M^{R,i}.
\]

Not regularized.

### Definition 19A. Design-signal kernel

\[
\Gamma^{\mathrm{act,design},i}(m, y, x^{\mathrm{pub}})
:=
U^{R,i}\!\big(m, \mu^{R,i,\mathrm{design}}_{t+1}\big),
\]

where the design-signal posterior uses the action-marginal tempered likelihood (temper-then-marginalize order).

**Order justification.** The temper-then-marginalize order is chosen because tempering first attenuates extreme likelihood ratios before marginalization, which prevents rare actions from dominating the marginal. The alternative marginalize-then-temper order would first average over actions (preserving extreme ratios) and then temper the result, potentially under-regularizing in the tails. Under the default semantics with \(\kappa_{\mathrm{temp}} < 1\) and \(\delta_{\mathrm{floor}} > 0\), the two orders yield identical results when the base likelihood is constant across actions; they differ when action-conditional likelihoods have high variance.

### Definition 20. Full per-rival kernels

For each rival \(i\) and each \((\omega^{\mathrm{act}}, \omega^{\mathrm{sd}}) \in \Omega^{\mathrm{chain}}\), define

\[
\Gamma^{\mathrm{full},(\omega^{\mathrm{act}},\omega^{\mathrm{sd}}),\,i}
:
\mathcal M^{R,i}\times\mathcal Y\times\mathcal X^{\mathrm{pub}}
\to
\mathcal M^{R,i}
\]

by

\[
\Gamma^{\mathrm{full},(\omega^{\mathrm{act}},\omega^{\mathrm{sd}}),\,i}(m,Y,x^{\mathrm{pub}})
=
\begin{cases}
m,
& \text{if } \omega^{\mathrm{act}}=\mathrm{blind},
\\[6pt]
\Gamma^{\mathrm{sd},i}\!\big(
\Gamma^{\mathrm{act},\omega^{\mathrm{act}},i}(m,Y^{\mathrm{act}},x^{\mathrm{pub}}),\;
Y^{\mathrm{sd}}
\big),
& \text{if } \omega^{\mathrm{act}} \neq \mathrm{blind},\;
\omega^{\mathrm{sd}} = 1,\;
Y^{\mathrm{sd}}\neq\varnothing,
\\[6pt]
\Gamma^{\mathrm{act},\omega^{\mathrm{act}},i}(m,Y^{\mathrm{act}},x^{\mathrm{pub}}),
& \text{otherwise}.
\end{cases}
\]

**Reading.** (1) Blind world: memory frozen. (2) Active learning with showdown enabled and terminal data present: action-channel fires, then showdown composes. (3) Otherwise: only action-channel fires. Setting \(\omega^{\mathrm{sd}} = 0\) suppresses showdown even when terminal revelation data exists; this is required for the telescopic decomposition to have well-defined counterfactual semantics.

**Effective cardinality note.** Case (1) applies identically for both \((\mathrm{blind}, 0)\) and \((\mathrm{blind}, 1)\), since the blind kernel ignores all signals including showdown. The effective number of functionally distinct worlds in \(\Omega^{\mathrm{chain}}\) is therefore 6, not 8. The nominal cardinality of 8 is retained for notational uniformity and to preserve the telescopic identity structure, but implementations should avoid redundant computation for the two blind-world elements.

**Backward compatibility.** The v0.30.4 kernel is recovered by fixing \(\omega^{\mathrm{sd}} = 1\).

### Definition 21. Joint kernel profiles

For each \((\omega^{\mathrm{act}}, \omega^{\mathrm{sd}}) \in \Omega^{\mathrm{chain}}\),

\[
\boldsymbol{\Gamma}^{(\omega^{\mathrm{act}},\omega^{\mathrm{sd}})}
:=
\big\{\Gamma^{\mathrm{full},(\omega^{\mathrm{act}},\omega^{\mathrm{sd}}),\,i}\big\}_{i\in\mathcal R}.
\]

**Shorthand for showdown-on profiles:**

\[
\boldsymbol{\Gamma}^{\mathrm{attrib}}
:=
\boldsymbol{\Gamma}^{(\mathrm{attrib},1)},
\quad
\boldsymbol{\Gamma}^{\mathrm{ref}}
:=
\boldsymbol{\Gamma}^{(\mathrm{ref},1)},
\quad
\boldsymbol{\Gamma}^{\mathrm{blind}}
:=
\boldsymbol{\Gamma}^{(\mathrm{blind},1)}.
\]

---

## 5. Dynamics

### Definition 22. Belief update

\[
\widetilde b_{t+1}
=
\widetilde\tau(\widetilde b_t,u_t,Z_{t+1}),
\qquad
Z_{t+1}\in\mathcal O=\mathcal Y.
\]

**Temporal convention reminder.** As established in Definition 7, \(Z_{t+1} := Y_t\) is a temporal indexing convention: the observation available at decision time \(t+1\) is the public signal generated at time \(t\).

### Definition 23. Full rival-state update

For each rival \(i\),

\[
m_{t+1}^{R,i}
=
\Gamma^{\mathrm{full},(\omega^{\mathrm{act}},\omega^{\mathrm{sd}}),\,i}(m_t^{R,i},Y_t,x_t^{\mathrm{pub}}),
\]

with \((\omega^{\mathrm{act}}, \omega^{\mathrm{sd}}) \in \Omega^{\mathrm{chain}}\) determined by the analytical world under consideration.

### Definition 24. Counterfactual reference world

The non-manipulative counterfactual world is the **joint reference profile** \(\boldsymbol{\Gamma}^{\mathrm{ref}} = \boldsymbol{\Gamma}^{(\mathrm{ref},1)}\).

**Showdown-on justification.** The contrafactual reference world uses \(\omega^{\mathrm{sd}} = 1\) because the showdown channel represents terminal revelations that occur independently of SICFUN's strategic choices (certified disclosures, end-of-game revelations). Suppressing showdown in the counterfactual would remove information that the rival would receive regardless of SICFUN's manipulation, producing a counterfactual that is weaker than the true no-manipulation scenario. The reference world should reflect the rival's best non-manipulated update, which includes processing all available revelatory data.

### Definition 25. Spot-conditioned polarization

\[
\mathrm{Pol}_t^i(\lambda \mid x_t^{\mathrm{pub}},\pi^{0,S},m_t^{R,i}).
\]

### 5A. Changepoint detection and non-stationarity handling

### Definition 26. Changepoint detection module

\[
\mathcal{D}^i := \big( \{r_t^i\}_{t \ge 1}, \, \mathrm{CPD}^i, \, \nu_{\mathrm{meta}}^i \big).
\]

### Definition 27. Run-length posterior update

\[
P(r_t^i = \ell \mid Y_{1:t})
\propto
\begin{cases}
P_{\mathrm{pred}}^i(Y_t \mid r_{t-1}^i = \ell - 1) \cdot (1 - h^i) \cdot P(r_{t-1}^i = \ell - 1 \mid Y_{1:t-1}),
& \ell \ge 1,
\\[4pt]
\displaystyle\sum_{\ell'} P_{\mathrm{pred}}^i(Y_t \mid r_t^i = 0) \cdot h^i \cdot P(r_{t-1}^i = \ell' \mid Y_{1:t-1}),
& \ell = 0.
\end{cases}
\]

### Definition 28. Changepoint-triggered prior reset

When \(P(r_t^i \le r_{\min} \mid Y_{1:t}) > \kappa_{\mathrm{cp}}\):

\[
\mu_t^{R,i} \leftarrow (1 - w_{\mathrm{reset}}) \cdot \mu_t^{R,i} + w_{\mathrm{reset}} \cdot \nu_{\mathrm{meta}}^i.
\]

**Admissible range.** \(w_{\mathrm{reset}} \in (0, 1]\). When \(w_{\mathrm{reset}} = 1\), the posterior is fully reset to the meta-prior; when \(w_{\mathrm{reset}} \to 0^+\), the reset is infinitesimally soft. The value \(w_{\mathrm{reset}} = 0\) is excluded because it would render the reset mechanism vacuous. The recommended default is \(w_{\mathrm{reset}} = 1\) (full reset) unless the deployment context requires gradual adaptation.

---

## 6. Value functions

### Definition 29. Belief-averaged reward

\[
\bar r(\widetilde b,u)
:=
\int_{\widetilde{\mathcal X}} r(\widetilde x,u)\,\widetilde b(d\widetilde x).
\]

### Definition 30. Policy-evaluation Q-function

\[
Q^{\pi,\boldsymbol{\Gamma}}(\widetilde b,u)
:=
\bar r(\widetilde b,u)
+
\gamma \,\mathbb E\!\big[
V^{\pi,\boldsymbol{\Gamma}}(\widetilde b')
\mid
\widetilde b,u
\big].
\]

### Definition 31. Value under policy

\[
V^{\pi,\boldsymbol{\Gamma}}(\widetilde b)
=
Q^{\pi,\boldsymbol{\Gamma}}(\widetilde b,\pi(\widetilde b)).
\]

### Definition 32. Optimal Q-function over a policy class

\[
Q^{*,\boldsymbol{\Gamma}}_\Pi(\widetilde b,u)
:=
\sup_{\pi\in\Pi} Q^{\pi,\boldsymbol{\Gamma}}(\widetilde b,u),
\qquad
V^{*,\boldsymbol{\Gamma}}_\Pi(\widetilde b)
:=
\sup_u Q^{*,\boldsymbol{\Gamma}}_\Pi(\widetilde b,u).
\]

**Existence conditions.** The supremum is well-defined whenever \(\Pi\) is non-empty and \(Q^{\pi,\boldsymbol{\Gamma}}(\widetilde b,u)\) is uniformly bounded above (guaranteed by A5, since \(|Q| \le R_{\max}/(1-\gamma)\)). The supremum is attained (i.e., a maximizer exists) when \(\Pi\) is compact in a topology that makes \(\pi \mapsto Q^{\pi,\boldsymbol{\Gamma}}(\widetilde b,u)\) upper semicontinuous. In finite or compact action/policy spaces, this is immediate. In general, the supremum may not be attained, in which case \(V^*\) is the value of the relaxed optimization. Results that require attainment state it as an explicit hypothesis.

### 6A. Distributionally robust value functions

### Definition 33. Ambiguity set

\[
\mathcal{B}_\rho(\widetilde b_t) := \big\{ \widetilde b' \in \Delta(\widetilde{\mathcal X}) : W_1(\widetilde b', \widetilde b_t) \le \rho \big\}.
\]

**Choice of \(W_1\).** The Wasserstein-1 distance is selected because: (i) the Kantorovich–Rubinstein duality \(W_1(\mu,\nu) = \sup_{\|f\|_{\mathrm{Lip}} \le 1} |\mathbb{E}_\mu[f] - \mathbb{E}_\nu[f]|\) provides a tractable dual reformulation for robust optimization; (ii) \(W_1\) metrizes weak convergence on compact spaces; and (iii) the resulting ambiguity set is convex, facilitating the convexity result of Theorem 7. Higher-order Wasserstein distances (\(W_p\), \(p > 1\)) would impose stronger regularity but lack the linear dual structure.

### Definition 34. Robust Q-function

\[
Q^{*,\boldsymbol{\Gamma},\rho}_\Pi(\widetilde b, u)
:=
\sup_{\pi \in \Pi} \inf_{\widetilde b' \in \mathcal{B}_\rho(\widetilde b)}
Q^{\pi,\boldsymbol{\Gamma}}(\widetilde b', u).
\]

---

## 7. Bluff

### Definition 35. Structural bluff

An action \(u_t\) is a structural bluff iff \(\mathrm{StructuralBluff}(c_t^S,u_t)=1\).

### Definition 36. Feasible action correspondence

\(\mathfrak A:\Delta(\widetilde{\mathcal X})\rightrightarrows \mathcal U.\)

### Definition 37. Feasible non-bluff action set

\[
\mathcal U_{\mathrm{nf}}(\widetilde b_t)
:=
\{
u\in\mathfrak A(\widetilde b_t)
\mid
\mathrm{StructuralBluff}(c_t^S,u)=0
\}.
\]

### Definition 38. Bluff gain (aggregate multiway form)

\[
\mathrm{Gain}_{\mathrm{bluff}}(\widetilde b,u;u^{\mathrm{cf}})
:=
Q^{*,\boldsymbol{\Gamma}^{\mathrm{attrib}}}_\Pi(\widetilde b,u)
-
Q^{*,\boldsymbol{\Gamma}^{\mathrm{ref}}}_\Pi(\widetilde b,u^{\mathrm{cf}}).
\]

### Definition 39. Exploitative bluff

An action \(u_t\) is an exploitative bluff iff: (1) it is a structural bluff; and (2) \(\sup_{u^{\mathrm{cf}}\in\mathcal U_{\mathrm{nf}}(\widetilde b_t)} \mathrm{Gain}_{\mathrm{bluff}}(\widetilde b_t,u_t;u^{\mathrm{cf}}) > 0\).

**Interpretive caveat.** The bluff gain as defined compares \(Q^*\) under the attributed kernel at action \(u\) against \(Q^*\) under the reference kernel at a different action \(u^{\mathrm{cf}}\). A positive bluff gain does not necessarily imply that the bluff action \(u\) derives its value from the bluff itself: if \(u\) also has high intrinsic value under the reference kernel (i.e., \(Q^{*,\boldsymbol{\Gamma}^{\mathrm{ref}}}(\widetilde b, u)\) is large), the gain conflates intrinsic action quality with manipulation rent. A pure measure of bluff value would require comparing the same action \(u\) under both kernels: \(Q^{*,\boldsymbol{\Gamma}^{\mathrm{attrib}}}(\widetilde b, u) - Q^{*,\boldsymbol{\Gamma}^{\mathrm{ref}}}(\widetilde b, u)\), which is precisely the per-rival manipulation rent \(\Delta_{\mathrm{manip}}^i\) (Definition 42) evaluated at \(u\). The current formulation captures "is the bluff better than the best non-bluff alternative?", not "how much value does the bluff mechanism add to this action?"

---

## 8. Strategic value decomposition

### 8.1 Per-rival analytical primitives

The canonical default background profile is \(\mathbf B^{-i}_{\mathrm{def}} := \boldsymbol{\Gamma}^{-i,\mathrm{attrib}}\).

### Definition 40. Total signal effect

\[
\Delta_{\mathrm{sig}}^{\,i\mid \mathbf B^{-i}}(\widetilde b,u)
:=
Q^{*,(\Gamma^{\mathrm{full},(\mathrm{attrib},1),i},\mathbf B^{-i})}_\Pi(\widetilde b,u)
-
Q^{*,(\Gamma^{\mathrm{full},(\mathrm{blind},1),i},\mathbf B^{-i})}_\Pi(\widetilde b,u).
\]

### Definition 41. Passive leakage

\[
\Delta_{\mathrm{pass}}^{\,i\mid \mathbf B^{-i}}(\widetilde b,u)
:=
Q^{*,(\Gamma^{\mathrm{full},(\mathrm{ref},1),i},\mathbf B^{-i})}_\Pi(\widetilde b,u)
-
Q^{*,(\Gamma^{\mathrm{full},(\mathrm{blind},1),i},\mathbf B^{-i})}_\Pi(\widetilde b,u).
\]

### Definition 42. Manipulation rent

\[
\Delta_{\mathrm{manip}}^{\,i\mid \mathbf B^{-i}}(\widetilde b,u)
:=
Q^{*,(\Gamma^{\mathrm{full},(\mathrm{attrib},1),i},\mathbf B^{-i})}_\Pi(\widetilde b,u)
-
Q^{*,(\Gamma^{\mathrm{full},(\mathrm{ref},1),i},\mathbf B^{-i})}_\Pi(\widetilde b,u).
\]

### Definition 43. Aggregate signal effect

\[
\Delta_{\mathrm{sig}}^{\mathrm{agg}}(\widetilde b,u)
:=
Q^{*,\boldsymbol{\Gamma}^{\mathrm{attrib}}}_\Pi(\widetilde b,u)
-
Q^{*,\boldsymbol{\Gamma}^{\mathrm{blind}}}_\Pi(\widetilde b,u).
\]

**Non-additivity warning.** \(\Delta_{\mathrm{sig}}^{\mathrm{agg}} \neq \sum_{i\in\mathcal R}\Delta_{\mathrm{sig}}^i\) in general. The discrepancy arises from cross-rival interaction effects. When rivals are conditionally independent given the public state (i.e., their kernels do not share hidden state beyond \(x^{\mathrm{pub}}\)), the additivity gap is bounded by the sum of pairwise interaction terms. A sufficient condition for exact additivity is that \(\boldsymbol{\Gamma}^{(\omega^{\mathrm{act}},\omega^{\mathrm{sd}})}\) decomposes as a product kernel \(\prod_{i \in \mathcal{R}} \Gamma^{\mathrm{full},(\omega^{\mathrm{act}},\omega^{\mathrm{sd}}),i}\) and the value function is additively separable across rivals.

### 8.2 World-index framework

SICFUN operates with two distinct value decompositions, each living in a different product space.

### Definition 44′. Chain space (for telescopic decomposition)

The **chain space** is

\[
\Omega^{\mathrm{chain}} := \Omega^{\mathrm{act}} \times \Omega^{\mathrm{sd}},
\]

where

\[
\Omega^{\mathrm{act}} := \{\mathrm{blind}, \mathrm{ref}, \mathrm{attrib}, \mathrm{design}\},
\qquad
\Omega^{\mathrm{sd}} := \{0, 1\}.
\]

This yields \(|\Omega^{\mathrm{chain}}| = 8\). It parameterizes the full kernel (Definition 20) and the telescopic edge/risk decomposition (§8.2A, §9D).

**Default telescopic chain:**

\[
(\mathrm{blind},0) \;\to\; (\mathrm{ref},0) \;\to\; (\mathrm{attrib},0) \;\to\; (\mathrm{attrib},1).
\]

**Coverage note.** The default chain visits 4 of the 8 elements of \(\Omega^{\mathrm{chain}}\). The remaining 4 worlds — \((\mathrm{design},0)\), \((\mathrm{design},1)\), \((\mathrm{ref},1)\), \((\mathrm{blind},1)\) — are not included in the canonical chain. Their omission is deliberate: (i) \((\mathrm{blind},1)\) is functionally identical to \((\mathrm{blind},0)\) (see effective cardinality note in Definition 20); (ii) \((\mathrm{ref},1)\) is accessible via an alternative chain if showdown-layer analysis of the reference world is needed; (iii) \((\mathrm{design},0)\) and \((\mathrm{design},1)\) enter through the signaling sub-decomposition (§8.4) rather than through the main telescopic chain. Alternative chains covering these worlds are valid instances of Proposition 8.1 and may be constructed as needed for diagnostic purposes.

### Definition 44. The four-world grid (for aggregate value decomposition)

The **grid space** is

\[
\Omega^{\mathrm{grid}} := \{\mathrm{blind}, \mathrm{attrib}\} \times \{\Pi^{\mathrm{ol}}, \Pi^S\}.
\]

The first coordinate selects the rival learning channel; the second selects the policy class over which the value optimum is taken. Define:

\[
V^{1,1}(\widetilde b) := V^{*,\boldsymbol{\Gamma}^{\mathrm{attrib}}}_{\Pi^S}(\widetilde b),
\qquad
V^{1,0}(\widetilde b) := V^{*,\boldsymbol{\Gamma}^{\mathrm{attrib}}}_{\Pi^{\mathrm{ol}}}(\widetilde b),
\]

\[
V^{0,1}(\widetilde b) := V^{*,\boldsymbol{\Gamma}^{\mathrm{blind}}}_{\Pi^S}(\widetilde b),
\qquad
V^{0,0}(\widetilde b) := V^{*,\boldsymbol{\Gamma}^{\mathrm{blind}}}_{\Pi^{\mathrm{ol}}}(\widetilde b),
\]

where the shorthand \(\boldsymbol{\Gamma}^{\mathrm{attrib}} = \boldsymbol{\Gamma}^{(\mathrm{attrib},1)}\) and \(\boldsymbol{\Gamma}^{\mathrm{blind}} = \boldsymbol{\Gamma}^{(\mathrm{blind},1)}\).

**Relationship between the two spaces.** The grid and chain spaces are **not** subsets of each other:

| Decomposition | Product space | Axes | What it decomposes |
|---|---|---|---|
| Four-world grid | \(\Omega^{\mathrm{grid}}\) | learning × policy scope | Value into control + signaling + interaction |
| Telescopic chain | \(\Omega^{\mathrm{chain}}\) | learning × showdown | Q-value into base + layer increments |

### Definition 45. Control value

\[
\Delta_{\mathrm{cont}}(\widetilde b) := V^{0,1}(\widetilde b)-V^{0,0}(\widetilde b).
\]

### Definition 46. Marginal signaling effect

\[
\Delta_{\mathrm{sig}}^*(\widetilde b) := V^{1,0}(\widetilde b)-V^{0,0}(\widetilde b).
\]

### Definition 47. Interaction term

\[
\Delta_{\mathrm{int}}(\widetilde b)
:=
V^{1,1}(\widetilde b) - V^{1,0}(\widetilde b) - V^{0,1}(\widetilde b) + V^{0,0}(\widetilde b).
\]

### 8.2A. Telescopic chain decomposition

### Definition 47A. Chain-indexed Q-function for baseline

For each \((\omega^{\mathrm{act}}, \omega^{\mathrm{sd}}) \in \Omega^{\mathrm{chain}}\),

\[
Q^{\bar\pi,(\omega^{\mathrm{act}},\omega^{\mathrm{sd}})}(\widetilde b, u)
:=
Q^{\bar\pi, \boldsymbol{\Gamma}^{(\omega^{\mathrm{act}},\omega^{\mathrm{sd}})}}(\widetilde b, u).
\]

### Definition 47B. Edge increment between chain worlds

For consecutive chain worlds \((\omega^{\mathrm{act}}_k, \omega^{\mathrm{sd}}_k) \to (\omega^{\mathrm{act}}_{k+1}, \omega^{\mathrm{sd}}_{k+1})\),

\[
\Delta^{\mathrm{edge}}_{k \to k+1}(\widetilde b, u)
:=
Q^{\bar\pi,(\omega^{\mathrm{act}}_{k+1},\omega^{\mathrm{sd}}_{k+1})}(\widetilde b, u)
-
Q^{\bar\pi,(\omega^{\mathrm{act}}_k,\omega^{\mathrm{sd}}_k)}(\widetilde b, u).
\]

### Proposition 8.1. Telescopic edge decomposition

For any ordered chain \((\omega^{\mathrm{act}}_0, \omega^{\mathrm{sd}}_0), \ldots, (\omega^{\mathrm{act}}_m, \omega^{\mathrm{sd}}_m)\) in \(\Omega^{\mathrm{chain}}\) and for all \(\widetilde b, u\):

\[
Q^{\bar\pi, (\omega^{\mathrm{act}}_m, \omega^{\mathrm{sd}}_m)}(\widetilde b, u)
=
Q^{\bar\pi, (\omega^{\mathrm{act}}_0, \omega^{\mathrm{sd}}_0)}(\widetilde b, u)
+
\sum_{k=0}^{m-1} \Delta^{\mathrm{edge}}_{k \to k+1}(\widetilde b, u).
\]

**Proof.** Telescoping identity. \(\square\)

**Canonical chain reading:**

| Increment | Interpretation |
|---|---|
| \(Q^{(\mathrm{ref},0)} - Q^{(\mathrm{blind},0)}\) | Edge from reference learning (no showdown) |
| \(Q^{(\mathrm{attrib},0)} - Q^{(\mathrm{ref},0)}\) | Edge from attribution correction (no showdown) |
| \(Q^{(\mathrm{attrib},1)} - Q^{(\mathrm{attrib},0)}\) | Edge from activating showdown |

### 8.3 Interpretive framework for the four-world decomposition

| SICFUN quantity | Layer | Interpretation |
|---|---|---|
| \(\Delta_{\mathrm{sig}}^*\) | Observational | Value of information passively revealed |
| \(\Delta_{\mathrm{cont}}\) | Interventional | Value change from strategic options |
| \(\Delta_{\mathrm{manip}}^i\) | Counterfactual | Advantage from rival belief reasoning |
| \(\Delta_{\mathrm{int}}\) | Cross-layer interaction | Residual |

### 8.4 Signaling sub-decomposition

### Definition 48. Design-signal effect

\[
\Delta_{\mathrm{sig,design}}^{\,i\mid \mathbf{B}^{-i}}(\widetilde b, u)
:=
Q^{*,(\Gamma^{\mathrm{full},(\mathrm{design},1),i},\,\mathbf{B}^{-i})}_\Pi(\widetilde b, u)
-
Q^{*,(\Gamma^{\mathrm{full},(\mathrm{blind},1),i},\,\mathbf{B}^{-i})}_\Pi(\widetilde b, u).
\]

### Definition 49. Realization-signal effect

\[
\Delta_{\mathrm{sig,real}}^{\,i\mid \mathbf{B}^{-i}}(\widetilde b, u)
:=
Q^{*,(\Gamma^{\mathrm{full},(\mathrm{attrib},1),i},\,\mathbf{B}^{-i})}_\Pi(\widetilde b, u)
-
Q^{*,(\Gamma^{\mathrm{full},(\mathrm{design},1),i},\,\mathbf{B}^{-i})}_\Pi(\widetilde b, u).
\]

### Definition 50. Canonical \(\Delta\)-vocabulary

**Per-rival primitives (grid-independent):** \(\Delta_{\mathrm{sig}}^i\), \(\Delta_{\mathrm{pass}}^i\), \(\Delta_{\mathrm{manip}}^i\), \(\Delta_{\mathrm{sig,design}}^i\), \(\Delta_{\mathrm{sig,real}}^i\).

**Aggregate grid primitives (\(\Omega^{\mathrm{grid}}\)):** \(\Delta_{\mathrm{cont}}\), \(\Delta_{\mathrm{sig}}^*\), \(\Delta_{\mathrm{int}}\), \(\Delta_{\mathrm{sig}}^{\mathrm{agg}}\).

**Chain primitives (\(\Omega^{\mathrm{chain}}\)):** \(\Delta^{\mathrm{edge}}_{k \to k+1}\), \(\Delta^{\mathrm{risk}}_{k \to k+1}\), \(\rho_{k \to k+1}\).

The grid decomposes optimal value \(V^{1,1}\) (Theorem 4). The chain decomposes baseline Q-value and robust local risk (Propositions 8.1, 9.7).

### 8.5 Stage-indexed reveal schedule

### Definition 51. Stage-indexed reveal schedule

For each decision stage \(\tau \in \mathcal{T}\) and each rival \(i\), define the optimal reveal threshold \(\tau_\tau^{*,i} \in \mathbb{R}\).

---

## 9. Structural results

### Theorem 1. Unconditional totality of the two-layer tempered update

If \(\delta_{\mathrm{floor}} > 0\) and \(\eta\) has full support, the posterior of Definition 15B is well-defined.

### Theorem 2. Posterior limits

(a) \(\kappa_{\mathrm{temp}} \to 1\) with \(\delta_{\mathrm{floor}} > 0\): converges to smoothed Bayes.
(b) \(\delta_{\mathrm{floor}} \to 0\) with \(\kappa_{\mathrm{temp}} < 1\): converges to pure power posterior.
(c) Joint limit: converges to standard Bayes on-path.

### Theorem 3. Exact per-rival signal decomposition under fixed background

\[
\Delta_{\mathrm{sig}}^{\,i\mid \mathbf B^{-i}}
=
\Delta_{\mathrm{pass}}^{\,i\mid \mathbf B^{-i}}
+
\Delta_{\mathrm{manip}}^{\,i\mid \mathbf B^{-i}}.
\]

### Theorem 3A. Signaling sub-decomposition

\[
\Delta_{\mathrm{sig}}^{\,i\mid \mathbf B^{-i}}
=
\Delta_{\mathrm{sig,design}}^{\,i\mid \mathbf B^{-i}}
+
\Delta_{\mathrm{sig,real}}^{\,i\mid \mathbf B^{-i}}.
\]

**Conditions for component annihilation.** The design-signal effect \(\Delta_{\mathrm{sig,design}}^i = 0\) when \(\Gamma^{\mathrm{act,design},i} = \Gamma^{\mathrm{act,blind},i}\), i.e., when the action-marginal likelihood carries no class-discriminative information. The realization-signal effect \(\Delta_{\mathrm{sig,real}}^i = 0\) when \(\Gamma^{\mathrm{act,attrib},i} = \Gamma^{\mathrm{act,design},i}\), i.e., when conditioning on the specific action taken adds no information beyond the action-marginal. In the limit where all actions are equally informative about class, \(\Delta_{\mathrm{sig,real}}^i\) dominates; when actions reveal class primarily through their aggregate frequency, \(\Delta_{\mathrm{sig,design}}^i\) dominates.

### Theorem 4. Exact aggregate value decomposition with interaction

\[
V^{1,1} = V^{0,0} + \Delta_{\mathrm{cont}} + \Delta_{\mathrm{sig}}^* + \Delta_{\mathrm{int}}.
\]

**Proof.** By Definitions 45–47:

\[
\Delta_{\mathrm{cont}} + \Delta_{\mathrm{sig}}^* + \Delta_{\mathrm{int}}
= (V^{0,1} - V^{0,0}) + (V^{1,0} - V^{0,0}) + (V^{1,1} - V^{1,0} - V^{0,1} + V^{0,0}).
\]

Expanding and canceling:

\[
= V^{0,1} - V^{0,0} + V^{1,0} - V^{0,0} + V^{1,1} - V^{1,0} - V^{0,1} + V^{0,0}
= V^{1,1} - V^{0,0}.
\]

Therefore \(V^{1,1} = V^{0,0} + \Delta_{\mathrm{cont}} + \Delta_{\mathrm{sig}}^* + \Delta_{\mathrm{int}}\). \(\square\)

### Theorem 5. Per-rival manipulation collapse under correct beliefs

If \(\Gamma^{\mathrm{full},(\mathrm{attrib},1),i} = \Gamma^{\mathrm{full},(\mathrm{ref},1),i}\), then \(\Delta_{\mathrm{manip}}^{\,i\mid \mathbf B^{-i}} \equiv 0\).

### Theorem 6. Coherence of the no-learning counterfactual

The no-learning world is obtained by restricting from \(\Pi^S\) to \(\Pi^{\mathrm{ol}}\), not by altering observation generation or rival dynamics.

### Theorem 7. Conditional convexity of robust value function under Wasserstein ambiguity

Under the following hypotheses (originally established in v0.30.4, repeated here for self-containment):

1. The abstracted state space \(\alpha_X(\widetilde{\mathcal X})\) is finite;
2. The belief-MDP transition is affine in the belief \(\widetilde b\);
3. The safety Bellman operator \(\mathcal{T}_{\mathrm{safe}}\) preserves convexity;
4. \(\rho > 0\) is fixed;

then \(V^{*,\boldsymbol{\Gamma},\rho}_\Pi(\widetilde b)\) is convex in \(\widetilde b\) for fixed \(\rho\).

### Theorem 8. Scalar adaptation safety bound (inherited)

Assume \(\beta \mapsto \mathrm{Exploit}(\pi^S_\beta)\) is lower-semicontinuous and quasiconvex on \([0,1]\). Then

\[
S(\varepsilon_{\mathrm{adapt}})
:=
\{\beta \in [0,1] : \mathrm{Exploit}(\pi^S_\beta) \le \varepsilon_{\mathrm{base}} + \varepsilon_{\mathrm{adapt}}\}
\]

is a closed interval containing \(0\).

**Scope note.** Subsumed by the local safety law of §9C when the full multiway framework is in use.

### Corollaries 1–4. (Inherited from v0.30.4.)

---

## 9A′. Formal exploitability

### Definition 52′. Joint rival profile class

\(\Sigma^{-S}\): class of admissible joint rival strategic profiles. If rivals are independent, \(\Sigma^{-S} = \prod_{i \in \mathcal{R}} \Pi^{R,i}\).

### Definition 52A. Robust performance functional

\[
J(\widetilde b; \pi, \sigma^{-S})
:=
\mathbb{E}^{\pi, \sigma^{-S}}\!\left[
\sum_{t=0}^{\infty} \gamma^t \, r(\widetilde X_t, u_t)
\;\middle|\;
\widetilde b_0 = \widetilde b
\right].
\]

### Definition 52B. Security value

\[
V^{\mathrm{sec}}(\widetilde b)
:=
\sup_{\pi \in \Pi^S}\;
\inf_{\sigma^{-S} \in \Sigma^{-S}}\;
J(\widetilde b; \pi, \sigma^{-S}).
\]

### Definition 52C. Pointwise exploitability

\[
\mathrm{Exploit}_{\widetilde b}(\pi)
:=
V^{\mathrm{sec}}(\widetilde b)
-
\inf_{\sigma^{-S} \in \Sigma^{-S}} J(\widetilde b; \pi, \sigma^{-S}).
\]

### Proposition 9.1. Non-negativity

\(\mathrm{Exploit}_{\widetilde b}(\pi) \ge 0\) for all \(\pi, \widetilde b\).

**Proof.** By Definition 52C, \(\mathrm{Exploit}_{\widetilde b}(\pi) = V^{\mathrm{sec}}(\widetilde b) - \inf_{\sigma^{-S}} J(\widetilde b; \pi, \sigma^{-S})\). By Definition 52B, \(V^{\mathrm{sec}}(\widetilde b) = \sup_{\pi'} \inf_{\sigma^{-S}} J(\widetilde b; \pi', \sigma^{-S}) \ge \inf_{\sigma^{-S}} J(\widetilde b; \pi, \sigma^{-S})\), since the supremum over all policies is at least as large as the value at any particular policy \(\pi\). Therefore \(\mathrm{Exploit}_{\widetilde b}(\pi) \ge 0\). \(\square\)

### Definition 52D. Uniform deployment exploitability

\[
\mathrm{Exploit}_{\mathcal{B}_{\mathrm{dep}}}(\pi)
:=
\sup_{\widetilde b \in \mathcal{B}_{\mathrm{dep}}}
\mathrm{Exploit}_{\widetilde b}(\pi).
\]

### Definition 53. Affine equilibrium deterrence

(Inherited.)

---

## 9B. Adaptation safety (strengthened)

### Definition 57. Adaptation safety — strong form (AS-strong)

A policy \(\pi \in \Pi^S\) satisfies adaptation safety with tolerance \(\varepsilon_{\mathrm{adapt}} \ge 0\) relative to baseline \(\bar\pi\) iff

\[
\forall\, \widetilde b \in \mathcal{B}_{\mathrm{dep}},\;
\forall\, \sigma^{-S} \in \Sigma^{-S},\quad
J(\widetilde b; \pi, \sigma^{-S})
\ge
J(\widetilde b; \bar\pi, \sigma^{-S})
-
\varepsilon_{\mathrm{adapt}}.
\tag{AS}
\]

### Definition 57A. Robust regret relative to baseline

\[
\mathrm{Reg}^{\mathrm{rob}}_{\widetilde b}(\pi \| \bar\pi)
:=
\sup_{\sigma^{-S} \in \Sigma^{-S}}
\big(
J(\widetilde b; \bar\pi, \sigma^{-S}) - J(\widetilde b; \pi, \sigma^{-S})
\big).
\]

### Proposition 9.2. AS-strong implies relative exploitability bound

If \(\pi\) satisfies (AS), then \(\mathrm{Exploit}_{\widetilde b}(\pi) \le \mathrm{Exploit}_{\widetilde b}(\bar\pi) + \varepsilon_{\mathrm{adapt}}\).

### Corollary 9.3. Total vulnerability budget

If \(\mathrm{Exploit}_{\mathcal{B}_{\mathrm{dep}}}(\bar\pi) \le \varepsilon_{\mathrm{base}}\), then \(\mathrm{Exploit}_{\mathcal{B}_{\mathrm{dep}}}(\pi) \le \varepsilon_{\mathrm{base}} + \varepsilon_{\mathrm{adapt}}\).

**Computability note.** Verifying the hypothesis \(\mathrm{Exploit}_{\mathcal{B}_{\mathrm{dep}}}(\bar\pi) \le \varepsilon_{\mathrm{base}}\) requires computing \(\sup_{\widetilde b \in \mathcal{B}_{\mathrm{dep}}} [V^{\mathrm{sec}}(\widetilde b) - \inf_{\sigma^{-S}} J(\widetilde b; \bar\pi, \sigma^{-S})]\), which is a minimax problem over the deployment belief set and the rival profile class. In general, this is computationally intractable without further structure. Tractable relaxations include: (i) restricting \(\mathcal{B}_{\mathrm{dep}}\) to a finite representative set of beliefs; (ii) restricting \(\Sigma^{-S}\) to a finite or parametric class; (iii) using the structural certificate \(B_\beta\) (Definition 65) as a conservative upper bound, which is verifiable via a finite linear program when the spot classifier \(s\) has finite range. The local Bellman-safe law (§9C) provides a sufficient condition that avoids direct computation of global exploitability.

### Definition 57B. Adaptation-safe policy class

\[
\Pi^S_{\mathrm{safe}}(\bar\pi, \varepsilon_{\mathrm{adapt}}, \mathcal{B}_{\mathrm{dep}})
:=
\big\{\pi \in \Pi^S : \text{(AS) holds}\big\}.
\]

### Definition 57C. Safe optimal value under chain world

\[
V^{\mathrm{safe},(\omega^{\mathrm{act}},\omega^{\mathrm{sd}})}(\widetilde b)
:=
\sup_{\pi \in \Pi^S_{\mathrm{safe}}}
V^{\pi,\boldsymbol{\Gamma}^{(\omega^{\mathrm{act}},\omega^{\mathrm{sd}})}}(\widetilde b).
\]

---

## 9C. Local Bellman-safe law

### Definition 58. One-step baseline loss

\[
\ell_{\bar\pi}(\widetilde b, u; \sigma^{-S})
:=
V^{\bar\pi}(\widetilde b; \sigma^{-S})
-
Q^{\bar\pi}(\widetilde b, u; \sigma^{-S}).
\]

### Definition 59. Robust one-step loss

\[
\ell^{\mathrm{rob}}_{\bar\pi}(\widetilde b, u)
:=
\sup_{\sigma^{-S} \in \Sigma^{-S}}
\big[\ell_{\bar\pi}(\widetilde b, u; \sigma^{-S})\big]_+.
\]

### Definition 60. Safety Bellman operator

\[
(\mathcal{T}_{\mathrm{safe}} B)(\widetilde b)
:=
\begin{cases}
0,
& \widetilde b \in \mathcal{B}_{\mathrm{term}},
\\[6pt]
\displaystyle\inf_{u \in \mathcal{U}}
\Big[
\ell^{\mathrm{rob}}_{\bar\pi}(\widetilde b, u)
+
\gamma \sup_{\sigma^{-S}} \mathbb{E}\big[B(\widetilde b_1) \mid \widetilde b_0 = \widetilde b, u_0 = u, \sigma^{-S}\big]
\Big],
& \widetilde b \notin \mathcal{B}_{\mathrm{term}}.
\end{cases}
\]

**Default convention.** The unqualified \(\mathcal{T}_{\mathrm{safe}}\) uses the operational world \((\mathrm{attrib}, 1)\). The chain-indexed version is \(\mathcal{T}^{(\omega^{\mathrm{act}},\omega^{\mathrm{sd}})}_{\mathrm{safe}}\), replacing the loss and expectation with their chain-world counterparts.

### Definition 61. Minimal safety certificate

\[
B^\star = \mathcal{T}_{\mathrm{safe}} B^\star.
\]

\(B^\star(\widetilde b)\) is the minimum robust degradation budget from \(\widetilde b\) onward. The chain-indexed version is \(B^{\star,(\omega^{\mathrm{act}},\omega^{\mathrm{sd}})}\). The unqualified \(B^\star := B^{\star,(\mathrm{attrib},1)}\).

### Proposition 9.5. Existence and uniqueness of \(B^\star\)

If \(\gamma \in (0,1)\), \(\ell^{\mathrm{rob}}_{\bar\pi}\) is bounded, and belief-level kernels are well-defined, then \(\mathcal{T}_{\mathrm{safe}}\) is a \(\gamma\)-contraction. \(B^\star\) exists and is unique.

**Derivation of \(\ell^{\mathrm{rob}}\) boundedness.** By Definition 59, \(\ell^{\mathrm{rob}}_{\bar\pi}(\widetilde b, u) = \sup_{\sigma^{-S}} [V^{\bar\pi}(\widetilde b; \sigma^{-S}) - Q^{\bar\pi}(\widetilde b, u; \sigma^{-S})]_+\). By A5, \(|r| \le R_{\max}\), so \(|V^{\bar\pi}| \le R_{\max}/(1-\gamma)\) and \(|Q^{\bar\pi}| \le R_{\max}/(1-\gamma)\). Therefore \(\ell^{\mathrm{rob}}_{\bar\pi}(\widetilde b, u) \le 2R_{\max}/(1-\gamma)\) for all \(\widetilde b, u\), regardless of \(\sigma^{-S}\). The supremum over \(\Sigma^{-S}\) is finite because it is bounded above by this constant. This holds without any regularity assumption on \(\Sigma^{-S}\) beyond non-emptiness.

### Definition 62. Canonical safe action set

\[
\mathcal{U}^\star_{\mathrm{safe}}(\widetilde b)
:=
\Big\{
u \in \mathcal{U}
:
\ell^{\mathrm{rob}}_{\bar\pi}(\widetilde b, u)
+
\gamma \sup_{\sigma^{-S}} \mathbb{E}\big[B^\star(\widetilde b_1) \mid \widetilde b, u, \sigma^{-S}\big]
\le
B^\star(\widetilde b)
\Big\}.
\]

The chain-indexed version is \(\mathcal{U}^{\star,(\omega^{\mathrm{act}},\omega^{\mathrm{sd}})}_{\mathrm{safe}}(\widetilde b)\).

### Definition 63. Safe-feasible policy under chain world

\[
\pi^{\mathrm{safe},(\omega^{\mathrm{act}},\omega^{\mathrm{sd}})}(\widetilde b)
\in
\arg\max_{u \in \mathcal{U}^{\star,(\omega^{\mathrm{act}},\omega^{\mathrm{sd}})}_{\mathrm{safe}}(\widetilde b)}
Q^{(\omega^{\mathrm{act}},\omega^{\mathrm{sd}})}(\widetilde b, u).
\]

### Theorem 9. Local-to-global sufficiency

If \(\pi\) selects \(\pi(\widetilde b) \in \mathcal{U}^\star_{\mathrm{safe}}(\widetilde b)\) for all relevant beliefs, then for all \(\widetilde b \in \mathcal{B}_{\mathrm{dep}}\) and all \(\sigma^{-S} \in \Sigma^{-S}\),

\[
V^{\pi}(\widetilde b; \sigma^{-S})
\ge
V^{\bar\pi}(\widetilde b; \sigma^{-S})
-
B^\star(\widetilde b).
\]

If \(\sup_{\widetilde b \in \mathcal{B}_{\mathrm{dep}}} B^\star(\widetilde b) \le \varepsilon_{\mathrm{adapt}}\), then \(\pi\) satisfies (AS).

**Proof.** Define \(D^\pi := V^{\bar\pi} - V^{\pi}\). Recursive expansion: \(D^\pi(\widetilde b; \sigma^{-S}) = \ell_{\bar\pi}(\widetilde b, u; \sigma^{-S}) + \gamma\,\mathbb{E}[D^\pi(\widetilde b_1; \sigma^{-S})]\). Induction with \(D^\pi(\widetilde b_1) \le B^\star(\widetilde b_1)\) and worst-case over \(\sigma^{-S}\) gives \(D^\pi(\widetilde b) \le B^\star(\widetilde b)\). \(\square\)

### Definition 64. Required adaptation budget

\[
\varepsilon^\star_{\mathrm{adapt}}
:=
\sup_{\widetilde b \in \mathcal{B}_{\mathrm{dep}}}
B^\star(\widetilde b).
\]

Deployment is feasible iff \(\varepsilon^\star_{\mathrm{adapt}} \le \varepsilon_{\mathrm{adapt}}\).

### Definition 65. Structural certificate (deployable form)

Let \(s : \mathcal{B}_{\mathrm{dep}} \to \mathcal{S}\) be a finite structural spot classifier and \(\tau : \mathcal{B}_{\mathrm{dep}} \to \mathcal{T}\) the public stage. For \(\beta \in \mathbb{R}_{\ge 0}^{\mathcal{S} \times \mathcal{T}}\),

\[
B_\beta(\widetilde b) := \beta_{s(\widetilde b),\, \tau(\widetilde b)}.
\]

**Structural constraints:** terminality, non-negativity, global bound, horizon monotonicity.

### Definition 66. Certification of \(B_\beta\)

\(B_\beta\) is valid iff \(B_\beta(\widetilde b) \ge (\mathcal{T}_{\mathrm{safe}} B_\beta)(\widetilde b)\) for all \(\widetilde b \in \mathcal{B}_{\mathrm{dep}}\).

### Proposition 9.6. Dominance

If \(B_\beta\) satisfies certification, then \(B^\star(\widetilde b) \le B_\beta(\widetilde b)\) for all \(\widetilde b\).

---

## 9D. World-aware risk decomposition

### Definition 67. Chain-indexed robust one-step loss

For each \((\omega^{\mathrm{act}}, \omega^{\mathrm{sd}}) \in \Omega^{\mathrm{chain}}\),

\[
\ell^{\mathrm{rob},(\omega^{\mathrm{act}},\omega^{\mathrm{sd}})}_{\bar\pi}(\widetilde b, u)
:=
\sup_{\sigma^{-S} \in \Sigma^{-S}}
\big[V^{\bar\pi,(\omega^{\mathrm{act}},\omega^{\mathrm{sd}})}(\widetilde b; \sigma^{-S}) - Q^{\bar\pi,(\omega^{\mathrm{act}},\omega^{\mathrm{sd}})}(\widetilde b, u; \sigma^{-S})\big]_+.
\]

### Definition 68. Risk increment between chain worlds

\[
\Delta^{\mathrm{risk}}_{k \to k+1}(\widetilde b, u)
:=
\ell^{\mathrm{rob},(\omega^{\mathrm{act}}_{k+1},\omega^{\mathrm{sd}}_{k+1})}_{\bar\pi}(\widetilde b, u)
-
\ell^{\mathrm{rob},(\omega^{\mathrm{act}}_k,\omega^{\mathrm{sd}}_k)}_{\bar\pi}(\widetilde b, u),
\]

where \((\omega^{\mathrm{act}}_k, \omega^{\mathrm{sd}}_k), (\omega^{\mathrm{act}}_{k+1}, \omega^{\mathrm{sd}}_{k+1}) \in \Omega^{\mathrm{chain}}\) are consecutive chain worlds.

### Proposition 9.7. Telescopic risk decomposition

\[
\ell^{\mathrm{rob},(\omega^{\mathrm{act}}_m,\omega^{\mathrm{sd}}_m)}_{\bar\pi}(\widetilde b, u)
=
\ell^{\mathrm{rob},(\omega^{\mathrm{act}}_0,\omega^{\mathrm{sd}}_0)}_{\bar\pi}(\widetilde b, u)
+
\sum_{k=0}^{m-1}
\Delta^{\mathrm{risk}}_{k \to k+1}(\widetilde b, u).
\]

**Sign convention.** Each \(\Delta^{\mathrm{risk}}_{k \to k+1}\) may be positive, negative, or zero.

### Definition 69. Marginal layer efficiency (audit metric, non-constitutive)

\[
\rho_{k \to k+1}(\widetilde b, u)
:=
\frac{\Delta^{\mathrm{edge}}_{k \to k+1}(\widetilde b, u)}{[\Delta^{\mathrm{risk}}_{k \to k+1}(\widetilde b, u)]_+}
\]

when \([\Delta^{\mathrm{risk}}_{k \to k+1}]_+ > 0\).

---

## 10. Computational architecture (non-normative)

### Definition 54. Particle belief representation

\[
\widetilde b_t \approx \hat{b}_t^C := \Big\{ \big(\widetilde x_t^{(j)}, w_t^{(j)}\big) \Big\}_{j=1}^{C}.
\]

### Definition 55. Particle belief MDP error bound

\[
\big| V^*(\widetilde b) - V^*(\hat{b}^C) \big|
\le
\frac{R_{\max}}{1-\gamma}
\cdot
\sqrt{2\,D_2(\widetilde b \| \hat{b}^C)}.
\]

**Derivation sketch.** The bound follows from Pinsker-type inequalities: \(\|V^*(\widetilde b) - V^*(\hat b^C)\| \le \frac{R_{\max}}{1-\gamma} \cdot \mathrm{TV}(\widetilde b, \hat b^C)\) by the Lipschitz property of the value function under total variation, and \(\mathrm{TV}(\mu,\nu) \le \sqrt{D_2(\mu \| \nu)/2}\) by a Rényi-divergence generalization of Pinsker's inequality (see, e.g., van Erven & Harremoës, 2014). Composing these two inequalities yields the stated bound.

### Definition 56. Factored particle filtering

\[
\hat{b}_t^C = \hat{b}_t^{C,\mathrm{pub}} \otimes \bigotimes_{i \in \mathcal{R}} \hat{b}_t^{C_i, i}.
\]

### 10B. Neural belief architecture recommendation

Linear recurrent architectures (S4/LRU family) recommended over Transformers for belief tracking.

**Qualification.** This recommendation applies when: (i) the belief update is approximately Markovian (consistent with A2), making fixed-dimensional state sufficient; (ii) the observation sequence is long (horizons \(T \gg 100\)), where Transformers' quadratic attention cost becomes prohibitive; and (iii) the factored particle structure (Definition 56) requires per-rival state tracking, which maps naturally to independent recurrent channels. For short-horizon problems or tasks requiring cross-temporal attention over sparse events, Transformer architectures may be preferable.

---

## 11. Canonical form

### Proposition 1

Under assumptions (A1′)–(A10), SICFUN-v0.31.1 is a first-order interactive partially observable decision process over \(\widetilde{\mathcal X}\), with:

1. unconditionally total inferential updating under the default two-layer tempered semantics;
2. explicit separation between action-channel kernels, design-signal kernels, and full kernels parameterized by \((\omega^{\mathrm{act}}, \omega^{\mathrm{sd}}) \in \Omega^{\mathrm{chain}}\);
3. per-rival exact signal decomposition with sub-decomposition into design-signal and realization-signal;
4. aggregate four-world value decomposition in \(\Omega^{\mathrm{grid}}\) with explicit interaction term;
5. telescopic edge/risk decomposition in \(\Omega^{\mathrm{chain}}\), exact at local action level;
6. multiway semantics with formal joint rival profile \(\Sigma^{-S}\);
7. formal multiway exploitability as gap from \(V^{\mathrm{sec}}\);
8. adaptation safety as uniform robust dominance (AS-strong);
9. local Bellman-safe law via \(\mathcal{T}_{\mathrm{safe}}\), \(B^\star\), and \(\mathcal{U}^\star_{\mathrm{safe}}\);
10. deployable structural certificate \(B_\beta\) with supersolution certification;
11. changepoint detection with changepoint-triggered retreat;
12. distributionally robust planning via Wasserstein ambiguity sets;
13. three orthogonal world axes with two non-overlapping product spaces;
14. particle belief MDP approximation with factored structure (non-normative).

\[
\boxed{\text{SICFUN is a first-order multiway interactive 13-tuple model over augmented hidden state}}
\]

\[
\boxed{\text{with two-layer tempered per-rival updating through action, design-signal, and showdown channels}}
\]

\[
\boxed{\text{with exact decomposition of strategic value via the four-world grid and the telescopic chain}}
\]

\[
\boxed{\text{with adaptation safety via Bellman-safe certification and formal multiway exploitability}}
\]

\[
\boxed{\text{and world-indexed local decomposition of edge and robust risk per modeling layer}}
\]

---

## 12. Relationship to prior versions

### 12.1 Changes from v0.31

| v0.31 element | v0.31.1 fix |
|---|---|
| Single \(\Omega = \Omega^{\mathrm{act}} \times \Omega^{\mathrm{sd}}\) conflating two decompositions | Two product spaces: \(\Omega^{\mathrm{chain}}\) (learning × showdown) and \(\Omega^{\mathrm{grid}}\) (learning × policy scope) |
| \(\Omega^{\mathrm{act}} = \{3 \text{ elements}\}\) in §8.2, \(\{4 \text{ elements}\}\) in A7/Def 20 | Canonical \(\Omega^{\mathrm{act}} = \{\mathrm{blind}, \mathrm{ref}, \mathrm{attrib}, \mathrm{design}\}\) everywhere |
| Four worlds declared as \(\Omega^\star \subset \Omega\) (type error) | Four worlds in \(\Omega^{\mathrm{grid}}\), not subset of \(\Omega^{\mathrm{chain}}\) |
| \(\omega^{\mathrm{sd}}\) not wired into Definition 20 | Full kernel \(\Gamma^{\mathrm{full},(\omega^{\mathrm{act}},\omega^{\mathrm{sd}}),i}\) with explicit showdown gating |
| Showhand profiles implicit | Shorthand \(\boldsymbol{\Gamma}^{\mathrm{attrib}} := \boldsymbol{\Gamma}^{(\mathrm{attrib},1)}\) explicit |
| 12-tuple without design-signal kernel | 13-tuple with \(\{\Gamma^{\mathrm{act,design},i}\}_{i \in \mathcal{R}}\) as constitutive component |

### 12.2 Backward compatibility

All v0.30.4 and v0.31 definitions, theorems, and corollaries are preserved. The v0.30.4 kernel behavior is recovered by fixing \(\omega^{\mathrm{sd}} = 1\). All v0.31 shorthands are the showdown-on specialization. The 13th constitutive component (\(\Gamma^{\mathrm{act,design},i}\)) was already defined in v0.31 as Definition 19A; its elevation to the constitutive tuple is a reclassification, not a new object.

---

## 13. Provenance

This specification is the world-algebra-closed successor of v0.31. Its mathematical structure is self-contained, with 69 definitions, 9 theorems, 5 corollaries, 12 assumptions (in 10 families: A1′, A2, A3/A3′, A4′, A5, A6/A6′, A7, A8, A9, A10), and 2 canonical propositions. The three-axis world framework resolves the type inconsistency between the four-world grid and the telescopic chain, completing the formal closure of the system.

SICFUN-v0.31.1 is the formally closed revision.

---

## Appendix A. Definition index

**Ordering note.** Definitions 54–56 (§10, computational architecture) are listed in numerical order below, but appear in the document body after §9D (Definitions 67–69). This reflects the historical insertion of §9B–§9D after §10 was established. A clean renumbering is deferred to the next major version.

| # | Name | Section |
|---|---|---|
| 1 | Private strategic classes | §3 |
| 2 | Current private strategic class | §3 |
| 3 | Aggressive-wager predicate | §3 |
| 4 | Structural-bluff predicate | §3 |
| 5 | Size-aware public action signal | §3 |
| 6 | Total public signal | §3 |
| 7 | Observation object and canonical identification | §3 |
| 8 | Signal routing convention | §3 |
| 9 | Real baseline of SICFUN | §3 |
| 10 | Attributed baseline (with partial-policy typing) | §3 |
| 11 | Reputational projection | §3 |
| 12 | Augmented hidden state space | §3 |
| 13 | Augmented hidden state | §3 |
| 14 | Operative belief | §3 |
| 15 | Tempering exponent and safety floor | §4.1 |
| 15A | Two-layer tempered likelihood | §4.1 |
| 15B | Posterior-on-class update | §4.1 |
| 15C | Exploitation interpolation parameter | §4.1 |
| 16 | State-embedding updater | §4.2 |
| 17 | BuildRivalKernel | §4.2 |
| 18 | Inferential action kernels (partial-policy typed) | §4.2 |
| 19 | Showdown kernel | §4.2 |
| 19A | Design-signal kernel | §4.2 |
| 20 | Full per-rival kernels (\(\Omega^{\mathrm{chain}}\)-indexed) | §4.2 |
| 21 | Joint kernel profiles (\(\Omega^{\mathrm{chain}}\)-indexed) | §4.2 |
| 22 | Belief update | §5 |
| 23 | Full rival-state update (\(\Omega^{\mathrm{chain}}\)-indexed) | §5 |
| 24 | Counterfactual reference world | §5 |
| 25 | Spot-conditioned polarization | §5 |
| 26 | Changepoint detection module | §5A |
| 27 | Run-length posterior update | §5A |
| 28 | Changepoint-triggered prior reset | §5A |
| 29 | Belief-averaged reward | §6 |
| 30 | Policy-evaluation Q-function | §6 |
| 31 | Value under policy | §6 |
| 32 | Optimal Q-function over a policy class | §6 |
| 33 | Ambiguity set | §6A |
| 34 | Robust Q-function | §6A |
| 35 | Structural bluff | §7 |
| 36 | Feasible action correspondence | §7 |
| 37 | Feasible non-bluff action set | §7 |
| 38 | Bluff gain | §7 |
| 39 | Exploitative bluff | §7 |
| 40 | Total signal effect | §8.1 |
| 41 | Passive leakage | §8.1 |
| 42 | Manipulation rent | §8.1 |
| 43 | Aggregate signal effect | §8.1 |
| 44′ | Chain space \(\Omega^{\mathrm{chain}}\) | §8.2 |
| 44 | Four-world grid \(\Omega^{\mathrm{grid}}\) | §8.2 |
| 45 | Control value | §8.2 |
| 46 | Marginal signaling effect | §8.2 |
| 47 | Interaction term | §8.2 |
| 47A | Chain-indexed Q-function for baseline | §8.2A |
| 47B | Edge increment between chain worlds | §8.2A |
| 48 | Design-signal effect | §8.4 |
| 49 | Realization-signal effect | §8.4 |
| 50 | Canonical Δ-vocabulary (grid + chain) | §8.4 |
| 51 | Stage-indexed reveal schedule | §8.5 |
| 52′ | Joint rival profile class | §9A′ |
| 52A | Robust performance functional | §9A′ |
| 52B | Security value | §9A′ |
| 52C | Pointwise exploitability | §9A′ |
| 52D | Uniform deployment exploitability | §9A′ |
| 53 | Affine equilibrium deterrence | §9A′ |
| 54 | Particle belief representation | §10A |
| 55 | Particle belief MDP error bound | §10A |
| 56 | Factored particle filtering | §10A |
| 57 | Adaptation safety — strong form | §9B |
| 57A | Robust regret relative to baseline | §9B |
| 57B | Adaptation-safe policy class | §9B |
| 57C | Safe optimal value under chain world | §9B |
| 58 | One-step baseline loss | §9C |
| 59 | Robust one-step loss | §9C |
| 60 | Safety Bellman operator (chain-indexable) | §9C |
| 61 | Minimal safety certificate \(B^\star\) (chain-indexable) | §9C |
| 62 | Canonical safe action set (chain-indexable) | §9C |
| 63 | Safe-feasible policy under chain world | §9C |
| 64 | Required adaptation budget | §9C |
| 65 | Structural certificate (deployable) | §9C |
| 66 | Certification of \(B_\beta\) | §9C |
| 67 | Chain-indexed robust one-step loss | §9D |
| 68 | Risk increment between chain worlds | §9D |
| 69 | Marginal layer efficiency | §9D |

---

## Appendix B. Audit note: endogenous changepoint vulnerability

(Inherited from v0.30.4.)

**Vulnerability description.** The changepoint detection module (§5A) assumes that changepoints are exogenous: the hazard rate \(h^i\) is a property of the rival's type dynamics, independent of SICFUN's actions. However, a sophisticated rival could trigger an apparent changepoint endogenously — for example, by deliberately shifting behavior to cause SICFUN's detector to fire, resetting the posterior \(\mu^{R,i}\) toward the meta-prior \(\nu_{\mathrm{meta}}^i\) (Definition 28). This creates a manipulation vector: the rival can "wash" SICFUN's accumulated information by inducing false changepoints.

**Impact.** If a rival can reliably trigger changepoint detection, the effective learning horizon is bounded by the rival's patience, not by SICFUN's observation window. This could degrade the convergence properties assumed in the exploitation framework and reduce the effective value of the signal decomposition.

**Mitigation (current).** The retreat rate \(\delta_{\mathrm{cp\text{-}retreat}}\) (Definition 15C) and the soft reset parameter \(w_{\mathrm{reset}}\) (Definition 28) provide partial protection: a soft reset (\(w_{\mathrm{reset}} < 1\)) preserves some posterior information even after a detected changepoint. Additionally, the adaptation safety framework (§9B–§9C) bounds the worst-case loss from any exploitation, including exploitation degraded by false changepoints.

**Open issue.** A formal analysis of the strategic interaction between SICFUN's changepoint detector and a rival's endogenous changepoint-triggering strategy is not provided in the current specification. Such an analysis would require modeling the rival's cost of inducing a false changepoint and SICFUN's best response in the resulting meta-game.

---

## Appendix C. World algebra design rationale

The world-index system has three orthogonal axes because SICFUN's value depends on three independent choices:

1. **How does the rival learn?** → \(\omega^{\mathrm{act}} \in \{\mathrm{blind}, \mathrm{ref}, \mathrm{attrib}, \mathrm{design}\}\)
2. **Is terminal revelation processed?** → \(\omega^{\mathrm{sd}} \in \{0, 1\}\)
3. **What policy class does SICFUN optimize over?** → \(\omega^{\mathrm{pol}} \in \{\Pi^{\mathrm{ol}}, \Pi^S\}\)

No single product of two of these axes captures everything. The four-world grid (Theorem 4) uses axes 1 × 3 to decompose *why* value exists — restricting the learning channel to the extremes \(\{\mathrm{blind}, \mathrm{attrib}\}\) because the grid decomposes aggregate value into control, signaling, and interaction, which only requires the no-learning and full-learning endpoints. The intermediate channels (\(\mathrm{ref}\) and \(\mathrm{design}\)) contribute to the telescopic chain, not to the aggregate grid. The telescopic chain (Propositions 8.1, 9.7) uses axes 1 × 2 to decompose *where* edge and risk come from, layer by layer. The two decompositions coexist and complement each other.

A hypothetical full product \(\Omega^{\mathrm{full}} := \Omega^{\mathrm{act}} \times \Omega^{\mathrm{sd}} \times \Omega^{\mathrm{pol}}\) would have \(4 \times 2 \times 2 = 16\) combinations. This is well-defined but unnecessary: no theorem in the current specification requires simultaneous variation of all three axes. The notation \(\Omega^{\mathrm{full}}\) is reserved for future extensions that may require the complete product space.
