# Lesson 7 · What is (and isn't) new

> A portfolio project is easy to oversell. This lesson does the opposite on purpose: it places
> every result of the course in the published work it rests on, says which results only
> reproduce known conclusions, and states the narrow part that seems to be new, with the
> searches still needed before anyone should believe it. It ends with what the project cannot
> claim, and where the next one picks up.

**Note:** [`docs/related-work.md`](../related-work.md) (every claim checked against its source) ·
**Report:** [`docs/report.md`](../report.md) ·
**Previous:** [Lesson 6](06-testing-on-someone-elses-plant.md) ·
**Course index:** [README](README.md)

## Learning objectives

By the end of this lesson you should be able to:

1. Place each technique of the course in its literature: static stealth, dynamic stealth,
   inertial spoofing monitors, fault-versus-attack classification, testbed data.
2. Distinguish a reproduction, an application and a contribution, and write a novelty
   statement that survives a reviewer.
3. Explain the relation between this project's "meters to alter" count and the security index.
4. State the limits of the results, and what a follow-on project would have to do.

---

## 1. A map of the course against the literature

| course result | where it comes from | status here |
|---|---|---|
| $a = Hc$ passes the bad-data test (Lessons 1, 4, 5) | Liu, Ning and Reiter, 2009 | reproduced, including on IEEE 14-bus |
| chi-square bad-data test, noncentral detection law | classical state estimation (Monticelli, 1999) | used, checked by Monte Carlo |
| a model trajectory leaves Kalman innovations unchanged (Lesson 5) | Mo and Sinopoli, 2010; Pasqualetti, Dörfler and Bullo, 2013; Teixeira *et al.*, 2015 | special case, proved for completeness |
| the INS exposes receivers-only spoofing through filter innovations (Lessons 4, 5) | Khanafseh *et al.*, 2014; Tanıl *et al.*, 2018 | known conclusion, reproduced on a simpler model |
| classifying anomalies as faults or attacks (Lesson 3) | Anwar, Mahmood and Shah, 2015; Roy and Dey, 2023 | application |
| PCA residual on testbed data (Lesson 6) | Jackson and Mudholkar, 1979; HAI (Shin *et al.*, 2020) | application |

Most of the course is the second and third columns: the established results, built from first
principles and checked numerically. That is what a course should be.

## 2. Reproduction, application, contribution

Three words that a novelty statement must keep apart:

* **Reproduction**: obtaining a published result again, here usually by deriving it and checking it
  in code (the stealthy-attack theorem, the INS monitor's conclusion).
* **Application**: using a known method on a new system or data set (gradient boosting on physics
  residuals, the PCA residual on HAI).
* **Contribution**: something a reader could not have taken from the literature.

A reviewer reads "we propose a novel detector" and asks what is novel about it. If the honest
answer is "the system it runs on", the claim is an application and should be written as one.

## 3. Meters to alter and the security index

Lesson 5 counted, for each bus of the IEEE 14-bus system, the meters an attacker must alter to
shift that bus's angle *alone*: the nonzeros of column $j$ of $H$. Sandberg, Teixeira and Johansson
(2010) defined the security index of a state as the fewest meters any stealthy attack needs while
moving that state, allowing the other states to move too:

```math
\alpha_{j} = \min_{c} \lVert Hc \rVert_{0} \quad \text{subject to} \quad c_{j} = 1.
```

The count of Lesson 5 is the feasible point $c = e_{j}$, so it is an upper bound: $\alpha_{j} \le \lVert He_{j} \rVert_{0}$. Moving a
neighbouring angle along with bus $j$ can cancel some meters, so the true index may be smaller. The
lesson's table is therefore a ranking of how exposed each angle is to the simplest stealthy attack,
not the index itself. The $\ell_{0}$ minimisation is a combinatorial problem; later work in that line
studied how to compute it for large networks.

## 4. What seems to be new, stated narrowly

Searches for the combination below did not find it as one published piece. That is a statement
about the searches, not a proof of absence.

1. **Attribution along a stealth dial.** One reproducible pipeline that sweeps a single attack from
   a one-channel offset to a fully consistent injection, holding its intended effect fixed, and
   measures detection *and* fault-versus-attack attribution along the way. Where a closed form
   exists it is computed and checked: the $(1-s)^{2}$ noncentrality law and exact scenario
   false-alarm rates.
2. **Detection in the units of an attack tree.** CPS-02 assumed a 0.1 miss probability for the
   navigation integrity check. This project measures it per attacker reach (0 for crude attacks,
   0.03 to 0.51 for receivers-only spoofing depending on the monitor, about 1 for a bus-level
   consistent injection), on the same synthetic system that CPS-01 and CPS-02 analyse in
   reliability and attack-tree terms.
3. **Physics cues versus habit cues.** The mimicry experiment separates what a strong classifier
   learns from physics from what it learns from the fault model and the assumed attacker
   magnitudes, and the Bayes argument bounds what any classifier can do against an exact mimic.

None of the three is a new algorithm or theorem. They are a way of evaluating, and a link between
detection and reliability modelling, that a reader could use.

## 5. What the project cannot claim

* **Absolute rates on real systems.** The navigation chain, its noise and its anomaly ranges are
  assumptions. The rates are conditional on them; comparisons and closed-form limits travel,
  absolute numbers do not.
* **Attribution on real data.** HAI has no labelled physical faults; no public data set used here
  does. The fault-versus-attack accuracy of Lesson 3 is measured on synthetic data only.
* **Robustness to an informed attacker.** Lesson 4 shows the opposite: an attacker who knows the
  fault model copies it, and one who reaches every sensor of a direction is invisible in it.
* **Anything about access.** By scope, the project models values on the bus, not how they got there
  ([`scope.md`](../scope.md)).

## 6. Where the next project picks up

Two threads lead out:

* **Physics-informed models for real plants.** Lesson 6 showed that a data-driven check cannot tell
  a new operating point from an attack. A model that knows the plant's physics, a first-principles
  model or a physics-informed neural network, would keep a new operating point "normal" by
  construction. That is the follow-on project in the series (process-physics attack detection).
* **Design rather than detection.** Lesson 5 showed that the bus-level attacker is stopped only by a
  witness out of reach, and only in the directions it observes. Which independent sources to add,
  and on which data path, is a design question for reliability and attack-tree models: back to
  CPS-01 and CPS-02, with measured detection probabilities in place of assumed ones.

## Exercises

1. Classify each as reproduction, application or contribution: (a) the IEEE 14-bus table of
   Lesson 5; (b) the HAI results; (c) the stealth sweep with attribution.
2. Rewrite this claim so that it survives review: "We present a novel Kalman-filter-based detector
   that defeats GNSS spoofing."
3. On a three-bus line network 1–2–3 with bus 1 as reference, flows on both branches and injections
   at all three buses, compute $\lVert He_{j} \rVert_{0}$ for both non-reference angles. Is moving the two angles together
   ever cheaper?
4. Name one experiment that would turn result 3 of Section 4 into something stronger, and say what
   data it would need.

### Solutions

1. (a) Reproduction (of the stealthy-attack property) plus an application of the meter count to a
   standard case. (b) Application. (c) The candidate contribution of Section 4, item 1.
2. "On a simplified loosely coupled GNSS/INS model, a Kalman-filter innovations monitor exposes
   receivers-only spoofing 97 % of the time at a 1 % false-alarm rate, in line with Khanafseh *et al.*
   (2014) and Tanıl *et al.* (2018); it cannot detect an injection that also rewrites the inertial
   data."
3. With branch reactances $x_{12}, x_{23}$, $H$ has rows: flow 1–2 sees $\theta_{2}$; flow 2–3 sees $\theta_{2}$ and $\theta_{3}$; injection 1
   sees $\theta_{2}$; injection 2 sees both; injection 3 sees both. So $\lVert He_{2} \rVert_{0} = 5$ and $\lVert He_{3} \rVert_{0} = 3$ (flow 2–3,
   injections 2 and 3). Moving both by the same amount ($c = (1, 1)$) leaves flow 2–3 and injections 2
   and 3 unchanged and touches only flow 1–2 and injections 1 and 2: three meters, cheaper than
   moving $\theta_{2}$ alone. This is exactly why the count of Lesson 5 is only an upper bound.
4. Run the classifier on real faults: a data set with labelled physical sensor faults and labelled
   attacks on the same system (for instance, faults injected on a testbed by its operators), so that
   the physics-versus-habit split can be measured outside simulation.

## References

See [`docs/related-work.md`](../related-work.md) for the full list with the checks made. The ones
used most in this lesson:

* Y. Liu, P. Ning and M. K. Reiter, *Proc. ACM CCS*, 2009.
* H. Sandberg, A. Teixeira and K. H. Johansson, *First Workshop on Secure Control Systems*, 2010.
* F. Pasqualetti, F. Dörfler and F. Bullo, *IEEE Transactions on Automatic Control* 58(11), 2013.
* Ç. Tanıl, S. Khanafseh, M. Joerger and B. Pervan, *IEEE Transactions on Aerospace and Electronic
  Systems*, 2018.
* A. Anwar, A. N. Mahmood and Z. Shah, *Proc. ACM CIKM*, 2015.
