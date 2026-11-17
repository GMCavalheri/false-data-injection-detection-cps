# Related work, and what is (and is not) new here

Every claim below was checked against the cited source (its abstract, its publisher's or
authors' page, or the paper itself) on 2026-11-17. Where only an abstract was available, the
claim is limited to what the abstract says.

## 1. Stealthy false data injection in static state estimation

* **Liu, Ning and Reiter (2009)** showed that an attacker who knows the measurement matrix of a
  power-grid state estimator can inject $a = Hc$ and pass the residual-based bad-data test while
  shifting the estimate by $c$. Lessons 1 and 4 use their construction; Lesson 5 reproduces the
  property on the IEEE 14-bus system, the kind of test case they used.
* **Sandberg, Teixeira and Johansson (2010)** defined *security indices*: the least effort, as the
  fewest measurements or the smallest corruption, an attacker needs to reach a goal without
  triggering bad-data alarms. The "meters to alter" table of Lesson 5 counts the meters an attack
  needs when it moves *one* angle and nothing else ($a = He_{j}$). That is an upper bound on their
  index for that state, whose minimum is taken over every attack that moves it; it is not their
  index.
* The chi-square bad-data test itself is classical state-estimation practice (Monticelli, 1999).

## 2. Stealth against dynamic estimators

* **Mo and Sinopoli (2010)** studied false data injection against a system monitored by a Kalman
  filter with a failure detector and controlled by LQG.
* **Pasqualetti, Dörfler and Bullo (2013)** showed that static monitors cannot detect attacks on
  the dynamics, that measurement attacks are easy to make undetectable by them, and that an attack
  on a linear system is undetectable if and only if it excites only the system's zero dynamics.
* **Teixeira, Shames, Sandberg and Johansson (2015)** described attacks by the adversary's model
  knowledge, disclosure and disruption resources, and analysed zero-dynamics and bias-injection
  attacks among others.

Lesson 5's statement, that an injection which is a trajectory of the filter's own model leaves
every innovation unchanged, is a simple special case of this body of work, proved there for
completeness. It is **not new**.

## 3. GNSS spoofing detection with inertial sensors

* **Khanafseh *et al.* (2014)** built a RAIM-style residual monitor on an integrated GPS/INS
  system to detect spoofing, using the INS as the source of redundancy.
* **Tanıl, Khanafseh, Joerger and Pervan (2018)** extended this to a Kalman-filter innovations
  monitor evaluated against worst-case spoofers, including spoofers that can track the vehicle's
  position.

The finding of Lessons 4 and 5, that receivers-only spoofing is exposed by the INS through the
filter's innovations while a bus-level injection that also rewrites the INS is not, agrees with
this line of work on a much simpler model. It is a **reproduction of a known conclusion**, not a
new one; the value here is that it is measured with the same pipeline as everything else, with
an exact false-alarm rate, and fed back into an attack tree (Section 6).

## 4. Telling faults from attacks

* **Anwar, Mahmood and Shah (2015)** framed the problem for smart grids: detection work existed
  but little distinguished cyber-attacks from physical faults, and misclassification leads to the
  wrong mitigation. They generated faults and attacks on the IEEE 30-bus system, projected the
  data with PCA and compared supervised classifiers, reporting that most of them performed poorly
  on the harder data.
* **Roy and Dey (2023, preprint)** studied when anomalies can be distinguished as physical faults
  or actuation attacks.

This project's Lesson 3 classifier belongs to this family. Its high accuracy (Lesson 3) comes
with the caveats Lesson 4 measures: it rests on the fault model and on assumed attacker habits,
and an exact fault mimic cannot be attributed by any classifier (a Bayes argument, not an
empirical finding).

## 5. Testbed data

* **SWaT** (Goh, Adepu, Junejo and Mathur, 2016), a six-stage water-treatment testbed, distributed
  on request.
* **HAI** (Shin, Lee, Yun and Kim, 2020), a boiler, turbine and water-treatment testbed with a
  hardware-in-the-loop simulator, public on GitHub. Its authors recommend the **eTaPR** measure
  (Hwang, Yun, Kim and Min, 2022) for time-series anomaly detection; Lesson 6 reports simpler
  per-second and per-event measures and says so.
* The data-driven static check of Lesson 6 is the principal-component residual (Q statistic) of
  **Jackson and Mudholkar (1979)**.

## 6. What is new here, stated narrowly

Searches for the combination below did not find it as one published piece. That is not proof of
absence; a paper would need a systematic search (Section 7) before claiming any of it.

1. **One reproducible pipeline that measures fault-versus-FDI attribution along a continuous
   stealth dial**, from a single-channel offset to a fully consistent injection, with the
   intended effect held fixed, and anchors every detection rate it can to a closed form (the
   $(1-s)^{2}$ noncentrality law, exact scenario false-alarm rates) checked against Monte Carlo.
2. **Detection measured in the units of an attack tree.** CPS-02 had to *assume* that the
   navigation integrity check misses a spoof with probability 0.1. Here that number is measured
   per attacker reach: about 0 for crude attacks, 0.03 to 0.51 for receivers-only spoofing
   depending on the monitor, about 1 for a bus-level consistent injection. That links a detection
   study to the reliability and attack-tree models of CPS-01 and CPS-02 on the same synthetic
   system.
3. **An explicit separation of physics cues from attacker-habit cues**: the mimicry experiment
   shows which part of a strong classifier's accuracy comes from the fault model and the assumed
   magnitudes, rather than from physics.
4. **A course**: seven lessons that derive every test from first principles and point to the code
   that computes it.

Nothing here is a new detection algorithm or a new theorem.

## 7. Searches still needed before any paper

* Systematic search (Scopus, IEEE Xplore, Google Scholar) for "fault attack discrimination" with
  "stealth" or "sophistication" sweeps, and for detection results reported as attack-tree or
  fault-tree probabilities.
* The aviation and ICAO/RTCA literature on GNSS integrity monitoring under spoofing (RAIM,
  ARAIM, INS coupling) beyond the two papers above.
* Process-physics and physics-informed (PINN) attack detection, for the follow-on project OT-02.

## References

* Y. Liu, P. Ning and M. K. Reiter, "False data injection attacks against state estimation in
  electric power grids," *Proc. 16th ACM Conference on Computer and Communications Security
  (CCS)*, 2009.
* H. Sandberg, A. Teixeira and K. H. Johansson, "On security indices for state estimators in
  power networks," *First Workshop on Secure Control Systems (SCS)*, CPS Week, 2010.
* A. Monticelli, *State Estimation in Electric Power Systems: A Generalized Approach*, Kluwer,
  1999.
* Y. Mo and B. Sinopoli, "False data injection attacks in control systems," *First Workshop on
  Secure Control Systems (SCS)*, CPS Week, 2010.
* F. Pasqualetti, F. Dörfler and F. Bullo, "Attack detection and identification in
  cyber-physical systems," *IEEE Transactions on Automatic Control* 58(11), 2715–2729, 2013.
* A. Teixeira, I. Shames, H. Sandberg and K. H. Johansson, "A secure control framework for
  resource-limited adversaries," *Automatica* 51, 2015.
* S. Khanafseh, N. Roshan, S. Langel, F.-C. Chan, M. Joerger and B. Pervan, "GPS spoofing
  detection using RAIM with INS coupling," *Proc. IEEE/ION PLANS*, 2014.
* Ç. Tanıl, S. Khanafseh, M. Joerger and B. Pervan, "An INS monitor to detect GNSS spoofers
  capable of tracking vehicle position," *IEEE Transactions on Aerospace and Electronic Systems*,
  2018.
* A. Anwar, A. N. Mahmood and Z. Shah, "A data-driven approach to distinguish cyber-attacks from
  physical faults in a smart grid," *Proc. 24th ACM CIKM*, 2015.
* T. Roy and S. Dey, "On distinguishability of anomalies as physical faults or actuation
  cyberattacks," arXiv:2311.09342, 2023.
* J. Goh, S. Adepu, K. N. Junejo and A. Mathur, "A dataset to support research in the design of
  secure water treatment systems," *CRITIS 2016*, LNCS 10242, 2017.
* H.-K. Shin, W. Lee, J.-H. Yun and H. Kim, "HAI 1.0: HIL-based augmented ICS security dataset,"
  *13th USENIX Workshop on Cyber Security Experimentation and Test (CSET)*, 2020.
* W.-S. Hwang, J.-H. Yun, J. Kim and B. G. Min, "Do you know existing accuracy metrics overrate
  time-series anomaly detections?," *Proc. 37th ACM/SIGAPP Symposium on Applied Computing (SAC)*,
  2022.
* J. E. Jackson and G. S. Mudholkar, "Control procedures for residuals associated with principal
  component analysis," *Technometrics* 21(3), 1979.
