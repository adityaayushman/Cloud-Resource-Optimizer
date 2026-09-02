# Corrected reference list

The report currently lists 17 references and cites **none of them in the body**,
while the four papers behind the methods actually used — Random Forest, XGBoost,
DQN and SHAP — are absent from the list entirely. That combination is the single
most likely thing to be challenged in a viva.

This file provides a list in which **every entry is cited at least once** and
**every method used has its source**. Numbering is IEEE style to match the
report's existing format.

---

## Reference list

**Foundations and definitions**

[1] P. Mell and T. Grance, "The NIST definition of cloud computing," National
Institute of Standards and Technology, Gaithersburg, MD, USA, Special
Publication 800-145, 2011.

[2] M. Armbrust, A. Fox, R. Griffith, A. D. Joseph, R. Katz, A. Konwinski,
G. Lee, D. Patterson, A. Rabkin, I. Stoica, and M. Zaharia, "A view of cloud
computing," *Communications of the ACM*, vol. 53, no. 4, pp. 50–58, 2010.

**Machine learning methods used**

[3] L. Breiman, "Random forests," *Machine Learning*, vol. 45, no. 1, pp. 5–32,
2001.

[4] T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," in
*Proc. 22nd ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining*, 2016,
pp. 785–794.

[5] S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model
predictions," in *Advances in Neural Information Processing Systems 30*, 2017,
pp. 4765–4774.

[6] S. M. Lundberg, G. G. Erion, and S.-I. Lee, "Consistent individualized
feature attribution for tree ensembles," *arXiv:1802.03888*, 2018.

[7] F. Pedregosa et al., "Scikit-learn: Machine learning in Python," *Journal of
Machine Learning Research*, vol. 12, pp. 2825–2830, 2011.

[8] F. T. Liu, K. M. Ting, and Z.-H. Zhou, "Isolation forest," in *Proc. 8th
IEEE Int. Conf. Data Mining (ICDM)*, 2008, pp. 413–422.

**Reinforcement learning**

[9] V. Mnih, K. Kavukcuoglu, D. Silver, A. A. Rusu, J. Veness, M. G. Bellemare,
A. Graves, M. Riedmiller, A. K. Fidjeland, G. Ostrovski, S. Petersen,
C. Beattie, A. Sadik, I. Antonoglou, H. King, D. Kumaran, D. Wierstra, S. Legg,
and D. Hassabis, "Human-level control through deep reinforcement learning,"
*Nature*, vol. 518, no. 7540, pp. 529–533, 2015.

[10] R. S. Sutton and A. G. Barto, *Reinforcement Learning: An Introduction*,
2nd ed. Cambridge, MA, USA: MIT Press, 2018.

**Evaluation methodology**

[11] S. Arlot and A. Celisse, "A survey of cross-validation procedures for model
selection," *Statistics Surveys*, vol. 4, pp. 40–79, 2010.

**Cloud resource allocation**

[12] A. Beloglazov and R. Buyya, "Optimal online deterministic algorithms and
adaptive heuristics for energy and performance efficient dynamic consolidation
of virtual machines in cloud data centers," *Future Generation Computer
Systems*, vol. 28, no. 5, pp. 755–768, 2012.

[13] Q. Zhang, M. F. Zhani, R. Boutaba, and J. L. Hellerstein, "Dynamic
heterogeneity-aware resource provisioning in the cloud," *IEEE Transactions on
Cloud Computing*, vol. 2, no. 1, pp. 14–28, 2014.

[14] R. N. Calheiros, R. Ranjan, A. Beloglazov, C. A. F. De Rose, and R. Buyya,
"CloudSim: A toolkit for modeling and simulation of cloud computing
environments and evaluation of resource provisioning algorithms," *Software:
Practice and Experience*, vol. 41, no. 1, pp. 23–50, 2011.

[15] M. Mao and M. Humphrey, "Auto-scaling to minimize cost and meet application
deadlines in cloud workflows," in *Proc. Int. Conf. High Performance Computing,
Networking, Storage and Analysis (SC)*, 2011, pp. 1–12.

[16] Z. Xiao, W. Song, and Q. Chen, "Dynamic resource allocation using virtual
machines for cloud computing environment," *IEEE Transactions on Parallel and
Distributed Systems*, vol. 24, no. 6, pp. 1107–1117, 2013.

[17] A. Verma, L. Pedrosa, M. Korupolu, D. Oppenheimer, E. Tune, and J. Wilkes,
"Large-scale cluster management at Google with Borg," in *Proc. 10th European
Conf. Computer Systems (EuroSys)*, 2015, pp. 1–17.

**Energy and sustainability**

[18] Y. C. Lee and A. Y. Zomaya, "Energy efficient utilization of resources in
cloud computing systems," *Journal of Supercomputing*, vol. 60, no. 2,
pp. 268–280, 2012.

[19] A. Shehabi, S. J. Smith, D. A. Sartor, R. E. Brown, M. Herrlin,
J. G. Koomey, E. R. Masanet, N. Horner, I. L. Azevedo, and W. Lintner, "United
States data center energy usage report," Lawrence Berkeley National Laboratory,
Berkeley, CA, USA, Rep. LBNL-1005775, 2016.

---

## Where each reference is cited

Every entry appears at least once. Add the markers in these places.

| Ref | Cite in | Purpose |
|---|---|---|
| [1] | §1.1, §2.1 | Definition of cloud computing |
| [2] | §1.1, §2.1, §4.7 | Elasticity and its management cost |
| [3] | §2.2, §3.1.3, §4.2 | Random Forest |
| [4] | §2.2, §3.1.3, §4.2 | XGBoost — the primary predictor |
| [5] | §2.2, §3.2.2, §4.4 | SHAP — the explainability method |
| [6] | §3.2.2 | TreeSHAP — exact attribution for tree ensembles |
| [7] | §3.1.3, Table 3.1 | scikit-learn implementations |
| [8] | §3.2.2, §4.4 | Isolation Forest — anomaly detection |
| [9] | §2.2, §3.2.2, §4.5, §4.6.1 | DQN — the RL algorithm |
| [10] | §2.2, §3.2.2, §4.5.1 | RL foundations, reward design |
| [11] | §3.1.2, §4.1 | Why tuning on the test set biases results |
| [12] | §2.2, §4.6.1 | Energy-aware VM consolidation |
| [13] | §2.2, §4.6.1 | Heterogeneity-aware provisioning |
| [14] | §3.1.3, §4.7 | Simulation-based evaluation is established practice |
| [15] | §2.2, §4.6.1 | Auto-scaling for cost and deadlines |
| [16] | §2.2 | Dynamic VM-based allocation |
| [17] | §4.7 | Bin-packing and fragmentation in real schedulers |
| [18] | §1.4, §4.6.1 | Energy-efficient resource use |
| [19] | §1.4, §4.6.1 | Data-centre energy consumption — SDG 13 |

---

## Removed, and why

These appear in the current list but are not cited anywhere and are not needed:

- Dean & Ghemawat, *MapReduce* — batch processing, unrelated to this work.
- Li et al., *traffic-aware VM placement* — network placement, not covered.
- Khazaei et al., *M/G/m/m+r queuing* — no queueing model is used here.
- Hwang, Fox & Dongarra, *Distributed and Cloud Computing* — general textbook,
  superseded by [1] and [2] for the specific claims made.
- Wood et al., *black-box VM migration* — no migration mechanism is implemented.
- Sonnek et al., *Starling* — affinity-aware migration, not implemented.
- Singh & Chana, *Q-aware provisioning* — QoS provisioning is discussed but the
  method is not used; drop or cite explicitly in §2.2.
- Verma, Ahuja & Neogi, *power-aware HPC placement* — superseded by [12] and
  [18] for the energy argument.

---

## A note on Table 2.1

Its current caption reads *"Summary of Related Work in **RUL Prediction**"* —
Remaining Useful Life, a predictive-maintenance topic with no connection to this
project. It is leftover text from another document and must be retitled, for
example:

> **Table 2.1 — Summary of related work in cloud resource optimisation**

The same contamination appears in Appendix A, which contains a
`load_and_process_data()` function that parses turbofan sensor columns and
computes Remaining Useful Life. Remove it when regenerating the appendix from
the working source.
