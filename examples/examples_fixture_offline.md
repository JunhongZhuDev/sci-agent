# Example questions and outputs (policy=offline, corpus=fixture)

> The fixture papers are fictional (synthetic test corpus). Re-run without `--fixture` on the real arXiv corpus, with an API key for the LLM agent.

## Q: Are learned neural operators really faster than numerical solvers? Compare the claims.

**Tool trace:** `extract_numeric_claims` → `search_papers`  (2 steps, 0.01s)

Evidence from the corpus points in different directions:
- Most studies compare against weak baselines: when the numerical solver is run at a resolution that matches the learned model's accuracy, reported speedups shrink by one to three orders of magnitude, and several learned solvers are slower. [fx-crit#abs]
- We review published comparisons between machine learning PDE solvers and numerical methods for fluid problems. [fx-crit#abs]
- On two-dimensional turbulent flow at viscosity 1e-4 the operator is about 1000 times faster than the pseudo-spectral solver used to generate training data, while reaching a relative error of 2 percent. [fx-fno#abs]
- We propose a spectral neural operator that learns the solution operator of the Navier-Stokes equations. [fx-fno#abs]
- We compare physics-informed neural networks with the finite element method for the Poisson and Allen-Cahn equations. [fx-pinn#abs]
STATUS: CONFLICTING
CONFIDENCE: low

## Q: Do physics-informed neural networks beat the finite element method?

**Tool trace:** `extract_numeric_claims` → `search_papers`  (2 steps, 0.01s)

Relevant evidence from the corpus:
- The finite element method is faster and more accurate in solving time; physics-informed networks do not beat finite elements in our tests, but a trained network evaluates new points quickly. [fx-pinn#abs]
- We compare physics-informed neural networks with the finite element method for the Poisson and Allen-Cahn equations. [fx-pinn#abs]
- Physics-informed networks may be useful for inverse problems or very high dimension. [fx-pinn#t1]
- Physics-informed networks fail to train for convection problems with large convection coefficients. [fx-fail#abs]
- A message-passing graph neural network learns mesh-based simulation for cloth, aerodynamics and structural mechanics. [fx-gnn#abs]
STATUS: SUPPORTED
CONFIDENCE: medium

## Q: What does the figure of relative error versus runtime show for the spectral operator and solver?

**Tool trace:** `find_figures` → `inspect_figure` → `extract_numeric_claims` → `search_papers`  (4 steps, 0.01s)

Relevant evidence from the corpus:
- Figure 2: Relative error versus runtime for the spectral operator and the pseudo-spectral solver on 2D turbulence. [fx-fno#fig2]
- Figure 2: Relative error versus runtime for the spectral operator and the pseudo-spectral solver on 2D turbulence. [fx-fno#fig2]
- On two-dimensional turbulent flow at viscosity 1e-4 the operator is about 1000 times faster than the pseudo-spectral solver used to generate training data, while reaching a relative error of 2 percent. [fx-fno#abs]
- We propose a spectral neural operator that learns the solution operator of the Navier-Stokes equations. [fx-fno#abs]
- method | reported speedup | corrected speedup
operator A | 1000x | 7x
PINN B | 100x | 0.1x
GNN C | 50x | 4x [fx-crit#tab1]
STATUS: SUPPORTED
CONFIDENCE: medium

## Q: What drag coefficient is reported for a hypersonic re-entry capsule at Mach 25?

**Tool trace:** `search_papers`  (1 steps, 0.0s)

The indexed corpus does not contain sufficient evidence to answer this question (query-term coverage of retrieved evidence: 0.20). Closest related evidence: "Table 1: Reported versus baseline-corrected speedup of learned solvers." [fx-crit#tab1]
STATUS: INSUFFICIENT
CONFIDENCE: medium

## Q: Run our own experiment: does a POD-MLP surrogate beat a coarse classical solver for the 1D Burgers equation, in and out of distribution?

**Tool trace:** `run_surrogate_experiment` → `extract_numeric_claims` → `search_papers`  (3 steps, 17.55s)

Relevant evidence from the corpus:
- Most studies compare against weak baselines: when the numerical solver is run at a resolution that matches the learned model's accuracy, reported speedups shrink by one to three orders of magnitude, and several learned solvers are slower. [fx-crit#abs]
- The finite element method is faster and more accurate in solving time; physics-informed networks do not beat finite elements in our tests, but a trained network evaluates new points quickly. [fx-pinn#abs]
- For forward problems in low dimension classical solvers remain preferable. [fx-pinn#t1]
- We compare physics-informed neural networks with the finite element method for the Poisson and Allen-Cahn equations. [fx-pinn#abs]
- We propose a spectral neural operator that learns the solution operator of the Navier-Stokes equations. [fx-fno#abs]
Our own experiment: 1D viscous Burgers equation experiment (surrogate vs classical solver). POD rank r=21. Best surrogate 'POD + MLP': in-dist error 15.98%, OOD error 35.55%. Coarse 32-mode solver: in-dist 0.49%, OOD 5.93%, at 11.5 ms/query vs surrogate 0.017 ms. Surrogate errors grow sharply outside the training viscosity range, while the classical solver's do not depend on any training distribution; the speed-up must be weighed against offline data-generation cost and the accuracy of an equally cheap classical baseline. [experiment:burgers_n400]
STATUS: SUPPORTED
CONFIDENCE: medium
