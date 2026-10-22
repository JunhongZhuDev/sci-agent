# Example questions and outputs (policy=llm, corpus=arxiv)

## Q: Do machine-learning solvers for fluid PDEs really deliver the large speedups over numerical solvers that are often reported?

**Tool trace:** `search_papers`  (2 steps, 10.24s)

Several papers suggest that machine learning (ML) solvers for fluid PDEs can deliver significant speedups. For instance, [2102.01010#t27] estimates that speed-ups in the range of 10³–10⁴ may be possible for 3D simulations, indicating substantial reductions in computational cost. However, the exact magnitude of speedups can vary depending on the problem and the specific ML approach used. [2110.02085#t4] highlights that while ML can enhance CFD, it is important to ensure that uncertainties are identified and quantified. The potential for large speedups is also noted in [2110.02085#t31], which discusses the energy consumption challenges in large-scale simulations and suggests that ML can be an enabler for more efficient operation of HPC facilities.

STATUS: SUPPORTED
CONFIDENCE: medium

## Q: Can physics-informed neural networks replace the finite element method for forward problems?

**Tool trace:** `search_papers`  (2 steps, 11.56s)

The paper "Can Physics-Informed Neural Networks beat the Finite Element Method?" [2302.04107#abs] suggests that, in their current form, physics-informed neural networks (PINNs) have not outperformed the finite element method (FEM) in terms of solution time and accuracy. However, the authors note that PINNs can be faster in evaluating the solved PDE in some experiments. Another paper, "FE-PINNs: finite-element-based physics-informed neural networks for surrogate modeling" [2412.07126#abs], proposes a method that integrates FEM with PINNs, leveraging the inverse isoparametric map of FEM to train neural networks for surrogate modeling. This approach shows reasonable accuracy in testing scenarios with linear boundary-value problems.

The evidence indicates that while PINNs have shown promise in certain applications, they have not yet replaced FEM for forward problems. The integration of FEM with PINNs might offer a more robust solution, but further research is needed to fully evaluate their performance and applicability.

STATUS: PARTIAL
CONFIDENCE: medium

## Q: Why do physics-informed neural networks fail on convection problems, and what fixes are proposed?

**Tool trace:** `search_papers`  (2 steps, 9.77s)

Physics-informed neural networks (PINNs) can fail on convection problems due to difficulties in capturing sharp transitions and predicting solutions past a certain timestep [2109.01050#fig1]. Specifically, for ν = 2, PINNs achieve a relative error of 50%, struggling to capture the "sharper" transitions while predicting the center of the solution better [2109.01050#t21].

To address these issues, ProPINN proposes a new method to identify and mitigate propagation failures in PINNs [2502.00803#fig1]. ProPINN introduces the gradient correlation of corresponding models between nearby points, which helps in pinpointing areas where the solution propagation is difficult. This method can improve the performance of PINNs on convection problems by better understanding and handling the propagation of solutions.

STATUS: SUPPORTED
CONFIDENCE: medium

## Q: In the Fourier Neural Operator paper, what does the figure comparing models on the Navier-Stokes equation show?

**Tool trace:** `find_figures`  (2 steps, 7.45s)

The figure comparing models on the Navier-Stokes equation is Figure 3 from the paper "Fourier Neural Operator for Parametric Partial Differential Equations" [2010.08895#fig3]. This figure benchmarks the performance of different models including a state-of-the-art neural network architecture (FCN), an operator method using PCA (PCANN), the original graph neural operator (GNO), and the multipole graph neural operator (MGNO) on the Navier-Stokes equation. The results likely show the accuracy or error of these models in solving the Navier-Stokes equation, highlighting the strengths and weaknesses of each approach.

STATUS: SUPPORTED
CONFIDENCE: medium

## Q: What drag coefficient do the papers report for a hypersonic re-entry capsule at Mach 25?

**Tool trace:** `search_papers`  (2 steps, 10.24s)

The papers do not directly report a drag coefficient for a hypersonic re-entry capsule at Mach 25. However, they discuss related metrics and experiments:

- [2307.14045#t42] discusses the calculation of relative errors for various quantities, including the density and local Mach number, but does not provide specific drag coefficients.
- [2309.00583#fig3] shows a cost-accuracy trade-off analysis for the drag coefficient, indicating that the Geometry-Informed Neural Operator (GINO) model is significantly faster than the OpenFOAM solver, but the exact drag coefficient value is not reported.
- [2212.07564#tab3] and [2212.07564#t25] focus on relative errors and mean squared errors for drag and lift coefficients, but again, specific values are not provided for a Mach 25 re-entry capsule.

STATUS: INSUFFICIENT
CONFIDENCE: low

## Q: Run our own experiment: does a POD-MLP surrogate beat a coarse classical solver for the 1D Burgers equation, in and out of distribution? Relate the result to the literature.

**Tool trace:** `run_surrogate_experiment`  (2 steps, 21.72s)

Our experiment on the 1D viscous Burgers equation with a coarse classical solver (10 modes) and a POD-MLP surrogate (POD rank 50) shows that the surrogate performs poorly out-of-distribution (OOD) compared to the classical solver. Specifically, the surrogate's relative L2 error is 97.172% ± 7.444% in-distribution and 100.798% OOD, while the classical solver's error is 31.753% ± 12.137% in-distribution and 27.777% OOD. The surrogate is also slower at 0.057 ms/query compared to the classical solver's 11.049 ms/query. The classical solver's performance is consistent across in- and out-of-distribution scenarios, while the surrogate's accuracy degrades sharply outside the training viscosity range.

This result aligns with the literature, which often cautions that surrogates can overfit to the training data and perform poorly when extrapolating. The classical solver, on the other hand, is a cheap and robust baseline that generalizes well, making it a reliable choice for scenarios where the training data is limited or the problem is complex.

STATUS: SUPPORTED
CONFIDENCE: high
