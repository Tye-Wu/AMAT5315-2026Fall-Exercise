# Week 5 execution plan

This plan treats the supplied Week 5 learning sheet as the assignment contract. The work is a two-part program: JAX establishes local forward/reverse automatic differentiation, then a Rust acoustic simulator uses Enzyme-generated local JVP/VJPs and a hand-composed time loop to produce Born data, a reverse-time image, and checkpointed replays.

## Scope and acceptance

1. **JAX AD (`artifacts/ad/`)** — differentiate the Lennard-Jones pair energy at `r=1.3` in 64-bit; save per-node forward tangents and reverse adjoints; compare both with `jax.grad` and centered finite differences on 601 samples; render `modes.png`, `graph.png`, and `grad-graph.png`; time full cluster gradients for `N=64,128,256,512,1024`. Require AD errors `<1e-12`, the requested scalar values within `1e-12`, and at 3072 inputs a forward/reverse timing ratio above 100.
2. **Acoustic forward (`artifacts/forward/`)** — implement the stated centered update, five-point Laplacian, zero outer boundary, sponge, Gaussian-footprint Ricker source, and post-update receiver sampling. Save traces/recordings in the documented NumPy format and reproduce the trace norm and per-shot peak amplitudes to relative error `<1e-4`. Plot the input geometry/pulse and gathers, and verify the step-150 echo is about 2% of the direct field.
3. **Enzyme Born/adjoint (`artifacts/born`, `artifacts/adjoint`)** — let Enzyme differentiate the actual timestep kernel, chain one JVP or VJP per step, use fixed source/damping/initial state, and sum velocity sensitivity over all shots. Require the transpose identity relative difference `<1e-9` and row-norm image peak within one cell of row 21. Full history is used only for the small reflector experiment.
4. **Treeverse (`artifacts/checkpoint-*`)** — implement complete-state saves/restores/replay under additional slot budgets 1, 3, 5, 10. Audit ordered gradients, valid restores and budget peaks; compare each checkpointed image with full history at relative error `<1e-9`. Plot schedule and work/storage curves. Never run Marmousi with full history.
5. **Marmousi (`artifacts/marmousi-*`)** — use six total saved states (initial plus five extra) on the licensed 805×269 model, 1200 steps and nine shots; render background, perturbation, x=10 km Born gather, and raw image with a shared image scale. Require image L2 norm `6.7037741e-4` within relative error `1e-4`, peak saved bytes `20,788,320`, and visible dipping structure in the designated upper-depth region.
6. **Release evidence** — list each committed evidence file and its generating command in `README.md`; commit source, figures/plots/JSON and `MARMOUSI-LICENSE`, but ignore `inputs/` and all `.npy` data. Push the exercise repository. No course submission is required this week.

## Execution platform and boundaries

JAX CPU is supported on this Apple ARM machine. The course Enzyme setup is validated only on x86_64 Linux; no local Linux container runtime is installed. Use a temporary Linux sandbox for Enzyme compilation and execution, transferring only the public assignment source and licensed course inputs, and destroy the sandbox at the end. If the selected image/toolchain cannot produce real Enzyme-generated derivatives, stop and document that specific blocker rather than substituting handwritten derivatives.

The final page's Alibaba Cloud sign-up and payment-method verification are personal account actions. The student must complete those in their own account after discussing expected Week 6 cost with the instructor; no GPU rental is part of Week 5.

## Reproducibility order

Run each Part's scripts from `week5/`. Inputs are downloaded from the course archive into the ignored `inputs/` directory. Generated evidence goes under `artifacts/`; NumPy `.npy` arrays are excluded from Git. Record exact toolchain versions, commands, norms, errors, profile/work counters, and artifact paths in the per-Part result JSON and `README.md`.
