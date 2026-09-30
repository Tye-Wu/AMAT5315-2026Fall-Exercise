# Week 2 Parts 3–5 design

This document records the implemented contract. The supplied Week 2 learning sheet is the authoritative assignment specification.

## Scientific model

The model is a two-dimensional Lennard-Jones fluid in reduced units (`sigma = epsilon = mass = kB = 1`). Pair energy is `4(r^-12-r^-6)` and force is its negative radial derivative. Fluid runs use a rectangular periodic box, minimum-image displacements, a cutoff `rc=2.5`, and an energy shift `U(r)-U(rc)` inside the cutoff. The shift makes energy continuous; it does not make the force continuous.

Velocity-Verlet is used because its time symmetry and symplectic structure keep bounded energy errors in an isolated trajectory. The implementation caches the latest acceleration, so after the first step only one new force evaluation is needed per step. Euler remains only as the deliberately poor comparison in the dimer exercise.

## Exact run and data contract

The default run is `N=100`, `rho=0.8`, `T=0.5`, `dt=0.01`, `eq_steps=2000`, `steps=10000`, `sample_every=50`, `seed=2026`, `cutoff=2.5`, and cell-list forces. It therefore saves exactly 200 production frames. Initial velocities are Gaussian with variance `T`; centre-of-mass velocity is removed and the equilibration thermostat uses the remaining `2N-2` degrees of freedom.

`run.json` stores the configuration and rectangular box. Each `traj.jsonl` record stores `step`, `t`, `pos`, `vel`, `E_pot`, and `E_kin`. Production is NVE unless `--ramp-to` is supplied. A ramp linearly changes the target and rescales every 50 production steps, including the exact final target.

## Independent acceptance check

`md check` does not reuse stored energies. It reconstructs a naive-force system from each frame and recomputes shifted potential and kinetic energy. It verifies wrapped coordinates, frame count and times, the first/last 10%-window energy drift, `T_speed=<v^2>/2`, a 24 equal-probability-bin Maxwell speed-shape test, and stored energy consistency. Pass bounds are drift `<2e-3`, temperature error `<0.05`, `chi2/22<2`, and maximum stored-energy error `<1e-8`.

## Performance design

The reference implementation offers identical `naive` and `cells` force paths. The cell grid uses `floor(L/rc)` cells per direction, making every cell at least as wide as the cutoff. Each particle checks its own and eight wrapped neighbours; duplicate wrapped cell IDs and atom pairs are removed. Equality tests include boundary-crossing pairs, pairs just inside/outside the cutoff, and a two-cell-wide box.

Timing uses three runs and reports the median plus range. Profiling first establishes that force evaluation is the dominant region. Scaling at `N=100,400,1600` verifies the expected crossover: cell setup can lose for small systems, but its advantage increases at fixed density as the all-pairs path grows quadratically.

## Structural interpretation

Movies combine particle positions with a rolling radial distribution function. Cold and hot constant-temperature runs provide the controlled comparison. The public 400-atom run ramps from `T=0.2` to `1.2`; loss of distant RDF peaks and a decrease in `RMS[g(r)-1]` for `r>2` provide quantitative evidence of loss of long-range order. This is evidence for the simulated model, not a claim about a specific experimental material.
