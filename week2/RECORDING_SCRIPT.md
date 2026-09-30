# Week 2 verification recording script

Target length: 90–110 seconds. The student should record this with their own voice while showing the terminal, figures, and public viewer.

1. Show the repository and say: “This week I built a two-dimensional Lennard-Jones molecular-dynamics program in Rust. The same tested force law grows from a two-particle example into a periodic fluid.”
2. Show `cargo test --release` passing. Say: “The tests verify the analytic force against a numerical derivative, Newton's third law, shifted-cutoff continuity, the exact lattice and output contracts, and equality of naive and cell-list forces.”
3. Show `md check week2/artifacts` ending in `PASS`. Read the four values: drift `4.03e-5`, speed temperature `0.525`, `chi2/22 = 1.033`, and stored-energy error `1.91e-15`. Say: “This independently rebuilds every saved frame, so it checks the output rather than trusting the simulator.”
4. Show `scaling.png`. Say: “Release mode is 0.288 seconds versus 4.333 seconds in debug. Cell lists have overhead at 100 atoms, but reach a 3.33-times speedup at 1600 atoms because they avoid the quadratic all-pairs search.”
5. Show `cold.mp4`, `hot.mp4`, and `melting.png`. Say: “At low temperature, several sharp radial-distribution peaks persist. At high temperature the distant peaks flatten toward one. During the 400-atom ramp, long-range RDF contrast falls from 0.348 to 0.097, which is quantitative evidence of loss of order.”
6. End on the public GitHub Pages viewer and show that the 400 atoms, 200 frames, temperature rise, and RDF update without signing in.

Upload the finished video to a GitHub Release, keep it at or below two minutes, and place its URL beside the Pages URL in `README.md`.
