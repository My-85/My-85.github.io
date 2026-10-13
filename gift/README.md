# GIFT — Flight workspace

Geometry-guided Imitation for Flight Tracking.

Public replay and trajectory editor: https://my-85.github.io/gift/

This folder contains the static website, vendored browser dependencies and
archived flight/reference data. Flight traces preserve their recorded positions
and measured attitudes; they are not newly generated flight simulations.

Real custom flights use the existing owner-authorized GPU service. Start Flight
opens its login/connection window; requests and results pass between the two
verified origins, with no service credential embedded in the public source.
Keep the connection window open while a flight is running.

The model remains the retained A4 baseline 9a. Display meshes are illustrative.
The method report contains research results as of 2026-10-12.

GitHub Pages publishes the repository main branch; .nojekyll is at the repository
root. No build step or server runtime is needed for these static files.
