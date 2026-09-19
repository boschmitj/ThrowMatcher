# Handball Motion Manifold Learning and Motion Synthesis

Bachelor thesis repository for combining low-frequency professional handball
tracking data with high-frequency motion-capture recordings of seven-metre
throws.

The main reproducible workflow is the League-to-Mocap throw matching pipeline
in [`PenaltyProcessing`](PenaltyProcessing/README.md). It detects the point of
release, builds a common trajectory representation, retrieves similar League
throws, evaluates a learned ranker, and reconstructs the selected continuation
at 300 Hz.

Other directories contain the markerless motion-capture pipeline, Blender and
MotionBuilder assets, and thesis material. The markerless pipeline retains its
upstream component documentation.

The raw League position files and Mocap recordings are not distributed in this
repository. See the Penalty Processing README for the required directory
layout and complete commands.
