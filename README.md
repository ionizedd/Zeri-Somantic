# Zeri-Somantic

A lightweight, experimental SNN.

This repo holds a rough, early slice of the Zeri Somatic core: a seed-deterministic
spiking neural network trained with actor-critic reinforcement learning (TD error,
eligibility traces, reward-modulated STDP). Give it a state vector and a reward,
it gives you an action, and it gets better with practice — no external ML
framework, no GPU, no network calls. Two files, zero dependencies, compiles in
under a second.

What's here:

- `core/zeri_brain.h` — the underlying single-header SNN (STDP learning,
  seed-derived weights, compact delta memory).
- `core/zeri_agent.h` — the actor-critic RL layer on top: epsilon-greedy /
  stochastic action selection, eligibility traces, genetic breeding between
  trained agents, surprise-gated synaptic forgetting.
- `demo/grid_demo.c` — a from-scratch agent learning to navigate a 7x7 grid
  maze purely from reward, no hand-coded pathing.

Build the demo:

```
gcc -O3 -mavx2 -mfma demo/grid_demo.c -lm -o grid_demo
./grid_demo
```

Or on MSVC: `cl /O2 /arch:AVX2 demo\grid_demo.c`

This is the rough/free slice while the fuller kit keeps getting worked on —
Unity and Roblox wrappers, an experimental tiny-footprint variant
(NovaSomatic), a browser-playable demo, and more built-in mazes — all
available at [Zeri Smol Kit on itch.io](https://parinov.itch.io/zeri-somatic).

MIT licensed. Use it in free or commercial projects, no attribution required
(though it's always appreciated).

### How this was built

Same story as the itch.io kit: a human made the calls on what this should do,
an AI coding assistant (Claude) wrote most of the implementation and did the
bulk of the debugging — including the reinforcement-learning tuning that got
the maze demo actually converging. The code is commented in-line with what
broke and why, on purpose, so it's followable by a person or by another AI
picking this up later.
