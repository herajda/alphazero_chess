# AlphaZero algorithm on Chess game

This is a personal attempt at implementing and training a chess agent that uses [AlphaZero](https://arxiv.org/abs/1712.01815)

## chess-library
This project uses [chess-library](https://github.com/Disservin/chess-library) as a git submodule located in `lib/chess-library/`.
After cloning this repository, run:

```bash
git submodule update --init
```

to download the library files.

## Setup
1. Install the Python requirements:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the helper script to compile the C++ components and fetch additional resources:
   ```bash
   scripts/setup.sh
   ```

3. Start training (requires a CUDA-capable GPU):
   ```bash
   python src/python/train_cpp.py
   ```

Stockfish is needed for evaluating the agent. Ensure a Stockfish binary is installed and available in your `PATH`.


