# Chess_AI

A project focused on developing a Chess AI.

## Methodology

This project aims to build a state-of-the-art (SOTA) chess engine by leveraging a combination of advanced techniques:

*   **Deep Reinforcement Learning (Deep RL)**: A neural network learns to evaluate board positions and predict promising moves.
*   **Monte Carlo Tree Search (MCTS)**: Explores the game tree, guided by the neural network's predictions, to select the best move.
*   **Self-Play**: The agent learns and improves by playing games against itself, generating the data needed for training the neural network.

This approach, inspired by systems like AlphaZero and Leela Chess Zero, combines the pattern recognition strengths of deep learning with the robust search capabilities of MCTS, enabling the agent to achieve high-level performance through automated learning.

## Crucial Libraries

*   **[`python-chess`](https://github.com/niklasf/python-chess)**: A pure Python chess library used for representing boards, generating legal moves, parsing move notation (like UCI), and evaluating board states.
*   **[`pygame`](https://www.pygame.org/)**: A set of Python modules designed for writing video games. Used here for visualizing the chessboard and interactions.
*   **[`gymnasium`](https://gymnasium.farama.org/)** (formerly OpenAI Gym): A standard API for reinforcement learning environments. Provides the framework for the AI agent to interact with the chess environment.

## Environment

This project utilizes and expands upon the [`chess-gym`](https://github.com/ryanrudes/chess-gym) library to create a custom Gymnasium environment suitable for training the Chess AI agent.
