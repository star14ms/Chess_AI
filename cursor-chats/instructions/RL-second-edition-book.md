## AlphaGo Zero: Learning from Scratch

Building upon the experience with *AlphaGo*, a DeepMind team developed ***AlphaGo Zero*** (Silver et al. 2017a).

**Key Differences from AlphaGo:**

*   **No Human Data:** In contrast to AlphaGo, this program used **no human data or guidance** beyond the basic rules of the game (hence the *Zero* in its name). It learned exclusively from *self-play reinforcement learning*.
*   **Raw Input:** Input consisted of "raw" descriptions of the placements of stones on the Go board.
*   **Learning Method:** Implemented a form of *policy iteration*, interleaving policy evaluation with policy improvement.
*   **MCTS During Learning:** Used *MCTS* to select moves throughout self-play reinforcement learning (unlike AlphaGo, which used MCTS only for live play *after* learning).
*   **Single Network:** Used only *one* deep convolutional ANN.
*   **Simpler MCTS:** Used a simpler version of MCTS.

### AlphaGo Zero's MCTS

*   **No Rollouts:** Did *not* include rollouts of complete games and therefore did not need a rollout policy.
*   **Leaf Node Termination:** Each MCTS simulation ended at a *leaf node* of the current search tree, not at a terminal game position.
*   **Network Guided:** Each MCTS iteration was guided by the output of the deep convolutional network (`f✓`).
*   **Network Input/Output:**
    *   **Input:** Raw board positions.
    *   **Output (Two Parts):**
        1.  **Value (`v`):** A scalar estimate of the probability that the current player will win.
        2.  **Policy (`p`):** A vector of move probabilities for all possible moves (stone placements + pass/resign).

### MCTS as Policy Improvement

Instead of selecting self-play actions directly according to the network's policy `p`, AlphaGo Zero used these probabilities (`p`) and the value (`v`) to direct the MCTS execution. MCTS then returned *new* move probabilities (`π`).

*   These MCTS-derived policies (`π`) benefitted from the numerous simulations run during the search.
*   The result was that the policy actually followed by AlphaGo Zero was an *improvement* over the network's initial policy (`p`).
*   Silver et al. (2017a) stated: "*MCTS may therefore be viewed as a powerful policy improvement operator.*"

### Network Architecture and Training

*   **Input:** A 19x19x17 image stack:
    *   8 planes: Current player's stones (current + 7 past configurations). Binary (1=stone, 0=no stone).
    *   8 planes: Opponent player's stones (current + 7 past configurations). Binary.
    *   1 plane: Constant value indicating player color (1=black, 0=white).
    *   *(Past positions and color were needed because Go is not strictly Markovian due to repetition rules and compensation points).*
*   **Architecture:** "Two-headed" network:
    *   Initial shared convolutional layers (41 layers, each followed by batch normalization, with residual skip connections).
    *   Split into two heads:
        1.  **Policy Head:** Fed 362 output units (19x19 + 1 pass/resign move) producing policy probabilities `p`. (Total 43 layers).
        2.  **Value Head:** Fed 1 output unit producing the value estimate `v`. (Total 44 layers).
*   **Training:**
    *   **Method:** Stochastic Gradient Descent (SGD) with momentum, regularization, and decaying step-size.
    *   **Data:** Batches sampled uniformly from the steps of the most recent 500,000 self-play games using the current best policy.
    *   **Exploration:** Extra noise added to the network's output `p` during self-play to encourage exploration.
    *   **Objective:** Update network weights to:
        *   Make network policy `p` more closely match the MCTS policy `π`.
        *   Make network value `v` more closely match the final game winner `z` (game outcome).
*   **Evaluation and Promotion:**
    *   Periodically (every 1,000 training steps), the latest network was evaluated against the current best policy over 400 games (using MCTS with 1,600 iterations per move).
    *   If the new policy won by a set margin (to reduce noise), it became the new best policy for subsequent self-play.

### Training Details and Performance

*   **Initial Training:** 4.9 million self-play games (~3 days).
    *   MCTS: 1,600 iterations per move (~0.4 seconds/move).
    *   Network Updates: 700,000 batches (2,048 configurations each).
*   **Results vs. Previous AlphaGo Versions:**
    *   Elo Ratings:
        *   AlphaGo Zero: 4,308
        *   AlphaGo (vs. Fan Hui): 3,144
        *   AlphaGo (vs. Lee Sedol): 3,739
    *   Predicted to win against prior versions with near certainty.
    *   **Defeated the Lee Sedol version 100-0** in a 100-game match under identical conditions.
*   **Comparison with Supervised Learning:**
    *   An identical network trained *only* on human expert moves (~30 million positions) initially played better and predicted human moves better.
    *   However, AlphaGo Zero surpassed the supervised version after just one day of self-play training.
    *   This suggested AlphaGo Zero had discovered strategies *different* from human play, including novel variations of classical sequences.
*   **Extended Training:**
    *   A larger network trained over 29 million self-play games (~40 days).
    *   Achieved Elo rating of 5,185.
    *   Defeated *AlphaGo Master* (Elo 4,858, strongest Go program at the time, used human data/features) 89-11 in a 100-game match.

### Conclusion and AlphaZero

AlphaGo Zero demonstrated that **superhuman performance could be achieved by pure reinforcement learning**, augmented by a simple MCTS and deep ANNs, with minimal domain knowledge and no human data.

This work led to ***AlphaZero*** (Silver et al. 2017b), a more general algorithm that improved over the world's best programs in Go, chess, and shogi, without even incorporating specific knowledge of Go rules beyond the basics.
