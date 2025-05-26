from gymnasium.envs.registration import register

register(
    id='Gomoku15x15-v0',
    entry_point='gym_gomoku.envs:GomokuEnv',
    kwargs={
        'board_size': 15,
    },
    nondeterministic=True,
)
