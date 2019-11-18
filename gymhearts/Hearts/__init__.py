from .Hearts import *

from gym.envs.registration import register

register(
    id='Hearts_Card_Game-v0',
    entry_point='gymhearts.Hearts.Hearts:HeartsEnv',
    kwargs={'playersName': ['Kazuma', 'Aqua', 'Megumin', 'Darkness'], 'maxScore': 100}
)