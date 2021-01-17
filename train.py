from utils.getoptions import get_options
from envs.LVRP import LVRP
from models.rl_process import rl_process

if __name__ == '__main__':
    config = get_options()
    env = LVRP(config)
    rl_process(env, config)
