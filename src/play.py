from policies import RandomPolicy
from recording import record_episode

if __name__ == "__main__":
    policy = RandomPolicy()
    record_episode(policy, deepmind_wrappers=False)