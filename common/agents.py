class Agent:
    def get_action(self, obs, recording=True):
        raise NotImplementedError

    def update(self, reward, done):
        """
        Add new rewards and information to buffer
        """
        raise NotImplementedError
