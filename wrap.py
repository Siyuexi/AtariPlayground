import gym
import cv2
import numpy as np
from collections import deque
"""
COPIED FROM OTHER GIT REPOSITORIES AND WHAT THE HELL IS PARAMETER K IN STACKFRAME?
CREATED BY SIYUEXI
2022.07.02
"""
class PreprocessFrame(gym.ObservationWrapper):
    """
    preprocess the observation of env
    """

    def __init__(self, env):
        super(PreprocessFrame, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1))

    def observation(self, observation):
        """
        raw data: [210, 160, 3]
        processed data: [84, 84, 1]
        :param observation:
        :return:
        """
        img = np.reshape(observation, [210, 160, 3]).astype(np.float32)
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        resized_screen = cv2.resize(img, (84, 110), interpolation=cv2.INTER_LINEAR)
        x_t = resized_screen[18:102, :]
        x_t = np.reshape(x_t, [84, 84, 1])
        return x_t.astype(np.uint8)


class LazyFrames(object):
    # a data structure for stacked frames
    def __init__(self, frames):
        # please using np.array(lazy_frame_instance) to convert obj
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=-1)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[..., i]


class FrameProcessEnv(gym.ObservationWrapper):
    # including every frame processing methods
    def __init__(self,env,width=84,height=84,grayscale=True) -> None:
        super().__init__(env)
        self.width = width
        self.height = height
        self.grayscale = grayscale
        # rgb2gray : 8bit integer for replay buffer memorize
        # need to convert these bytes(8bit data) to float when training
        if self.grayscale:
            self.observation_space = gym.spaces.Box(low=0, high=255,
                shape=(self.height, self.width, 1), dtype=np.uint8)
        else:
            self.observation_space = gym.spaces.Box(low=0, high=255,
                shape=(self.height, self.width, 3), dtype=np.uint8)

    def observation(self,frame):
        if self.grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        if self.grayscale:
            frame = np.expand_dims(frame, -1)
        return frame


class FrameStackEnv(gym.Wrapper):
    # stacking k frames as one
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(shp[:-1] + (shp[-1] * k,)), dtype=env.observation_space.dtype)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))


class FrameTransposeEnv(gym.ObservationWrapper):
    # transpose frame structure to (channel, width, height)
    def __init__(self, env) -> None:
        super().__init__(env)

    def observation(self, frame):
        # convert LazyFrame instance to numpy array
        frame = np.array(frame)
        # transpose sturcture to (c, w, h)
        frame = frame.transpose(2, 0, 1)
        return frame


class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max):
        super().__init__(env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1)
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)


class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._obs_buffer = np.zeros((2,)+env.observation_space.shape, dtype=np.uint8)
        self._skip       = skip

    def step(self, action):
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2: self._obs_buffer[0] = obs
            if i == self._skip - 1: self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)

    def reward(self, reward):
        return np.sign(reward)


class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.lives = 0
        self.was_real_done  = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        lives = self.env.unwrapped.ale.lives()

        if lives < self.lives and lives > 0:
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs



def get_env(name,noop_max, skip, width, height, n, deepmind_wrapper):
    
    env = gym.make(name, obs_type='image')

    # normal wrappers
    env = EpisodicLifeEnv(env)
    env = NoopResetEnv(env, noop_max)
    env = MaxAndSkipEnv(env, skip)
    env = ClipRewardEnv(env)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env) 
    
    # frame wrappers
    if deepmind_wrapper:
        env = PreprocessFrame(env)
    else:
        env = FrameProcessEnv(env, width, height, True)
        env = FrameStackEnv(env, n)
        env = FrameTransposeEnv(env)

    return env


"""UNIT TESTING"""
# # import atari_py
# # print(atari_py.list_games())
# # env = get_env("BreakoutNoFrameskip-v4")
# env = FrameTransposeEnv(FrameStackEnv(FrameProcessEnv(gym.make("BreakoutNoFrameskip-v4", obs_type="image"), 84, 84, True), 4))
# print(env)
# from tqdm import tqdm
# obs = env.reset()
# for step in tqdm(range(100)):
#     # random sampling actions
#     action = env.action_space.sample()
#     # interact with environment
#     obs, reward, done, info = env.step(action)
# print("obs")
# obs = np.array(obs)
# print(obs.shape)
# np.set_printoptions(threshold=np.inf)
# # print(obs)