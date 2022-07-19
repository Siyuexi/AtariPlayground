import numpy as np
"""
PLEASE JUST USE LITE BUFFER AND IGNORE OTHERS.
CREATED BY SIYUEXI
2022.07.03
"""

class Exp():
    # one step TD experience
    def __init__(self):
        # basic quadro tuple
        self.__s = None
        self.__a = None
        self.__r = None
        self.__s_ = None
        # memorize TD error for random delete in buffer
        self.__e = -1

    def set_quadro(self, s, a, r, s_):
        self.__s = s
        self.__a = a
        self.__r = r
        self.__s_ = s_

    def set_error(self, e):
        self.__e = e

    def get_quadro(self):
        return self.__s, self.__a, self.__r, self.__s_

    def get_error(self):
        return self.__e


class Buffer():
    # a wrapped Exp array
    def __init__(self, memory_capacity, w, h, n):
        # image settings
        self.w = w
        self.h = h
        self.n = n
        # buffer settings
        self.capacity = memory_capacity
        self.memory = [Exp() for i in range(memory_capacity)]
        # index means the index of the last operation element: increasing by order or selecting by random weight
        self.index = 0
        self.is_full = False

    def push_quadro(self, s, a, r, s_):
        # if it is full, we first choose an experience to delete by weighted_random_select
        if self.is_full:
            self.weighted_select()
            self.memory[self.index].set_quadro(s, a, r, s_)
        # else we insert experience by order
        else:
            self.memory[self.index].set_quadro(s, a, r, s_)
            self.index += 1
            if self.index == self.capacity:
                self.is_full = True
    
    def push_error(self, e, index):
        # error = mean(all historical error, new error)
        if self.memory[index].get_error() == -1:
            self.memory[index].set_error(e)
        else:
            e_ = self.memory[index].get_error()
            self.memory[index].set_error((e + e_)/2.0)

    def pull_all(self):
        # if it is full, we randomly choose an experience in all memory buffer
        if self.is_full:
            index = np.random.randint(0, self.capacity)
        # else we randomly choose an experience in current buffer capacity
        else:
            index = np.random.randint(0, self.index)
        return self.memory[index].get_quadro(), index # quadro shape: [n, w, h] array 8bit

    def weighted_select(self):
        # select a min_e_index as new last operation index when memory is full
        # using weighted sampling skills to acts like a priority queue
        index_list = [(self.index + i) % self.capacity for i in range(10)]
        min_e_index = index_list[0]
        for i in index_list:
            e = self.memory[min_e_index].get_error()
            e_ = self.memory[i].get_error()
            if e > e_:
                min_e_index = i
        self.index = min_e_index

    def get_batch(self, batch_size):
        # to get a batch of experiece: [b, k, w, h] and their index
        state_batch = np.zeros([batch_size, self.n, self.w, self.h])
        action_batch = np.zeros(batch_size)
        reward_batch = np.zeros(batch_size)
        state_batch_ = np.zeros([batch_size, self.n, self.w, self.h])
        index_batch = [0 for i in range(batch_size)]
        for i in range(batch_size):
            (s, a, r, s_), index = self.pull_all()
            state_batch[i, :, :, :] = s
            action_batch[i] = a
            reward_batch[i] = r
            state_batch_[i, :, :, :] = s_
            index_batch[i] = index
        
        return state_batch, action_batch, reward_batch, state_batch_, index_batch

    def set_batch(self, error, index_batch):
        # to set a batch of error e with their index
        batch_size = len(index_batch)
        for i in range(batch_size):
            self.push_error(error, index_batch[i])


class Buffer_lite():
    def __init__(self, pool_size, frame_history_len):
        self.pool_size = pool_size
        self.frame_history_len = frame_history_len

        self.memories = None
        self.obs_shape = None

        self.number_of_memories = 0
        self.next_idx = 0

    def _check_idx(self, cur_idx):
        """
        if memory pool cannot meet "frame_history_len" frames, then padding 0.

        situation 1: cur_idx < frame_history_len and memory pool is not full      --> padding 0
        situation 2: cur_idx < frame_history_len and memory pool is full          --> no padding
        situation 3: appear "stop" flag (check from end to start)                 --> padding 0
        situation 4: other                                                        --> no padding

        :return: idx_flag, missing_context, start_idx, end_idx
        """
        end_idx = cur_idx + 1  # exclusive
        start_idx = end_idx - self.frame_history_len  # inclusive
        is_sit_3 = False

        # situation 1 or 2 or 3
        if start_idx < 0:
            start_idx = 0
            missing_context = self.frame_history_len - (end_idx - start_idx)

            # situation 1
            if self.number_of_memories != self.pool_size:
                # not check end frame
                for idx in range(start_idx, end_idx-1):
                    # 0, 1|, 0, 0|, ...
                    if self.memories[idx % self.pool_size]['done']:
                        start_idx = idx + 1
                        is_sit_3 = True

                    if is_sit_3:
                        missing_context = self.frame_history_len - (end_idx - start_idx)
                        return 3, missing_context, start_idx, end_idx

                return 1, missing_context, start_idx, end_idx

            # situation 2
            else:
                for idx in range(start_idx, end_idx-1):
                    if self.memories[idx % self.pool_size]['done']:
                        start_idx = idx + 1
                        is_sit_3 = True

                    if is_sit_3:
                        missing_context = self.frame_history_len - (end_idx - start_idx)
                        return 3, missing_context, start_idx, end_idx

                # not check end frame
                for i in range(missing_context, 0, -1):
                    idx = self.pool_size - i
                    if self.memories[idx % self.pool_size]['done']:
                        start_idx = (idx + 1) % self.pool_size
                        is_sit_3 = True

                    if is_sit_3:
                        # ..., end_idx|, ..., |start_idx, ., end
                        if start_idx > end_idx:
                            missing_context = self.frame_history_len - (self.pool_size - start_idx + end_idx)
                        else:
                            missing_context = self.frame_history_len - (end_idx - start_idx)
                        return 3, missing_context, start_idx, end_idx

                start_idx = self.pool_size - missing_context
                return 2, 0, start_idx, end_idx

        # situation 3: appear "stop" flag
        for idx in range(start_idx, end_idx-1):
            if self.memories[idx % self.pool_size]['done']:
                start_idx = idx + 1
                is_sit_3 = True

            if is_sit_3:
                missing_context = self.frame_history_len - (end_idx - start_idx)
                return 3, missing_context, start_idx, end_idx

        return 4, 0, start_idx, end_idx

    def _encoder_observation(self, cur_idx):
        """
        concatenate recent "frame_history_len" frames
        obs: (c, h, w) => (frame_history_len*c, h, w)
        :param cur_idx: current frame's index
        :return: tensor
        """

        encoded_observation = []

        idx_flag, missing_context, start_idx, end_idx = self._check_idx(cur_idx)

        if missing_context > 0:
            for i in range(missing_context):
                encoded_observation.append(np.zeros_like(self.memories[0]['obs']))

        # situation 3 in situation 2
        if start_idx > end_idx:
            for idx in range(start_idx, self.pool_size):
                encoded_observation.append(self.memories[idx % self.pool_size]['obs'])
            for idx in range(end_idx):
                encoded_observation.append(self.memories[idx % self.pool_size]['obs'])
        else:
            for idx in range(start_idx, end_idx):
                encoded_observation.append(self.memories[idx % self.pool_size]['obs'])

        # encoded_observation: [k, c, h, w] => [k*c, h, w]
        encoded_observation = np.concatenate(encoded_observation, 0)
        return encoded_observation

    def encoder_recent_observation(self):
        """
        concatenate recent "frame_history_len" frames
        :return:
        """
        assert self.number_of_memories > 0

        current_idx = self.next_idx - 1
        # when next_idx == 0
        if current_idx < 0:
            current_idx = self.pool_size - 1

        return self._encoder_observation(current_idx)

    def sample_memories(self, batch_size):
        """
        choose randomly "batch_size" memories (batch_size, )
        :param batch_size:
        :return:
        """
        # ensure s_{i+1} is exist
        sample_idxs = np.random.randint(0, self.number_of_memories-1, [batch_size])

        # [batch_size, frame_history_len*c, h, w]
        obs_batch = np.zeros(
            [batch_size, self.obs_shape[0] * self.frame_history_len, self.obs_shape[1], self.obs_shape[2]])
        next_obs_batch = np.copy(obs_batch)
        action_batch = np.zeros([batch_size, 1])  # [batch_size, ]
        reward_batch = np.zeros([batch_size, 1])  # [batch_size, ]
        done_batch = []

        for i in range(batch_size):
            obs_batch[i] = self._encoder_observation(sample_idxs[i])
            next_obs_batch[i] = self._encoder_observation(sample_idxs[i] + 1)
            action_batch[i] = self.memories[sample_idxs[i]]['action']
            reward_batch[i] = self.memories[sample_idxs[i]]['reward']
            done_batch.append(self.memories[sample_idxs[i]]['done'])

        return obs_batch, next_obs_batch, action_batch, reward_batch, done_batch

    def store_memory_obs(self, frame):
        """
        store observation of memory
        :param frame: numpy array
                      Array of shape (img_h, img_w, img_c) and dtype np.uint8
        :return:
        """
        # obs is a image (h, w, c)
        frame = frame.transpose(2, 0, 1)  # c, w, h

        if self.obs_shape is None:
            self.obs_shape = frame.shape

        if self.memories is None:
            self.memories = [dict() for i in range(self.pool_size)]

        self.memories[self.next_idx]['obs'] = frame
        index = self.next_idx

        self.next_idx = (self.next_idx + 1) % self.pool_size
        self.number_of_memories = min([self.number_of_memories + 1, self.pool_size])

        return index

    def store_memory_effect(self, index, action, reward, done):
        """
        store other information of memory
        :param action: scalar
        :param done: bool
        :param reward: scalar
        :return:
        """
        self.memories[index]['action'] = action
        self.memories[index]['reward'] = reward
        self.memories[index]['done'] = done
    


def get_mem(memory_capacity, w, h, n, lite_buffer):
    if lite_buffer:
        mem = Buffer_lite(memory_capacity, n)
    else:
        mem = Buffer(memory_capacity, w, h, n) 
    return mem


"""UNIT TESTING"""
# a = np.zeros([2,2,3,4])
# print(a.shape)
# print(a)
# b = np.ones([2,3,4])
# a[1,:,:,:] = b
# print(a.shape)
# print(a)