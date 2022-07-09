import numpy as np
"""
A REPLAY BUFFER.USING WEIGHTED STOCHASTIC SAMPLING SKILLS TO ACTS LIKE A PRIORITY QUEUE.
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
    pass


def get_mem(memory_capacity, w, h, n, lite_buffer):
    if lite_buffer:
        mem = Buffer_lite()
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