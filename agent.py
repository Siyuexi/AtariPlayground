import torch
import numpy as np
import os
import time
import visdom
import loguru
from tqdm import tqdm
from collections import OrderedDict
"""
PLAY
CREATED BY SIYUEXI
2022.07.04
"""
class Player():
    def __init__(self, id, net, env, mem, hp, game, mode, ddqn) -> None:
        self.id = id
        self.game = game
        self.mode = mode
        self.ddqn = ddqn
        # infrastructure
        self.net = net[0]
        self.subnet = net[1]
        self.env = env
        self.mem = mem
        # hyper parameters
        self.hp = hp
        self.train_start = hp['train_start']
        self.test_start = hp['test_start']
        # device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        
        
    def execute(self):
        if self.mode == "train":
            self._train()
        else:
            self._test()

    def _train(self):
        # collecting experience first
        self.__experience_collect()
        # then training agent
        self.__train_loop()

    def _test(self):
        # testing agent first
        self.__test_loop()
        # then visualize
        self.__experience_visualize()

    def __experience_collect(self):
        print("collecting experience...")
        last_state = self.env.reset()
        for step in tqdm(range(self.train_start)):
            # random sampling actions
            action = self.env.action_space.sample()
            # interact with environment
            state, reward, done, info = self.env.step(action)
            """ self.mem.push_quadro(last_state, action, reward, state) """
            cur_index = self.mem.store_memory_obs(last_state)
            self.mem.store_memory_effect(cur_index, action, reward, done)
            if done:
                state = self.env.reset()
            last_state = state
        print("experience collecting finished...")

    
    def __train_loop(self):
        # extract hyper parameters
        gamma = self.hp['gamma']
        epsilon = self.hp['epsilon']
        batch_size = self.hp['batch_size']
        iteration = self.hp['iteration']
        lr = self.hp['lr']
        f = self.hp['f']
        f_save = self.hp['f_save']
        f_update = self.hp['f_update']
        f_epoch = self.hp['f_epoch']
        f_episode = self.hp['f_episode']

        # learning settings
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.RMSprop(self.net.parameters(), lr=lr)
        self.net = self.net.to(self.device)
        self.subnet = self.subnet.to(self.device)

        # if exists, load parameters from former model
        load_flag = False
        if os.path.exists("./checkpoint/" + self.game + self.id + ".pth"):
            self.net.load_state_dict(torch.load("./checkpoint/" + self.game + self.id + ".pth"))
            self.subnet.load_state_dict(torch.load("./checkpoint/" + self.game + self.id + ".pth"))
            load_flag = True
            print("latest checkpoint loaded...")
        
        # log settings
        viz = visdom.Visdom(env="player_" + self.game + self.id, log_to_filename="./log/viz/" + self.game + self.id + ".log")
        log = loguru.logger
        # log.remove(handler_id=None)
        log.add("./log/uru/" + self.game + self.id + ".log")
        
        # cache 
        epoch = 1 # number of ended training
        epoch_ = 0 # last epoch
        episode = 1 # number of ended games
        episode_ = 0 # last episode
        mean_reward = 0 # re-calculated every episode
        mean_step = 0 # re-calculated every episode
        avg_reward = 0 # re-calculated every f_episode episode
        avg_step = 0 # re-calculated every f_episode episode
        avg_loss = 0 # re-calculated every f_epoch steps
        log_step = 0 # from 1 to f_epoch
        log_game = 0 # from 1 to f_episode

        # main loop
        # convert list to numpy array to tensor
        last_state = self.env.reset()
        print("training agent...")
        # in DRL, one epoch = one step (or multiple steps) TD process
        # in DRL, one episode = one ended game
        epsilon_ = epsilon
        for step in tqdm(range(1, iteration + 1)):
            # encode obs as k frames state
            """ last_state = torch.from_numpy(np.array(last_state)).unsqueeze(0).float().to(self.device) # state shape: [1, k, w, h] tensor float """
            cur_index = self.mem.store_memory_obs(last_state)
            encoded_state = self.mem.encoder_recent_observation()
            encoded_state = change_to_tensor(encoded_state).unsqueeze(0)
            # get expierence
            # if it is a new trained model, epsilon_ changed from 1 to epsilon linearly to accelerate training
            if step < self.test_start and not load_flag:
                epsilon_ = 1 - (1 - epsilon) * (1.0 * step / self.test_start)
            # sampling for replay buffer
            if np.random.random() > epsilon_:
                """ action = self.net(last_state).max(dim=1)[1].item() """
                action = self.net(encoded_state).max(dim=1)[1].item()
            else: 
                action = self.env.action_space.sample()

            state, reward, done, info = self.env.step(action)
            """
            last_state = last_state.squeeze(0).cpu().byte().numpy() # last_state shape: [k, w, h] array 8bit
            self.mem.push_quadro(last_state, action, reward, state)
            """
            self.mem.store_memory_effect(cur_index, action, reward, done)

            mean_reward += reward
            mean_step += 1
            if done:
                # state = torch.from_numpy(np.array(self.env.reset())).unsqueeze(0).float().to(self.device) # state shape: [1, k, w, h] tensor 32bit
                state = self.env.reset()
                avg_reward += mean_reward
                avg_step += mean_step
                mean_reward = 0
                mean_step = 0

                episode += 1

            last_state = state

            # get backward
            if step % f == 0:
                """
                state_batch, action_batch, reward_batch, state_batch_, index_batch = self.mem.get_batch(batch_size)
                state_batch = torch.from_numpy(state_batch).float().to(self.device) # s shape: [b, a] float
                action_batch = torch.from_numpy(action_batch).unsqueeze(1).long().to(self.device) # a shape: [b. 1] long
                reward_batch = torch.from_numpy(reward_batch).unsqueeze(1).float().to(self.device)# r shape: [b, 1] float
                state_batch_ = torch.from_numpy(state_batch_).float().to(self.device) # s_ shape: [b, a] float
                """
                state_batch, state_batch_, action_batch, reward_batch, _ = self.mem.sample_memories(batch_size)
                state_batch, state_batch_ = change_to_tensor(state_batch), change_to_tensor(state_batch_)
                action_batch, reward_batch = change_to_tensor(action_batch, torch.int64), change_to_tensor(reward_batch)

                
                # estimate Q-Star 
                q_values = self.net(state_batch) # q shape: [b, a] tensor float
                q_pred = q_values.gather(dim=1, index=action_batch) # q shape: [b, 1] tensor float
                # estimate Q-Star_ : Target Network or Double DQN

                if self.ddqn:
                    # Double DQN
                    q_values_ = self.net(state_batch_).detach() # q shape: [b, a] tensor float
                    action_batch_ = q_values_.max(dim=1)[1].unsqueeze(1) # a_ shape: [b, 1] int
                    q_target = reward_batch + gamma * self.subnet(state_batch_).gather(dim=1, index=action_batch_) # q shape: [b, 1] tensor float
                else:
                    # Target Network
                    q_values_ = self.subnet(state_batch_).detach()
                    q_target = reward_batch + gamma * q_values_.max(dim=1)[0].unsqueeze(1)
                    
                # calculating loss
                loss = criterion(q_pred, q_target)
                avg_loss += loss.item()

                """
                update error weight
                error = loss
                error = error.detach().cpu().numpy()
                self.mem.set_batch(error, index_batch)
                """

                # backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch += 1

            # update subnet parameters
            if step % f_update == 0:
                self.subnet.load_state_dict(OrderedDict(self.net.state_dict()))

            # save checkpoint per f_save steps
            if step % f_save == 0:
                torch.save(self.net.state_dict(), "./checkpoint/temp/" + self.game + self.id + str(step) + ".pth")

            # log per f_epoch epoches, skip the same epoch
            if epoch % f_epoch == 0 and epoch_ != epoch:
                log_step += 1
                avg_loss = avg_loss / f_epoch
                viz.line([avg_loss], [log_step], win="average_loss", update='append', 
                    opts=dict(title="average_loss per %d batches" % f_epoch, xlabel="batches/%d" % f_epoch, ylabel="average_loss"))
                log.debug( 'Epoch [{}]\tAverageLoss: {:.6f}\t'.format(epoch, avg_loss)) 
                epoch_ = epoch               
                avg_loss = 0

            # log per f_episode episodes, skip the same episode
            if episode % f_episode == 0 and episode_ != episode:
                log_game += 1
                avg_reward = avg_reward / f_episode
                avg_step = avg_step / f_episode
                viz.line([avg_reward], [log_game], win="average_reward", update='append',
                    opts=dict(title="average_reward per %d games" % f_episode, xlabel="games/%d" % f_episode, ylabel="average_reward"))
                viz.line([avg_step], [log_game], win="average_steps", update='append',
                    opts=dict(title="average_steps per %d games" % f_episode, xlabel="games/%d" % f_episode, ylabel="average_steps"))
                log.info( 'Episode [{}]\tAverageReward: {:.6f}\tAverageStep: {:.6f}\t'.format(episode, avg_reward, avg_step))
                episode_ = episode
                avg_reward = 0
                avg_step = 0        
            
        # save final model parameters
        torch.save(self.net.state_dict(), "./checkpoint/" + self.game + self.id + ".pth")
        print("agent training finished...")

    
    def __test_loop(self):
        # loading parameters
        assert os.path.exists("./checkpoint/" + self.game + self.id + ".pth"), "CHECKPOINT NOT EXSITS."
        self.net.load_state_dict(torch.load("./checkpoint/" + self.game + self.id + ".pth"))
        self.net.to(self.device)
        # cache
        episode = 0
        epoch = 0
        total_reward = 0

        state = self.env.reset()
        print("testing agent...")
        while episode < 10:

            cur_index = self.mem.store_memory_obs(state)
            encoded_state = self.mem.encoder_recent_observation()  # numpy: [m*c, h, w]

            sample = np.random.random()
            if sample > 0.05:
                # numpy: [m*c, h, w] => tensor: [1, m*c, h, w]
                encoded_state = change_to_tensor(encoded_state).unsqueeze(0)
                action = self.net(encoded_state).max(dim=1)[1].item()
            else:
                action = self.env.action_space.sample()

            # action = env.action_space.sample()
            state, reward, done, _ = self.env.step(action)
            self.mem.store_memory_effect(cur_index, action, reward, done)

            total_reward += reward
            epoch += 1
            self.env.render()
            time.sleep(0.05)

            if done:
                print("game %d done in %d steps with %d return" % (episode, epoch, total_reward))
                time.sleep(1)
                self.env.reset()
                episode += 1
                epoch = 0
                total_reward = 0
        
        self.env.close()
        print("agent testing finished")
        
    def __experience_visualize(self):
        pass


def change_to_tensor(data_np, dtype=torch.float32):

    data_tensor = torch.from_numpy(data_np).type(dtype)
    if torch.cuda.is_available():
        data_tensor = data_tensor.cuda()
    return data_tensor


"""UNIT TESTING"""
# >>> import torch
# >>> x = torch.tensor([[1,2],[3,4]])
# >>> x
# tensor([[1, 2],
#         [3, 4]])
# >>> x.max(dim=1)
# torch.return_types.max(
# values=tensor([2, 4]),
# indices=tensor([1, 1]))
# >>> x[0].view(2,-1)
# tensor([[1],
#         [2]])
#
# >>> x = torch.tensor([1,2,3,4])
# >>> x.shape
# torch.Size([4])
# >>> x
# tensor([1, 2, 3, 4])
# >>> x.unsqueeze(1)
# tensor([[1],
#         [2],
#         [3],
#         [4]])