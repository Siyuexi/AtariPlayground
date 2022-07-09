import torch
import numpy as np
import os
import time
import visdom
import loguru
from tqdm import tqdm
"""
PLAY
CREATED BY SIYUEXI
2022.07.04
"""
class Player():
    def __init__(self, net, env, mem, hp, game, mode, train_start, test_start) -> None:
        self.game = game
        self.mode = mode
        # infrastructure
        self.net = net[0]
        self.subnet = net[1]
        self.env = env
        self.mem = mem
        # hyper parameters
        self.hp = hp
        # start epoch
        self.train_start = train_start
        self.test_start = test_start
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
            self.mem.push_quadro(last_state, action, reward, state)
            if done:
                state = self.env.reset()
            last_state = state
        print("experience collecting finished...")

    
    def __train_loop(self):
        # extract hyper parameters
        gamma = self.hp['gamma']
        epsilon = self.hp['epsilon']
        batch_size = self.hp['batch_size']
        epoch = self.hp['epoch']
        lr = self.hp['lr']
        f_save = self.hp['f_save']
        f_update = self.hp['f_update']
        f_epoch = self.hp['f_epoch']
        f_episode = self.hp['f_episode']

        # learning settings
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(self.net.parameters(), lr= lr)
        self.net = self.net.to(self.device)
        self.subnet = self.subnet.to(self.device)

        # if exists, load parameters from former model
        if os.path.exists("./checkpoint/" + self.game + ".pth"):
            self.net.load_state_dict(torch.load("./checkpoint/" + self.game + ".pth"))
            self.subnet.load_state_dict(torch.load("./checkpoint/" + self.game + ".pth"))
            print("latest checkpoint loaded...")
        
        # log settings
        viz = visdom.Visdom(env="player_" + self.game, log_to_filename="./log/viz/" + self.game + ".log")
        log = loguru.logger
        # log.remove(handler_id=None)
        log.add("./log/uru/" + self.game + ".log")
        
        # cache 
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
        state = torch.from_numpy(np.array(self.env.reset())).unsqueeze(0).float().to(self.device) # state shape: [1, k, w, h] tensor 32bit
        print("training agent...")
        # in DRL, one epoch = one step (or multiple steps) TD process
        # in DRL, one episode = one ended game
        for step in tqdm(range(1, epoch + 1)):
            # get expierence
            last_state = state
            # viz.images(state[0], win="current frames")
            if np.random.random() > epsilon:
                action = self.net(state).max(dim=1)[1].cpu().numpy()
            else: 
                action = self.env.action_space.sample()

            state, reward, done, info = self.env.step(action)
            state = torch.from_numpy(np.array(state)).unsqueeze(0).float().to(self.device) # state shape: [1, k, w, h] tensor 32bit
            last_state = last_state.squeeze(0).cpu().byte().numpy() # last_state shape: [k, w, h] array 8bit
            last_state_ = state.squeeze(0).cpu().byte().numpy() # last_state_ shape: [k, w, h] array 8bit
            self.mem.push_quadro(last_state, action, reward, last_state_)

            mean_reward += reward
            mean_step += 1
            if done:
                self.env.seed()
                state = torch.from_numpy(np.array(self.env.reset())).unsqueeze(0).float().to(self.device) # state shape: [1, k, w, h] tensor 32bit
                episode += 1
                avg_reward += mean_reward
                avg_step += mean_step
                mean_reward = 0
                mean_step = 0

            # get backward
            if step % batch_size == 0:
                state_batch, action_batch, reward_batch, state_batch_, index_batch = self.mem.get_batch(batch_size)
                state_batch = torch.from_numpy(state_batch).float().to(self.device) # s shape: [b, a] float
                action_batch = torch.from_numpy(action_batch).unsqueeze(1).long().to(self.device) # a shape: [b. 1] long
                reward_batch = torch.from_numpy(reward_batch).unsqueeze(1).float().to(self.device)# r shape: [b, 1] float
                state_batch_ = torch.from_numpy(state_batch_).float().to(self.device) # s_ shape: [b, a] float
                
                # estimate Q-Star 
                q_values = self.net(state_batch) # q shape: [b, a] tensor float
                q_pred = q_values.gather(dim=1, index=action_batch) # q shape: [b, 1] tensor float
                # estimate Q-Star_
                q_values_ = self.net(state_batch_).detach() # q shape: [b, a] tensor float
                action_batch_ = q_values_.max(dim=1)[1].unsqueeze(1) # a_ shape: [b, 1] int
                q_target = reward_batch + gamma * self.subnet(state_batch_).gather(dim=1, index=action_batch_) # q shape: [b, 1] tensor float

                # calculating loss
                loss = criterion(q_pred, q_target)
                avg_loss += loss.item()

                # update error weight
                error = loss
                error = error.detach().cpu().numpy()
                self.mem.set_batch(error, index_batch)

                # backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # log per f_epoch steps
            if step % f_epoch == 0:
                log_step += 1
                avg_loss = avg_loss / f_epoch
                viz.line([avg_loss], [log_step], win="average_loss", update='append', 
                    opts=dict(title="average_loss per %d steps" % f_epoch, xlabel="steps/%d" % f_epoch, ylabel="average_loss"))
                log.debug( 'Epoch [{}/{}]\tAverageLoss: {:.6f}\t'.format(step , epoch, avg_loss))                
                avg_loss = 0

            # log per f_episode games
            if episode % f_episode == 0:
                # skip the same episode
                if episode_ == episode:
                    continue
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

            # update subnet parameters
            if step % f_update == 0:
                self.subnet.load_state_dict(self.net.state_dict())

            # save checkpoint per f_save steps
            if step % f_save == 0:
                torch.save(self.net.state_dict(), "./checkpoint/temp/" + self.game + "_" + str(log_step) + ".pth")
                
            
        # save final model parameters
        torch.save(self.net.state_dict(), "./checkpoint/" + self.game + ".pth")
        print("agent training finished...")

    
    def __test_loop(self):
        # loading parameters
        assert os.path.exists("./checkpoint/" + self.game + ".pth"), "CHECKPOINT NOT EXSITS."
        self.net.load_state_dict(torch.load("./checkpoint/" + self.game + ".pth"))
        # cache
        total_reward = 0
        total_step = 0

        state = torch.from_numpy(np.array(self.env.reset())).unsqueeze(0).float()
        print("testing agent...")
        while(True):
            if np.random.random() > 0.05:
                action = self.net(state).max(dim=1)[1].numpy()
            else: 
                action = self.env.action_space.sample()
            time.sleep(0.05)
            state, reward, done, info = self.env.step(action)
            state = torch.from_numpy(np.array(state)).unsqueeze(0).float()

            # calculate reward
            total_reward += reward
            total_step += 1

            # render
            self.env.render()
        
            if done:
                print("total reward: " + str(total_reward))
                print("total step: " + str(total_step))
                self.env.close()
                break

        print("agent testing finished")
        
    def __experience_visualize(self):
        pass


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