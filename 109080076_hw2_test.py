import torch
import numpy as np
import cv2
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT

class MarioNet(torch.nn.Module):
        def __init__(self, input_dim, output_dim):
            super().__init__()
            c,h,w = input_dim
            # print(input_dim)
            self.online = self.build_model(c,output_dim)
            self.target = self.build_model(c,output_dim)
            self.target.load_state_dict(self.online.state_dict())
            
            # 先 froze 住 target model 的參數
            for p in self.target.parameters():
                p.requires_grad = False
        def forward(self, input, model):
            if model == "online":
                return self.online(input)
            elif model == "target":
                return self.target(input)
        def build_model(self,c,output_dim):
            return torch.nn.Sequential(
                torch.nn.Conv2d(c, 32, kernel_size=8, stride=4),
                torch.nn.ReLU(),
                torch.nn.Conv2d(32, 64, kernel_size=4, stride=2),
                torch.nn.ReLU(),
                torch.nn.Conv2d(64, 64, kernel_size=3, stride=1),
                torch.nn.ReLU(),
                torch.nn.Flatten(),
                torch.nn.Linear(3136, 512),
                torch.nn.ReLU(),
                torch.nn.Linear(512, output_dim))

class Mario():
    def __init__(self,state_dim,action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = "cpu"        
        # some parameter for network
        self.net = MarioNet(self.state_dim,self.action_dim).float()
        self.net = self.net.to(device = self.device)

        #  some parameter for q learning
        self.explore = -1    

    def act(self,state):
        if np.random.rand() < self.explore:
            # 客製化 action 的 random 選擇
            # action = [NOOP, RIGHT, RIGHT A, RIGHT B, RIGHT A B, A, LEFT, LEFT A, LEFT B, LEFT A B,down,up]
            # weight = [ 1,    1,     1,        1,        2,      1,    1,     1,     1,        1,    1,  1]
            # weight = [1,3,1,3,3,3,1,1,1,1,1,2]
            # weight = [1,3,4,1,1,3,0.5,0.5,0.5,0.5,0,0]
            # weight = [1,1,1,1,1,1,1,1,1,1,1,1]
            # weight = [1,3,1,1,1,1,0.5,0.5,0.5,0.5,0.1,0.1]

            weight = [1 , 1 , 1 ,1 , 1 ,1 , 0.5 , 0.5 , 0 , 0 , 0 , 0 ]
            action = np.random.choice(self.action_dim, p = weight/np.sum(weight))
            # action= np.random.randint(self.action_dim)
        else:
            state = torch.tensor(state).to(self.device)
            state = state.unsqueeze(0)
            action_value = self.net(state, model = "online")
            action = torch.argmax(action_value, axis=1).item()
        return action 
    def load_model(self, path):
        checkpoint = torch.load(path,map_location=self.device)
        self.net.online.load_state_dict(checkpoint['model'])
        self.net.online.eval()
        self.net.target.load_state_dict(checkpoint['model'])
        self.net.target.eval()
        self.explore = checkpoint['exp']
        # self.explore = 0.3
        print(f"load model from {path} explore rate {self.explore}")
        
        
class Agent():
    def __init__(self):
        self.player = Mario((4,84,84),12)
        # player.load_model("./109080076_hw2_data")
        self.player.load_model("./109080076_hw2_data.py")
        self.stack = np.zeros((4,84,84),dtype=np.float32)
        self.count_frame = 0
        self.lastaction = 0

    def act(self, observation):
        # 2 似乎也不錯
        if self.count_frame %3 !=0:
            self.count_frame += 1
            return self.lastaction
        else:
            self.count_frame = 0
            state = self.preprocess(observation)
            action = self.player.act(state)
            self.lastaction = action
            return  action

        # state = self.preprocess(observation)
        # action = self.player.act(state)
        # return  action

    def preprocess(self, observation):
        # gray scale and resize
        observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        observation = cv2.resize(observation, (84, 84), interpolation=cv2.INTER_AREA)
        observation = np.expand_dims(observation, axis=0)
        observation = observation/255.0
        self.stack [:-1] = self.stack[1:]
        self.stack[-1] = observation
        return self.stack
        # stack 4 frame together


# import matplotlib.pyplot as plt

# env = gym_super_mario_bros.make('SuperMarioBros-v0')
# env = JoypadSpace(env, COMPLEX_MOVEMENT)
# agent = Agent()
# epoch = 10
# total_reward = 0
# for _ in range(epoch):
#     state = env.reset()
#     done = False
#     round_reward = 0
#     times =0
#     while not done:
#         action = agent.act(state)
#         next_state, reward, done, info = env.step(action)
#         round_reward += reward
#         state = next_state
#         env.render()
#     print(f"round reward {round_reward}")
#     total_reward += round_reward
# print(f"average reward {total_reward/epoch}")
# env.close()
