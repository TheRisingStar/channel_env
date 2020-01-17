import math
import gym
import logging
import random
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import channelConfig as channelConfig

logger = logging.getLogger(__name__)

CORRECT_REWARD = channelConfig.CORRECT_REWARD  #wrong to correct
CR_TO_CR_REWARD = channelConfig.CR_TO_CR_REWARD  #correct to another correct
PUNISH_REWARD = channelConfig.PUNISH_REWARD  #wrong to itself
WR_TO_WR_REWARD = channelConfig.WR_TO_WR_REWARD  #wrong to another wrong
CR_TO_WR_REWARD = channelConfig.CR_TO_WR_REWARD  #correct to wrong
STUBBORN_REWARD = channelConfig.STUBBORN_REWARD #stick to a certain correct channel

OBSERV_BATCH = channelConfig.OBSERV_BATCH

CHANNEL_CNT = channelConfig.CHANNEL_CNT
STATE_CNT = channelConfig.STATE_CNT #double of channal count

#BLOCK_CNT = channelConfig.BLOCK_CNT
USR_CNT = channelConfig.USR_CNT
REFRESH = channelConfig.REFRESH
#REFRESH_METHOD_OLD = channelConfig.REFRESH_METHOD_OLD
#JAMMER_TYPE = channelConfig.JAMMER_TYPE


class Jammer:
    def __init__(self):
        self.type = channelConfig.JAMMER_TYPE
        self.block_cnt = channelConfig.BLOCK_CNT
        self.state = 0
        self.channel_p = channelConfig.CHANNEL_P[self.state]
        self.p_aggre=[0 for x in range(CHANNEL_CNT+1)]
        for i in range (1,CHANNEL_CNT+1):
            self.p_aggre[i] = self.channel_p[i-1] + self.p_aggre[i-1]

    def changeState(self):
        if self.type == 'Markov_jammer':
            if self.state == 0:
                self.state = 1
            else:
                self.state = 0
            self.channel_p = channelConfig.CHANNEL_P[self.state]
            print (self.channel_p)
            for i in range (1,CHANNEL_CNT+1):
                self.p_aggre[i] = self.channel_p[i-1] + self.p_aggre[i-1]


    def act(self):
        self.changeState()
        channel_available = dict()
        for i in range(CHANNEL_CNT):
            channel_available[i+1] = 1
        if self.type == 'Random_jammer_1':
            # A jammer blockes certain channels with fixed probability
            # The number of jammed channels is different at each timeslot
            for i in range (CHANNEL_CNT):
                ran = random.random()
                print("channel:", i+1)
                if ran < self.channel_p1[i]:
                    #print("random num:", ran)
                    #print("channel_p:", self.channel_p1[i])
                    #print("channel blocked")
                    channel_available[i+1] = 0         
                else:
                    #print("random num:", ran)
                    #print("channel_p:", self.channel_p1[i])
                    #print("channel available")
                    channel_available[i+1] = 1        
        elif self.type == 'Random_jammer_2' or self.type == 'Markov_jammer':
            ran = random.random()
            for i in range (CHANNEL_CNT):
                if (ran >= self.p_aggre[i]) and (ran < self.p_aggre[i+1]):
                    channel_available[i+1] = 0
            available_cnt = CHANNEL_CNT - 1
            while available_cnt > (CHANNEL_CNT-self.block_cnt):
                ran = random.random()
                for i in range (CHANNEL_CNT):
                    if (ran >= self.p_aggre[i]) and (ran < self.p_aggre[i+1]):
                        channel_available[i+1] = 0
                available_cnt = 0
                for i in range (CHANNEL_CNT):
                    available_cnt += channel_available[i+1]
        else:
            print ("No jammer named:" + JAMMER_TYPE)
        return channel_available

class ChannelEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 2
    }

    def __init__(self):
        self.states = range(1,STATE_CNT+1) 
        self.x=[0 for x in range(CHANNEL_CNT)]
        self.y=[0 for x in range(CHANNEL_CNT)]
        self.channel_cnt = CHANNEL_CNT
        self.jammer = Jammer()
        self.state_batch = [0 for x in range(OBSERV_BATCH)]

        for i in range (CHANNEL_CNT):
            if (i % 5 == 0):
                self.x[i] = 200
            elif (i % 5 == 1) :
                self.x[i] = 400
            elif (i % 5 == 2) :
                self.x[i] = 600
            elif (i % 5 == 3) :
                self.x[i] = 800
            else:
                self.x[i] = 1000

        for i in range (CHANNEL_CNT):
            self.y[i] = 1100 - 200 * int(i / 5)

        self.channel_p1 = [0.9, 0.11, 0.85, 0.92, 0.95, 0.99, 0.8, 0.60, 0.98, 0.99, 0.90, 0.05, 0.80, 0.05, 0.91, 0.93, 0.80, 0.95, 0.92, 0.08, 0.90, 0.93, 0.99, 0.37, 0.95, 0.91, 0.99, 0.87, 0.04, 0.91] 

        self.channel_available = self.jammer.act()

        self.actions = [0 for x in range(CHANNEL_CNT)]
        for i in range (CHANNEL_CNT):
            self.actions[i] = i+1

        self.rewards = dict()        
        self.calculateReward()
        self.t = dict()            
        self.updateStateTransfert()

        self.gamma = 0.8         
        self.viewer = None
        self.state = None



    def getGamma(self):
        return self.gamma

    def getStates(self):
        return self.states

    def getAction(self):
        return self.actions
    def getChannelAvailable(self):
        return self.channel_available
    def setAction(self,s):
        self.state=s
    def getRewardChart(self):
        print ("\nReward Chart\n",self.rewards)
    def getTChart(self):
        print ("\nTransformation Chart\n",self.t)

    def isChannelBlocked(self, s):
        return s <= self.channel_cnt

    def getChannelNumber(self, s):
        # get channel number from state
        if s > self.channel_cnt:
            return s - self.channel_cnt
        else:
            return s

    def setStateBatch(self, s):
        self.state_batch = s

   

    def calculateReward(self):
        for i in range (STATE_CNT):   #i in 0-9 means blocked, 10-19 means available
            for j in range (CHANNEL_CNT):
                key = "%d-%d"%(i+1, j+1)
                if self.isChannelBlocked(i):             #i is blocked
                    if (self.channel_available[j+1] == 1):
                        self.rewards[key] = CORRECT_REWARD       #wrong to correct
                    elif (self.channel_available[j+1] == 0):
                        if (i==j):
                            self.rewards[key] = PUNISH_REWARD     #wrong to itself
                        else:
                            self.rewards[key] = WR_TO_WR_REWARD   #wrong to another wrong
                else:                             # i-10 is available
                    if (self.channel_available[j+1] == 0):
                        self.rewards[key] = CR_TO_WR_REWARD           #correct to wrong
                    elif (self.channel_available[j+1] == 1):
                        if (self.getChannelNumber(i) == j):
                            self.rewards[key] = CORRECT_REWARD    #correct to itself
                        else:
                            self.rewards[key] = CR_TO_CR_REWARD   #correct to another correct

    def updateStateTransfert(self):
        for i in range (STATE_CNT):
            for j in range (CHANNEL_CNT):
                key = "%d-%d"%(i+1, j+1)
                if (self.channel_available[j+1] == 0):
                    self.t[key] = j+1
                else:
                    self.t[key] = j + 1 + CHANNEL_CNT

    def step(self, action):
        state = self.state
        print("present state:", state)
        self.channel_available = self.jammer.act()

        key = "%d-%d"%(state, action)  
        print("key:", key)
        #print("reward:",self.rewards[key])
        if key in self.t:
            print("key in dict:", key)
            next_state = self.t[key]
            print("next state:", next_state)
        else:
            print("key not in dict:", key)
            next_state = state
        self.state = next_state

        is_terminal = False

#        if self.channel_available[next_state]:
        if (next_state > CHANNEL_CNT):
                   is_terminal = True #was True

        if key not in self.rewards:
            print ("key not in reward dict")
            r = 0.0
        else:
            print ("key in reward dict")
            r = self.rewards[key]

        if (self.jammer.type == 'Markov_jammer'):
            avail_sum_state = OBSERV_BATCH
            avail_sum_state_next = OBSERV_BATCH                     
            for i in range (OBSERV_BATCH):                           
                if (self.isChannelBlocked(self.state_batch[i])):    
                    avail_sum_state -= 1
            for i in range (OBSERV_BATCH-1):
                if (self.isChannelBlocked(self.state_batch[i])):     
                    avail_sum_state_next -= 1                       
            if (self.isChannelBlocked(next_state)):                 
                avail_sum_state_next -= 1
            print(avail_sum_state)
            print(avail_sum_state_next)

           # if (next_state == 2 * self.state_batch[0]):
           #     r = 500
           # elif (next_state == 0.5 * self.state_batch[0]):
           #     r = -300

           # elif (avail_sum_state < avail_sum_state_next):
           #     r = 100#CORRECT_REWARD
           # elif(avail_sum_state > avail_sum_state_next):
           #     r = PUNISH_REWARD
           # elif(avail_sum_state_next == OBSERV_BATCH):
           #     r = 50#STUBBORN_REWARD
           # else:
           #     r = -10

            r = avail_sum_state_next

        return np.array(next_state), r, 0 ,{}

    def refresh(self):
        if (REFRESH):
            self.channel_available = self.jammer.act()
            self.calculateReward()
            self.updateStateTransfert()

    def reset(self):
        i = int(random.random() * CHANNEL_CNT) + 1
        if self.channel_available[i]:
            self.state = i + CHANNEL_CNT
        else:
            self.state = i
        return np.array(self.state)

    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        screen_width = 1200
        screen_height = 1200

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            #self.line1 = rendering.Line((100,300),(500,300))
            self.chnl = []
            for i in range(CHANNEL_CNT):
                self.chnl.append(rendering.make_circle(50))
                self.circletrans = rendering.Transform(translation=(self.x[i], self.y[i]))
                self.chnl[i].add_attr(self.circletrans)
                if self.channel_available[i + 1]:
                    self.chnl[i].set_color(0,255,0)      
                else:
                    self.chnl[i].set_color(255,0,0) #blocked - red
            self.robot= rendering.make_circle(30)
            self.robotrans = rendering.Transform()
            self.robot.add_attr(self.robotrans)
            self.robot.set_color(255, 255, 255)

            for i in range(CHANNEL_CNT):
                self.viewer.add_geom(self.chnl[i])

            self.viewer.add_geom(self.robot)

        else:
            for i in range(CHANNEL_CNT):
                if self.channel_available[i + 1]:
                    self.chnl[i].set_color(0,255,0)        #available - green
                else:
                    self.chnl[i].set_color(255,0,0) #blocked - red


        if self.state is None: return None
        #self.robotrans.set_translation(self.x[self.state-1],self.y[self.state-1])
        if (self.state > CHANNEL_CNT):
            self.robotrans.set_translation(self.x[self.state-1-CHANNEL_CNT], self.y[self.state-1-CHANNEL_CNT])
        else:
            self.robotrans.set_translation(self.x[self.state-1], self.y[self.state-1])


        return self.viewer.render(return_rgb_array=mode == 'rgb_array')