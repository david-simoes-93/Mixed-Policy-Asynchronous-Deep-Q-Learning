import random
import matplotlib.pyplot as mpl
import numpy as np

from mixedQ.WolFPHC import wolfq
from mixedQ.WPL2 import wpl2
from mixedQ.EMAQL import emaql
from mixedQ.GIGA import gigawolf
from mixedQ.PGAPP import pgaapp
from mixedQ.Q import q
from mixedQ.Projection import projection


def get_action(prob, s):
    if random.random()<exploration_rate:
        return int(random.random()*number_of_actions)
    action_prob = random.random()*sum(prob[s])
    for i in range(len(prob[s])):
        if action_prob<prob[s][i]:
            return i
        action_prob -= prob[s][i]
    return -1


#mpl.grid(True)
#axes = mpl.gca()
#axes.set_xlim([xmin,xmax])
#axes.set_ylim([0,1])
mpl.show()

for game in [5]: #range(4): #todo

    number_of_agents = 2
    number_of_states = 1
    number_of_actions = 2
    exploration_rate = 0.05
    learning_rate = .01
    future_rewards_importance = 0.9

    if game==0:
        R1=[[ 1,-1],
            [-1, 1]] #matching pennies
        R2=[[-1, 1],
            [ 1,-1]]
    elif game==1:
        R1=[[0, 3],
            [1, 2]] # tricky
        R2=[[3, 2],
            [0, 1]]
    elif game==2:
        R1=[[1.00, 1.75],
            [1.25, 1.00]] # biased
        R2=[[1.75, 1.00],
            [1.00, 1.25]]
    elif game==3:
        number_of_actions = 3
        R1 = [[0, -1, 1],
              [1, 0, -1],
              [-1, 1, 0]]  # rock paper scissors
        R2 = [[0, 1, -1],
              [-1, 0, 1],
              [1, -1, 0]]  # rock paper scissors
    elif game==4:
        number_of_actions = 4
        R1 = [[1, 1, -1, -1],
              [-1, 1, -1, 1],
              [1, -1, 1, -1],
              [-1, -1, 1, 1]]  # rock paper scissors
        R2 = [[-1, -1, 1, 1],
              [1, -1, 1, 1],
              [-1, 1, -1, 1],
              [1, 1, -1, -1]]  # rock paper scissors
    elif game==5:
        number_of_actions = 4
        R1 = [[-1, -1, -1, -1],
              [-1, 2, 1, 3],
              [-1, 3, 2, 1],
              [-1, 1, 3, 2]]  # rock paper scissors
        R2 = [[-1, -1, -1, -1],
              [-1, 2, 3, 1],
              [-1, 1, 2, 3],
              [-1, 3, 1, 2]]  # rock paper scissors

    for algorithm in [1]: #range(5): #todo
        print("Alg: ",algorithm)

        plot_prob1=[]
        plot_prob2=[]
        plot_prob3 = []
        plot_prob4 = []
        plot_opp_prob1=[]
        plot_opp_prob2=[]
        plot_q0=[]
        plot_q1=[]
        plot_phc=[]

        log_p1=[]
        log_p2=[]

        Q1 = np.zeros((number_of_states,number_of_actions)).tolist()
        Q2 = np.zeros((number_of_states,number_of_actions)).tolist()

        C1 = [0]*number_of_states
        C2 = [0]*number_of_states

        Prob1 = np.full((number_of_states,number_of_actions), 1/number_of_actions).tolist()
        #Prob1[0][0]= random.random()
        #Prob1[0][0]=0.5
        #Prob1[0][1] = 1 - Prob1[0][0] / 2
        #Prob1[0][2] = 1 - Prob1[0][0] / 2
        #if number_of_actions>2:
        #    Prob1[0][2] = 0
        Prob2 = np.full((number_of_states,number_of_actions), 1/number_of_actions).tolist()
        #Prob2[0][0]= random.random()
        #Prob2[0][0]=0.5
        #Prob2[0][1]=1-Prob2[0][0]
        #if number_of_actions>2:
        #    Prob2[0][2] = 0

        Prob_1 = np.zeros((number_of_states,number_of_actions)).tolist()
        Prob_2 = np.zeros((number_of_states,number_of_actions)).tolist()
        #Prob_1 = list(Prob1)
        Prob_1[0][0]=Prob1[0][0]
        Prob_1[0][1]=Prob1[0][1]
        Prob_2[0][0]=Prob2[0][0]
        Prob_2[0][1]=Prob2[0][1]
        if number_of_actions > 2:
            Prob_1[0][2] = Prob1[0][2]
            Prob_2[0][2] = Prob2[0][2]
        if number_of_actions > 3:
            Prob_1[0][3] = Prob1[0][3]
            Prob_2[0][3] = Prob2[0][3]
        #Prob_2 = list(Prob2)
        its = []
        for iteration in range(5000000):
            action1 = get_action(Prob1, 0)
            action2 = get_action(Prob2, 0)
            #action1 = int(random.random()*number_of_actions) if random.random()<exploration_rate else (0 if Q1[0][0]>Q1[0][1] else 1)
            #action2 = int(random.random()*number_of_actions) if random.random()<exploration_rate else (0 if Q2[0][0]>Q2[0][1] else 1)
            reward1 = R1[action1][action2]
            reward2 = R2[action1][action2]

            #print(0, 0, action1, reward1, Q1, Prob1, Prob_1, C1)
            #print(0, 0, action1, reward1, Q1, Prob1, Prob_1, C1)

            if algorithm==0:        # Wolf-PHC
                learning_rate = 0.01
                eta_win = max(0.000001,learning_rate/(100+iteration/2000))
                eta_lose=2*eta_win

                wolfq(0, 0, action1, reward1, Q1, Prob1, Prob_1, C1, learning_rate, future_rewards_importance, eta_win, eta_lose)
                wolfq(0, 0, action2, reward2, Q2, Prob2, Prob_2, C2, learning_rate, future_rewards_importance, eta_win, eta_lose)
            elif algorithm==1:      # WPL
                learning_rate = 0.01
                eta = max(0.000001,(learning_rate)/(100+iteration/2000))
                #eta = learning_rate/100.0

                wpl2(0, 0, action1, reward1, Q1, Prob1, learning_rate, future_rewards_importance, exploration_rate, eta)
                wpl2(0, 0, action2, reward2, Q2, Prob2, learning_rate, future_rewards_importance, exploration_rate, eta)
            elif algorithm==2:      # EMA-QL
                learning_rate = 0.01
                eta_win = max(0.000001,learning_rate/(100+iteration/2000))
                eta_lose = 2*eta_win

                emaql(0, 0, action1, reward1, Q1, Prob1, learning_rate, future_rewards_importance, eta_win, eta_lose)
                emaql(0, 0, action2, reward2, Q2, Prob2, learning_rate, future_rewards_importance, eta_win, eta_lose)
            elif algorithm==3:      # GIGA-WolF
                learning_rate = 0.01
                eta = max(0.000001,(learning_rate)/(100+iteration/2000))

                gigawolf(0, 0, action1, reward1, Q1, Prob1, Prob_1, learning_rate, future_rewards_importance, eta)
                gigawolf(0, 0, action2, reward2, Q2, Prob2, Prob_2, learning_rate, future_rewards_importance, eta)
            elif algorithm==4:      # PGA-APP
                learning_rate = 0.01
                eta = max(0.000001,learning_rate/(100+iteration/2000))
                gamma = 1 if game == 2 else 10

                pgaapp(0, 0, action1, reward1, Q1, Prob1, learning_rate, future_rewards_importance, eta, gamma)
                pgaapp(0, 0, action2, reward2, Q2, Prob2, learning_rate, future_rewards_importance, eta, gamma)
            elif algorithm==5:      # Q-learning (not mixed)
                learning_rate = 0.01
                q(0, 0, action1, reward1, Q1, learning_rate, future_rewards_importance, 0)
                q(0, 0, action2, reward2, Q2, learning_rate, future_rewards_importance, 1)
                Prob1[0][0] = Q1[0][0]
                Prob1[0][1] = Q1[0][1]
                projection(Prob1[0], 0)
                Prob2[0][0] = Q2[0][0]
                Prob2[0][1] = Q2[0][1]
                projection(Prob2[0], 0)
            """elif algorithm==6:      # WPL
                learning_rate = 0.01
                eta = max(0.000001,(learning_rate)/(100+iteration/2000))
                #eta = learning_rate/100.0

                wpl3(0, 0, action1, reward1, Q1, Prob1, learning_rate, future_rewards_importance, exploration_rate, eta)
                wpl3(0, 0, action2, reward2, Q2, Prob2, learning_rate, future_rewards_importance, exploration_rate, eta)
            """
            #Prob2[0][0] = 0.2
            #Prob2[0][1] = 1 - Prob2[0][0]

            if iteration % 1000 == 0:
                #print(iteration)
                log_p1.append(list(Prob1[0]))
                log_p2.append(list(Prob2[0]))

                its.append(iteration*2)
                plot_prob1.append(Prob1[0][0])
                plot_prob2.append(Prob2[0][0])
                if number_of_actions>3:
                    plot_prob3.append(Prob1[0][2])
                    plot_prob4.append(Prob1[0][3])
                plot_opp_prob1.append(Prob2[0][0])
                plot_opp_prob2.append(Prob2[0][1])
                #plot_q0.append(Q1[0][0])
                #plot_q1.append(Q1[0][1])

                mpl.clf()
                mpl.plot(its, plot_prob1, 'r')
                mpl.plot(its, plot_prob2, 'r--')
                if number_of_actions>3:
                    mpl.plot(its, plot_prob3, 'y')
                    mpl.plot(its, plot_prob4, 'y--')
                #mpl.plot(its, plot_opp_prob1, 'g')
                #mpl.plot(its, plot_opp_prob2, 'g--')

                #mpl.plot(its, plot_q0)
                #mpl.plot(its, plot_q1)
                mpl.pause(0.0001)

            if (iteration) % 10000 == 0:
                print('1 - ', action1, "%5.2f" % reward1, '['+' '.join('{0:.{1}f}'.format(k, 3) for k in Q1[0])+']',
                      '['+' '.join('{0:.{1}f}'.format(k, 3) for k in Prob1[0])+']',
                      '['+' '.join('{0:.{1}f}'.format(k, 3) for k in Prob_1[0])+']')
                print('2 - ', action2, "%5.2f" % reward2, '['+' '.join('{0:.{1}f}'.format(k, 3) for k in Q2[0])+']',
                      '['+' '.join('{0:.{1}f}'.format(k, 3) for k in Prob2[0])+']',
                      '['+' '.join('{0:.{1}f}'.format(k, 3) for k in Prob_2[0])+']')
                print()
        #input("")


        with open("MaQL.txt", "a") as myfile:
            #print(str(algorithm), str(plot_prob1))
            myfile.write(str(algorithm)+" "+str(log_p1)+" "+str(log_p2)+"\n")

