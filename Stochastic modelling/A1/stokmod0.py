import numpy as np
#import matplotlib.pyplot as plt

tracker_1 = []
tracker_2 = []
tracker_3 = []

def pandemic(days): #days=timesteps
    beta = 0.05
    gamma = 0.1
    alfa = 0.01
    rnum = 0
    P = np.array([[1-beta,beta,0],[0,1-gamma,gamma],[alfa,0,1-alfa]])
    state = 0
    current_days_in_state = 0
    #tracker_1 = []
    #tracker_2 = []
    #tracker_3 = []
    for n in range(days):
        rnum=np.random.uniform(0,1)
        if state == 0:
            if rnum<P[0][0]:
                current_days_in_state += 1 #holder styr på antall dager uten å endre state
                continue
            else:
                tracker_1.append(current_days_in_state+1)
                current_days_in_state = 0
                state = 1
        elif state == 1:
            if rnum<P[1][1]:
                current_days_in_state += 1
                continue
            else:
                tracker_2.append(current_days_in_state+1)
                current_days_in_state = 0
                state = 2
        else:
            if rnum<P[2][2]:
                current_days_in_state += 1
                continue
            else:
                tracker_3.append(current_days_in_state+1)
                current_days_in_state = 0
                state = 0
    return None

#Finner forventingsverdier
print('Forventing P_12 \t Forventing P_23 \t Forventing P_31')
for j in range(5):
        pandemic(18250)
        total_1 = 0
        total_2 = 0
        total_3 = 0
        for i in range(len(tracker_1)):
            total_1 += tracker_1[i]
        real_total_1 = total_1/len(tracker_1)
        for i in range(len(tracker_2)):
             total_2 += tracker_2[i]
        real_total_2 = total_2/len(tracker_2)
        for i in range(len(tracker_3)):
            total_3 += tracker_3[i]
        real_total_3 = total_3/len(tracker_3)
        print(f'{real_total_1:.3f} \t\t\t\t{real_total_2:.3f} \t\t\t\t {real_total_2:.3f}')