import numpy as np
import matplotlib.pyplot as plt

def pandemic(days,N,sim_N): #days=timesteps, N=personer
    verdier = np.zeros([days,3])
    max_I_min_T = np.zeros([sim_N,2])
    max_I = 0
    min_T = 0
    for sim_num in range(sim_N):
        state = np.array([950,50,0])
        temp_state = state
        gamma = 0.1
        alfa = 0.01
        beta = 0.5*state[1]/N #Formell som er oppgitt for beta
        rnum = 0 #Hvor mange som går fra state i til i+1
        P = np.array([[1-beta,beta,0],[0,1-gamma,gamma],[alfa,0,1-alfa]])
        max_I=50
        min_T=0
        for k in range(days):
            beta = 0.5*state[1]/N
            P[0][0] = 1-beta
            P[0][1] = beta
            for j in range(3): #Skal oppdaterestate hver timestep, men det gjøres ikke.
                rnum = np.random.binomial(state[j],1-P[j][j])
                if j < 2:
                    temp_state[j]=state[j] - rnum
                    temp_state[j+1] = state[j+1] + rnum
                else:
                    temp_state[j]=state[j]-rnum
                    temp_state[0] = temp_state[0] + rnum
            state=temp_state
            if sim_N == 1:
                verdier[k][0]=state[0]
                verdier[k][1]=state[1]
                verdier[k][2]=state[2]
            if state[1]>max_I:
                max_I = state[1]
                min_T = k
        if sim_N>1:
            max_I_min_T[sim_num][0]=max_I
            max_I_min_T[sim_num][1]=min_T
    if sim_N>1:
        return max_I_min_T
    return verdier
#f)

days = 300
verdier=pandemic(days,1000,1)

S_verdier = []
I_verdier = []
R_verdier = []
timesteps = np.arange(0,days,1)
for i in verdier:
    S_verdier.append(i[0])
for i in verdier:
    I_verdier.append(i[1])
for i in verdier:
    R_verdier.append(i[2])
plt.figure()
plt.title("Smittede")
plt.plot(timesteps,S_verdier)
plt.plot(timesteps,I_verdier)
plt.plot(timesteps,R_verdier)
plt.legend(["Susceptible","Infected","Recovered"])
plt.xlabel("Day")
plt.ylabel("Personer")
plt.show()

# #g)

verdier_minne_waste = pandemic(36500,1000,1)
print("State of Y_n for the last 10 days of the 100 year period")
for i in range(10):
    print(verdier_minne_waste[-1-i])
    
#h)
N_sim = 1000
max_min = pandemic(300,1000,N_sim)
total_max_I = 0
total_time_to_max_I = 0
for i in max_min:
    total_max_I += i[0]
    total_time_to_max_I += i[1]
mean_I = total_max_I/N_sim
mean_T = total_time_to_max_I/N_sim
print(f'\nForventet max infected fra {N_sim} simuleringer er: {mean_I} personer')
print(f'Forventet min tid til max infected fra {N_sim} simuleringer er: {mean_T} dager')
    