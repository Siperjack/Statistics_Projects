import numpy as np
import matplotlib.pyplot as plt

"""1a) Hvorfor Markov? I hver instans er har du med en viss sansylighet sjansen for å forbli i samme state eller endre state,
og disse sannsynlighetene avhenger kun av hvilke state du er i. Disse sannsynlighetene er altså stasjonære i tid. Dette 
ser vi igjen ved at vi har fått oppgitt at soujurn time i hver state er exponensialfordelt, som nettop har denne egenskapen
at den er memoryless."""


# P01 = (1 - alpha)*np.exp(1/my_S)
# P02 = (alpha)*np.exp(1/my_S)
# P10 = 1*np.exp(1/my_L)
# P20 = 1*np.exp(1/my_H) ê^-landa

S,L,H = 0,1,2 #susceptible,lightly infected,infected heavely
alpha = 0.1
my_S,my_L,my_H = 100,7,20#1/days
    
def epidemic_sim(t,N):
    parameters = [my_S,my_L,my_H]
    results_list = []#liste der alle partallsindekser er en state, og oddetallsindekser er dager i staten i elementet før
    for i in range(N):
        state = 1
        T = 0
        results = []
        while(T<t):        
            rand = np.random.uniform(0,1)
            days = np.random.exponential(parameters[state])#Var[days] = E[days]*2, så at vi får mye var burde forventes
            results.append(state)#legger til state
            results.append(days)#legger til dager
            if state == S:
                if rand<alpha:
                    state = H#frisk til veldig syk
                else: 
                    state = L
            else:
                state = S
            T += days
        results[-1] -= T%t #sørger for at tid etter simulasjonen er ferdig ikke legges med
        results_list.append(results)
    return results_list
    

""" Oppgave 1c """
result_list_1 = epidemic_sim(5*365,1)
days_list_1 = result_list_1[0][1::2]#plukker ut annehvert element i listen, altså de som representerer dager
states_list_1 = result_list_1[0][0::2]#plukker ut alle states

for i in range(1,len(days_list_1)):
    days_list_1[i]+=days_list_1[i-1] #lager akkumulativ days tabell
    
plt.figure()
plt.title("Simulation of the common cold over 5 years")
plt.xlabel("days")
plt.ylabel("state")
plt.ylim(-0.1, 2.1)
for i in range(1,len(days_list_1)-1):
    plt.plot(days_list_1[i:i+2:1],[states_list_1[i-1],states_list_1[i-1]],"b",linewidth=0.8)
plt.show()

""" Oppgave 1d """

result_list_2 = epidemic_sim(1000*365,1)
days_list_2 = result_list_2[0][1::2]#plukker ut annehvert element i listen, altså de som representerer dager
states_list_2 = result_list_2[0][0::2]#plukker ut alle states

tot_days = np.array([0,0,0]) #[0]=Susceptible,[1]=Lightly sick, [2]=Highly sick
analytisk = np.array([0.5*100,0.45*7,0.05*20])/sum([0.5*100,0.45*7,0.05*20]) #stemmer ikke helt, må sjekke mot marius sitt svar

for i in range(len(days_list_2)):#itererer gjennom resultatene og legger til dager i hver tilstand i til tot_days[i]
    tot_days[states_list_2[i]] += days_list_2[i]
print(f'Susceptable | Lightly sick | Highly sick \n {(tot_days/sum(tot_days))[0]:.2f}\t{(tot_days/sum(tot_days))[1]:.2f}\t{(tot_days/sum(tot_days))[2]:.2f}\n {analytisk[0]:.2f}\t{analytisk[1]:.2f}\t{analytisk[2]:.2f}') #Normaliserer med days in state/tot days

""" Oppgave 1e """

result_list_3 = epidemic_sim(1000*365,1)
days_list_3 = result_list_2[0][1::2]#plukker ut annehvert element i listen, altså de som representerer dager
states_list_3 = result_list_2[0][0::2]#plukker ut alle states

days_between_H = []
count = 0
for i in range(len(days_list_3)-1):#itererer gjennom resultatene og legger til dager i hver tilstand i til tot_days[i]       
    if states_list_3[i] == 2: #Begynner og telle etter første gang vi er veldig syke
        count += 1 #brukes til å legge til en ny ventetid hver gang vi er veldig syke
        days_between_H.append(0) #lager nytt element hvor vi legger til dager
    if count > 0 and states_list_3[i+1]!=2:#Før vi har hvert veldig syke en gang er det ikke noe interessante ting å tracke.
                                            #legger til dager tilbrakt i neste state så lenge den ikke er H.
        days_between_H[count-1] += days_list_3[i+1]

v_S_firststep = 10*(my_S+my_L*(1-alpha))        
print(f'Estimert tid mellom to større forskjølelser fra: \nSimulasjon på 10000 dager: {sum(days_between_H)/len(days_between_H):.2f}\nFirst step analasys: {v_S_firststep:.2f}')     

    
    
    
    
    
    
    
    
    
    
    