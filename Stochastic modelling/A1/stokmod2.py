import numpy as np
import matplotlib.pyplot as plt


def poisson_p(lam,days,target):#returnerer sanns for at fra og med "target" hendelser intreffer
    p = 0
    for x in range(target):
        N_x=(lam*days)**x*np.exp(-lam*days)/np.math.factorial(x)
        p += N_x
    return 1-p

def insurance_claims_sim(lam,days,sim_number): #lam=intensity
    results = []
    for i in range(sim_number):  
        X = np.random.poisson(lam*days)
        results.append(X)
    return results

def insurance_sum_sim(lam,days,gamma,sim_number):
    money_list = []
    for j in range(sim_number):
        tot_money = 0
        N = np.random.poisson(lam*days)
        for i in range(N):
            money = np.random.exponential(1/gamma)#huske at gamma er rate og ikke scale
            tot_money += money
        money_list.append(tot_money)
    return money_list

def calc_mean_and_var(money_list):
    total_1 = 0
    total_2 = 0
    N=len(money_list)
    for i in money_list:
        total_1 += i
    mean = total_1/N
    for i in money_list:
        total_2 += (i-mean)**2
    variance = (total_2/(N+1))
    return mean,variance

#Spesifisering av parametere        
days = 59
lam = 1.5
gamma = 10
print("Teoretisk forventning: ")
print(poisson_p(1.5,59,100+1))#intensity,days,strengt mer enn 100 (aka 100+1)

realizations = insurance_claims_sim(1.5,59,1000)
count = 0
for i in realizations:
    if i>100:
        count+=1
print("Gjennomsnitt fra simulation: ")
print(count/float(len(realizations)))

x = [i for i in range(days)]
plt.figure()
plt.title("Insurance claims")
plt.legend(["S_n","I_n","R_n"])
plt.xlabel("Days")
plt.ylabel("Claims")
for i in range(10):
    y_i = insurance_claims_sim(lam,1,days)
    for j in range(len(y_i)):
        if j>0:
            y_i[j]+=y_i[j-1]
    plt.plot(x,y_i,'_',"hline")
plt.show()
print("Forventning: \t Varianse")
for i in range(10):
    mean,var = calc_mean_and_var(insurance_sum_sim(lam,days,gamma,1000))
    print(f'{mean:.3f} \t\t\t\t{var:.3f}')









