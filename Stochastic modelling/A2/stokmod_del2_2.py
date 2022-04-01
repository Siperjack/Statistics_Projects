import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

observed_values = np.array([[0.30, 0.5],[0.35, 0.32],[0.39, 0.40],[0.41, 0.35],[0.45, 0.60]])
X_B = np.zeros([len(observed_values),1])
for i in range(len(observed_values)):
    X_B[i] = observed_values[i][1]#lagrer de realiserte verdiene i en vektor til senere uten første index
n=51
theta_grid = np.linspace(0.25,0.5,51)#LAgrer data oppgitt i oppgaven
x = np.array([0,0,0,0,0])
E_theta = 0.5
Var_theta = 0.5**2
mu_vec = np.full([n,1],E_theta)
Sigma = np.zeros([n,n],dtype = 'float') #Denne matrisen og den over inneholder all informasjon jeg trenger for denne oppgaven
# Sigma_B = np.zeros([len(observed_values),len(observed_values)])
def corr(theta_i,theta_j):
    return (1+15*abs(theta_i-theta_j))*np.exp(-15*abs(theta_i-theta_j))*Var_theta
def construct_cond_cov_matrix(vec_A,vec_B):#skal ta inn indekser til theta som er med i A og hvilke som er med i B
    Sigma_AA = np.delete(np.delete(Sigma,vec_B,0),vec_B,1)#Hvor A er alle theta-verdiene vi ikke har data på
    Sigma_AB = np.delete(np.delete(Sigma,vec_B,0),vec_A,1)
    Sigma_BA = np.delete(np.delete(Sigma,vec_A,0),vec_B,1)
    Sigma_BB = np.delete(np.delete(Sigma,vec_A,0),vec_A,1)
    return [[Sigma_AA,Sigma_AB],[Sigma_BA,np.linalg.inv(Sigma_BB)]] #NB: Fra og med her finner jeg ikke noen feil til oppgave 1a,
#så det har gjort meg usikker på denne funksjonen.

def prediction_intervall(Var_vec,alpha=0.9): #Kunne gjor mer generell, men siden vi skal ha alpha 90% så lagde jeg bare denne med
    if alpha == 0.9:#data fra blå bok.
        z = 1.645
    return z * np.sqrt(Var_vec)
    

for i in range(n):#Konstruerer kovariansmatrisen til den totale ubettingede distribusjonen bassert på korrelasjonsfunksjonen
    for j in range(n):
        if i==j:
            Sigma[i][j] = Var_theta
        else:
            Sigma[i][j] = corr(theta_grid[i],theta_grid[j])*Var_theta

indexes_A = []#Finner indexene til elementene som skal til blokk A og blokk B i den betingede sannsynligheten
indexes_B = []#Finner indexene til elementene som skal til blokk A og blokk B i den betingede sannsynligheten
theta_vec_A = []#Finner indexene til elementene som skal til blokk A og blokk B i den betingede sannsynligheten
mu_A = np.full([n-5,1],0.5)#Finner indexene til elementene som skal til blokk A og blokk B i den betingede sannsynligheten
mu_B = np.full([5,1],0.5)#Finner indexene til elementene som skal til blokk A og blokk B i den betingede sannsynligheten
for i in range(n): #Finner indexene til elementene som skal til blokk A og blokk B i den betingede sannsynligheten
    is_in = False
    for j in range(len(observed_values)):
        if observed_values[j][0] == round(theta_grid[i],3): #ungår avrudingsfeil fra linspace
            is_in = True
    if is_in:
        indexes_B.append(i)
    else:
        indexes_A.append(i)
        theta_vec_A.append(theta_grid[i])
        
Sigma_cond = construct_cond_cov_matrix(indexes_A,indexes_B) #Er denne riktig burde vi fullstendig kunne beskrive den betingede
error_from_mu = X_B-mu_B# sannsynligheten med formlene under med dataen fra denne matrisen
mu_C =mu_A + Sigma_cond[0][1]@Sigma_cond[1][1]@error_from_mu #Finner den betingede forventningsvektoren
Sigma_C =Sigma_cond[0][0]-Sigma_cond[0][1]@Sigma_cond[1][1]@Sigma_cond[1][0] #Finner den betingede covariansmatrisen

mu_tot = mu_C
Var_vec =np.diag(Sigma_C)
for i in range(len(observed_values)): #Konstruerer E[f(theta)] og Var[f(theta)] alle theta i theta_grid, inkludert de realiserte verdiene
    mu_tot = np.insert(mu_tot,indexes_B[i],X_B[i])
    Var_vec = np.insert(Var_vec,indexes_B[i],0)
    
    

X_upper = mu_tot + prediction_intervall(Var_vec,0.9)#Bare legger til det 90% øvre prediksjonsintervallet
X_lower = mu_tot - prediction_intervall(Var_vec,0.9) #Symmetri i den gaussiske fordelingen tillater å bare trekke fra for nedre del.
plt.figure()
plt.title("Betinget distribusjon med hensyn til evaluerte verdier")
plt.xlabel("theta")
plt.ylabel("y(theta)")
plt.plot(theta_grid,mu_tot,"r")
plt.plot(theta_grid,X_upper,"g")
plt.plot(theta_grid,X_lower,"g")
# plt.ylim(0.2,1.2)
plt.show() #Blir veldig hakkete av ukjent grunn, kom aldri til bunns i det. Vennligst gi tilbakemedling om dere ser det.

"""Oppgave2b)"""

# def standardise(mu_vec,Sigma_matrix):
target = np.full([n-5,1],0.3)
L = np.linalg.cholesky(Sigma_C) #Chovelsy faktor som skal brukes til standardisering
L_inv = np.linalg.inv(L)
Z = norm.cdf(L_inv@(target-mu_C))#Standardiserer og setter inn i denne funksjonen, ut får vi en vektor med alle sannsynlighetene for 
#at theta får en score under 0.3

plt.figure()
plt.title("Sannsynlighet for en score under 0.3")
plt.xlabel("theta")
plt.ylabel("P(y(theta)<0.3)")
plt.plot(theta_vec_A,Z,"r")
plt.show() #Blir ikke riktig da den er bassert på forrige oppgave sine resultater som ikke virker riktig. Merk at de realiserte verdiene
#ikke er plottet inn med sannsynlighet 0. Dette har de da de er observert til å være over 0.3, og de er da ikke under 0.3

"""2c"""

def add_point(observed_list, point):
    for i in range(len(observed_list)):
        if point[0] < observed_list[i][0]:
            observed_list_new = np.insert(observed_list,i,point,axis=0)
    return observed_list_new

observed_values=add_point(observed_values,np.array([0.33,0.40]))
k = len(observed_values)
X_B1 = np.zeros([k,1])


for i in range(len(observed_values)):
    X_B1[i] = observed_values[i][1]#lagrer de realiserte verdiene i en vektor til senere uten første index

indexes_A1 = []#Finner indexene til elementene som skal til blokk A og blokk B i den betingede sannsynligheten
indexes_B1 = []#Finner indexene til elementene som skal til blokk A og blokk B i den betingede sannsynligheten
theta_vec_A1 = []#Finner indexene til elementene som skal til blokk A og blokk B i den betingede sannsynligheten
mu_A1 = np.full([n-k,1],0.5)#Finner indexene til elementene som skal til blokk A og blokk B i den betingede sannsynligheten
mu_B1 = np.full([k,1],0.5)#Finner indexene til elementene som skal til blokk A og blokk B i den betingede sannsynligheten
for i in range(n): #Finner indexene til elementene som skal til blokk A og blokk B i den betingede sannsynligheten
    is_in = False
    for j in range(k):
        if observed_values[j][0] == round(theta_grid[i],3): #ungår avrudingsfeil fra linspace
            is_in = True
    if is_in:
        indexes_B1.append(i)
    else:
        indexes_A1.append(i)
        theta_vec_A1.append(theta_grid[i])
        
Sigma_cond1 = construct_cond_cov_matrix(indexes_A1,indexes_B1) #Er denne riktig burde vi fullstendig kunne beskrive den betingede
error_from_mu1 = X_B1-mu_B1# sannsynligheten med formlene under med dataen fra denne matrisen
mu_C1 = mu_A1 + Sigma_cond1[0][1]@Sigma_cond1[1][1]@error_from_mu1 #Finner den betingede forventningsvektoren
Sigma_C1 =Sigma_cond1[0][0]-Sigma_cond1[0][1]@Sigma_cond1[1][1]@Sigma_cond1[1][0] #Finner den betingede covariansmatrisen

mu_tot1 = mu_C1
Var_vec1=np.diag(Sigma_C1)
for i in range(len(observed_values)): #Konstruerer E[f(theta)] og Var[f(theta)] alle theta i theta_grid, inkludert de realiserte verdiene
    mu_tot1 = np.insert(mu_tot1,indexes_B1[i],observed_values[i][1])#noe her
    Var_vec1 = np.insert(Var_vec1,indexes_B1[i],0)
    
X_upper1 = mu_tot1 + prediction_intervall(Var_vec1,0.9)#Bare legger til det 90% øvre prediksjonsintervallet
X_lower1 = mu_tot1 - prediction_intervall(Var_vec1,0.9) #Symmetri i den gaussiske fordelingen tillater å bare trekke fra for nedre del.
plt.figure()
plt.title("Betinget distribusjon med hensyn til evaluerte verdier")
plt.xlabel("theta")
plt.ylabel("y(theta)")
plt.plot(theta_grid,mu_tot1,"r")
plt.plot(theta_grid,X_upper1,"g")
plt.plot(theta_grid,X_lower1,"g")
# plt.ylim(0.2,1.2)
plt.show() #Blir veldig hakkete av ukjent grunn, kom aldri til bunns i det. Vennligst gi tilbakemedling om dere ser det.


target1 = np.full([n-k,1],0.3)
L1 = np.linalg.cholesky(Sigma_C1) #Chovelsy faktor som skal brukes til standardisering
L_inv1 = np.linalg.inv(L1)
Z1= norm.cdf(L_inv1@(target1-mu_C1))#Standardiserer og setter inn i denne funksjonen, ut får vi en vektor med alle sannsynlighetene for 
#at theta får en score under 0.3

plt.figure()
plt.title("Sannsynlighet for en score under 0.3")
plt.xlabel("theta")
plt.ylabel("P(y(theta)<0.3)")
plt.plot(theta_vec_A1,Z1,"r")
plt.show() #Blir ikke riktig da den er bassert på forrige oppgave sine resultater som ikke virker riktig. Merk at de realiserte verdiene
#ikke er plottet inn med sannsynlighet 0. Dette har de da de er observert til å være over 0.3, og de er da ikke under 0.3

suggestion = [0,0]
for i in range(len(Z1)):
    if suggestion[0]<Z1[i]:
        suggestion = [Z1[i],theta_vec_A1[i]]
print(f"Vi vil anbefale forskerene å velge theta lik: {suggestion[1]}")
