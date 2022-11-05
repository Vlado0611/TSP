import numpy as np 
import random
from numpy.core.numeric import cross
from numpy.lib.function_base import select
import matplotlib.pyplot as plt


distances = np.loadtxt("dantzig42_d.txt", dtype='i', delimiter=" ")
# matrica koja predstavlja udaljenosti
mutation_rate = 0.05
# stopa mutacije

#region Utility
def calculateTotalDistance(gene):
    cost = 0
    
    for i in range(len(gene)-1):
        i1 = gene[i] - 1
        i2 = gene[i+1] - 1
        
        cost += distances[int(i1)][int(i2)]

    return cost+distances[0][int(i2)]

def findMinFitness(population):
    minfitness = calculateTotalDistance(population[0])
    fittest = population[0]
    for pop in population:
        fitness = calculateTotalDistance(pop)
        if fitness < minfitness:
            minfitness = fitness
            fittest = pop

    return minfitness, fittest

def twoOptSwap(route, i, k):
    new_route = []
    new_route[0:i] = route[0:i]

    for j in range(i,k+1):
        new_route.append(route[k-j+i])

    new_route[k+1:] = route[k+1:]
    return new_route

def twoOptAlg(route):
    n = len(route)
    tempRoute = route
    for i in range(1,n-2):
        for j in range(i+2,n):
            d1 = calculateTotalDistance(tempRoute)
            d2 = calculateTotalDistance(twoOptSwap(tempRoute,i,j))

            if d1 > d2:
                tempRoute = twoOptSwap(tempRoute,i,j)

    return tempRoute

def fitnessFunc(population):
    fitness = []
    for i in range(len(population)):
        fitness.append(calculateTotalDistance(population[i]))

    return fitness

def selectionProbability(fitness):
    # s obzirom da najmanja vrijednost fitness funkcije znaci da je
    # jedinka najbolja, trazimo inverz i smijestamo ga u trueFitness
    # da bi mogli da iskoristimo Roulette Wheel Selection
    trueFitness = 1/fitness
    probability = []
    sum = np.sum(trueFitness)
    
    for i in range(len(trueFitness)):
        probability.append(trueFitness[i]/sum)

    return np.array(probability)

def fitnessMatrix(population):
    sorted = []
    for i in range(len(population)):
        fit = calculateTotalDistance(population[i])
        sort = list(population[i])
        sort.append(fit)
        sorted.append(sort)        


    sorted = np.array(sorted)
    sorted = sorted[np.argsort(sorted[:,len(population[1])])]
    sorted = np.delete(sorted, len(sorted[0])-1, axis=1)
    return sorted

def checkChild(child):
    check = set()
    for i in range(len(child)):
        if child[i]!=0:
            if child[i] not in check:
                check.add(child[i])
            elif child[i] in check:
                print("Greska je ", child)
                print("Element", child[i])

#endregion

#region Selections
def rouletteWheel(population, selectionProb, n):
    # sluzice da odabere koje jedinke ostaju, kao i da 
    # odredi roditelje sledece generacije
    selected = []

    for i in range(n):
        selected.append(population[np.random.choice(len(population),p=selectionProb)])

    return np.array(selected)

def selection(population, selectedPop=0):
    # funkcija koja bira 50% dobre populacije, 25% srednje ,15% lose populacije
    if selectedPop == 0:
        
        good_pop = np.zeros((int(len(population)*0.5), len(population[1])))
        average_pop = np.zeros((int(len(population)*0.25),len(population[1])))
        bad_pop = np.zeros((int(len(population)*0.15),len(population[1])))

        bad_pop[0:len(bad_pop)-1] = population[-len(bad_pop):-1]
        bad_pop[len(bad_pop)-1] = population[-1]
        pop_half = int(len(population)/2)
        average_pop[0:len(average_pop)] = population[pop_half-len(average_pop):pop_half]
        good_pop[0:len(good_pop)] = population[0:len(good_pop)] 

        parents = good_pop
        parents = np.append(parents, average_pop, axis=0)
        parents = np.append(parents, bad_pop, axis=0)
        
        return parents
    else:
        selected = np.zeros((selectedPop,len(population[1])))
        selected[0:len(selected)] = population[0:len(selected)]
        
        return selected
#endregion

#region Crossovers
def PMX(parent1, parent2, ind=0, firstCrossPoint=None, secondCrossPoint=None):
    # partially mapped crossover
    n = len(parent1)
    
    if not ind:
        firstCrossPoint = np.random.randint(1,n-2) # prva tacka podjele
        secondCrossPoint = np.random.randint(firstCrossPoint+1,n) # druga tacka podjele

    middleCross1 = parent1[firstCrossPoint:secondCrossPoint] # odsjeceni segment prvog
    middleCross2 = parent2[firstCrossPoint:secondCrossPoint] # odsjeceni segment drugog 

    child = np.zeros(len(parent1))

    child[firstCrossPoint:secondCrossPoint] = middleCross1 # u child stavimo odsjeceni segment prvog
    child[0] = 1 # svaka ruta pocinje jedinicom
    mapping = [[x,middleCross1[i]] for i, x in enumerate(middleCross2) if x not in middleCross1]
    mapping = np.array(mapping)
    
    # mapping mapira crossovere izmedju 2 parenta
    i = 0
    while i < len(mapping):
        # provjerimo mapiranje, stavljamo mapping[i][0] u child na poziciju na kojoj se nalazi mapping[i][1] u parentu 2
        # pod uslovom da je ta pozicija u childu prazna
        map, = np.where(parent2==mapping[i][1])
        # np.where vraca vektor [index, tip podatka], pa map[0] predstavlja indeks
        if(child[map[0]] == 0):
            # ako je pozicija u childu prazna, izvrsavamo
            child[map[0]] = mapping[i][0]
        else:
            # ako nije prazna, gledamo broj u parentu 2 na mapping[i][1] (tojest map[0]), i 
            # trazimo na koji broj u parentu 1 je on mapiran
            # i na mjesto tog broja u parentu 2 upisujemo mapping[i][0]
            new_map, = np.where(parent2==parent1[map[0]])
            while parent1[new_map[0]] in middleCross1:
                new_map, = np.where(parent2==parent1[new_map[0]])

            child[new_map[0]] = mapping[i][0]

        i += 1

    
    for i in range(len(parent2)):
        if child[i] == 0:
            child[i] = parent2[i]

    #checkChild(child)

    return child, firstCrossPoint, secondCrossPoint

def OX1(parent1, parent2):
    # nasumicno bira segmente iz oba parenta, stavlja ih u child nizove 
    # zatim popunjava redosljed childa 1 po relativnom redosljedu elemenata u parentu 2
    # a popunjava child 2 po relativnom redosljedu u parentu 1
    n = len(parent1)

    firstCrossPoint = np.random.randint(1,n-2) # prva tacka podjele
    secondCrossPoint = np.random.randint(firstCrossPoint+1,n) # druga tacka podjele

    child1 = np.zeros(len(parent1))
    child2 = np.zeros(len(parent2))

    child1[0] = 1
    child2[0] = 1
    child1[firstCrossPoint:secondCrossPoint] = parent1[firstCrossPoint:secondCrossPoint]
    child2[firstCrossPoint:secondCrossPoint] = parent2[firstCrossPoint:secondCrossPoint]

    ind = secondCrossPoint
    helpInd = ind

    while 0 in child1:
        if parent2[helpInd] not in child1:
            child1[ind] = parent2[helpInd]
            ind += 1
        
        helpInd += 1

        if ind == len(parent2):
            ind = 1

        if helpInd == len(parent2):
            helpInd = 1

    ind = secondCrossPoint
    helpInd = ind

    while 0 in child2:
        if parent1[helpInd] not in child2:
            child2[ind] = parent1[helpInd]
            ind += 1
        
        helpInd += 1
        
        if ind == len(parent1):
            ind = 1

        if helpInd == len(parent1):
            helpInd = 1

        
    return child1, child2

def OX2(parent1, parent2, indices=None):
    # nasumicno bira pozicije u nizu
    # zatim pronalazi pozicije tih elemenata u drugom nizu
    # na tim pozicijama kopira elemente u redu u kom su se pojavili u parentu 2
    # na ostalim pozicijama kopira elemente iz parenta 1
    
    if(indices==None):
        fixed = int(len(parent1)/2)
        appointed = set()
        indices = []
        i = 0

        while i < fixed:
            num = np.random.randint(1,len(parent1))
            if num not in appointed:
                appointed.add(num)
                indices.append(num)
                i += 1
    
    #print(indices)
    child1 = np.zeros(len(parent1))
    indices.sort()
    parentIndices = []
    for i in range(len(indices)):
        ind, = np.where(parent1==parent2[indices[i]])
        parentIndices.append(ind[0])
    #parentIndices.sort()
    
    j = 0
    for i in range(len(parent1)):
        if i not in parentIndices:
            child1[i] = parent1[i]
        else:
            child1[i] = parent2[indices[j]]
            j += 1
    
    return child1, indices

def POS(parent1, parent2, indices=None):
    # position based crossover
    # kao OX2, sem sto na pronadjenim pozicijama u parentu 1 stavlja parente na tim pozicijama iz parenta 2
    if(indices==None):
        fixed = int(len(parent1)/2)
        appointed = set()
        indices = []
        i = 0

        while i < fixed:
            num = np.random.randint(1,len(parent1))
            if num not in appointed:
                appointed.add(num)
                indices.append(num)
                i += 1
    
    #print(indices)
    child1 = np.zeros(len(parent1))
    child1[0] = 1
    indices.sort()
    
    for i in indices:
        child1[i] = parent2[i] 

    j, = np.where(child1==0)
    j = j[0]

    for i in range(len(parent1)):
        if parent1[i] not in child1:
            child1[j] = parent1[i] 
            j, = np.where(child1==0)
            if len(j)>0:
                j = j[0]

    return child1, indices
#endregion

#region Mutations
def EM(child):
    # exchange mutation
    # nasumicno bira 2 clana niza i mijenja im mjesta
    rand1 = np.random.randint(1,len(child)-1)

    rand2 = rand1
    while rand1 == rand2:
        rand2 = np.random.randint(1,len(child)-1)

    child[rand1], child[rand2] = child[rand2], child[rand1]

    return child   

def DM(child):
    # displacement mutation
    # nasumicno bira segment niza, zatim nasumicno bira indeks i stavlja cijeli segment na taj indeks
    n = len(child)
    firstCrossPoint = np.random.randint(1,n-2)
    secondCrossPoint = np.random.randint(firstCrossPoint+1,n)

    crossSection = child[firstCrossPoint:secondCrossPoint]
    tempChild = child[0:firstCrossPoint]
    tempChild = np.append(tempChild, child[secondCrossPoint:])
    
    randPoint = np.random.randint(1,len(tempChild))
    mutated = tempChild[0:randPoint]
    mutated = np.append(mutated, crossSection)
    mutated = np.append(mutated, tempChild[randPoint:])

    return mutated

def IM(child):
    # insertion mutation
    # nasumicno odabere jedan element niza, zatim nasumicno bira poziciju i stavlja element na tu poziciju
    randCity = np.random.randint(1,len(child))
    city = child[randCity]
    tempChild = child[0:randCity]
    tempChild = np.append(tempChild, child[randCity+1:])

    randPoint = np.random.randint(1,len(tempChild))
    mutated = tempChild[0:randPoint]
    mutated = np.append(mutated, city)
    mutated = np.append(mutated, tempChild[randPoint:])

    return mutated

def SIM(child):
    # simple inversion mutation
    # uzima nasumicno 2 tacke presjeka, zatim uzima dio niza izmedju te dvije tacke i obrce redosljed elemenata
    n = len(child)

    firstCrossPoint = np.random.randint(1,n-3) # prva tacka podjele
    secondCrossPoint = np.random.randint(firstCrossPoint+2,n) # druga tacka podjele

    crossSection = child[firstCrossPoint:secondCrossPoint]
    crossSection = crossSection[::-1]

    mutated = child[0:firstCrossPoint]
    mutated = np.append(mutated,crossSection)
    mutated = np.append(mutated,child[secondCrossPoint:])

    return mutated

def IVM(child):
    # inversion mutation
    # uzima segment niza, invertuje ga, i stavlja ga nasumicno na neko mjesto u nizu
    n = len(child)
    firstCrossPoint = np.random.randint(1,n-3)
    secondCrossPoint = np.random.randint(firstCrossPoint+2,n)

    crossSection = child[firstCrossPoint:secondCrossPoint]
    crossSection = crossSection[::-1]
    tempChild = child[0:firstCrossPoint]
    tempChild = np.append(tempChild, child[secondCrossPoint:])
    
    randPoint = np.random.randint(1,len(tempChild))
    mutated = tempChild[0:randPoint]
    mutated = np.append(mutated, crossSection)
    mutated = np.append(mutated, tempChild[randPoint:])

    return mutated
#endregion    

def geneticAlgorithm():

    iter = 0
    itermax = 501
    generations = []

    total_pop = 4*len(distances)  # ukupna populacija
    selected_pop = int(0.1*total_pop)  # selektovana populacija

    population = []

    for i in range(total_pop):
        population.append([1])
    # postavljamo prve clanove svakog obilaska na 1
    added = set(())

    i = 0
    while len(added) < total_pop:
        # dok nismo napravili populaciju
        for j in range(1, len(distances)):
            # popunjavamo ostatak niza sa ostalim gradovima
            while True:
                # dok ne nadjemo broj koji se vec ne nalazi u nizu
                r = random.randint(2, len(distances)) # generisemo nasumican broj r
                if population[i].count(r) == 0:
                    # ako se on ne nalazi u nizu, dodajemo ga
                    population[i].append(r)
                    break
        
        if tuple(population[i]) not in added:
            added.add(tuple(population[i]))
        else:
            # ako vec postoji ovakva jedinka, pravimo novu
            population[i] = [1]
            i -= 1

        i += 1

    population = np.array(population) # zapravo matrica iako se dobija array funkcijom

    
    for i in range(len(population)):
        # vrsimo optimizaciju preko 2-opt algoritma
        population[i] = twoOptAlg(population[i])
        
    minFit, fittest = findMinFitness(population)
    print("Generation: {0} The best fitness:{1}".format(iter,str(minFit)))

    for iter in range(1,itermax):

        '''
        # Roulette wheel selekcija
        fitness = np.array(fitnessFunc(population))
        selectionProb = selectionProbability(fitness)

        new_pop = rouletteWheel(population,selectionProb, selected_pop)

        parents = rouletteWheel(population,selectionProb, total_pop-selected_pop)
        '''
        sorted_pop = fitnessMatrix(population)
        new_pop = selection(sorted_pop,selected_pop)
        #new_pop[0] = twoOptAlg(new_pop[0])
        parents = selection(sorted_pop)

        while len(new_pop) < len(population):
            # nasumicno biramo 2 roditelja 
            p1Ind = np.random.randint(0,len(parents)-1)
            p2Ind = np.random.randint(0,len(parents)-1)
            parent1 = parents[p1Ind]
            parent2 = parents[p2Ind]
            
            
            '''PMX'''    
            #child1, firstCross, secondCross = PMX(parent1, parent2) # koristimo firstCross i secondCross da bi 2 childa sa 
            #child2,t1,t2 = PMX(parent2, parent1, 1, firstCross, secondCross) # jednako velikim crossoverom
            
            '''OX1'''
            #child1, child2 = OX1(parent1, parent2)
            '''OX2'''
            #child1, indices = OX2(parent1, parent2)
            #child2, indices = OX2(parent2, parent1, indices)

            '''POS'''
            child1, indices = POS(parent1, parent2)
            child2, indices = POS(parent2, parent1, indices)

            mutation = random.random()           
            if mutation < mutation_rate:
                child1 = IVM(child1)
            
            mutation = random.random()
            if mutation < mutation_rate:
                child2 = IVM(child2)

            new_pop = np.append(new_pop,[child1],axis=0)
            new_pop = np.append(new_pop,[child2],axis=0)

        population = new_pop
        
        if iter%100 == 0:
            for i in range(len(population)):
                population[i] = twoOptAlg(population[i])
        
        #minFit, fittest = findMinFitness(population)
        #generations.append(minFit)
        #print("Generation: {0}  The best fitness:{1}".format(iter,str(minFit)))
    
    minFit, fittest = findMinFitness(population)
    print("The best path is {0} with the length {1}".format(fittest,minFit))
    #plt.plot(range(itermax-1), generations)
    #plt.show()

geneticAlgorithm()
