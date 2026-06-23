__author__ = "Winnie Davis"
__organization__ = "COSC343/AIML402, University of Otago"
__email__ = "davwi705@student.oatgo.ac.nz"
__noteToReader__ ="The unused code and comments were left in. I did consider removing them but I think it helps show the thought process" \
"                   I hope this is neat enough"

import numpy as np
import settings as settings

agentName = "<my_agent>"
#When playing against self, the opponent doesn't evolve - it's the copy of your population at the start of the training session.  So, if you want to train against a stronger self, you break down the training against self into multiple sessions.  For instance,
#trainingSchedule = [('self', 50), ('self', 50), ('self', 25)]
#trainingSchedule = [('self', 50), ('self', 50), ('self', 25)]
#trainingSchedule = [("self", 1), ("random", 1)] #cannot train over more than 500 games
#trainingSchedule = [('random', 50)] # r100 s50 s100
trainingSchedule = [('random', 75), ('self',50),('self',25),('self',50) ,('self',25)] # r100 s50 s100


# This is the class for your snake/agent
class Snake:

    

    def __init__(self, nPercepts, actions):
        '''
         You should initialise self.chromosome member variable here (whatever you choose it
         to be - a list/vector/matrix of numbers - and initialise it with some random
         values)
        '''

        # Percepts -> L1 -> L2 -> L3 -> Out
        self.nPercepts = nPercepts #INPUTS
        self.hiddenL1= 30# 30
        self.hiddenL2= 10 #10
        self.hiddenL3= 20 #20
        self.actions = actions # OUTPUTS
        self.mutationRate = 0.1

        self.chromosomeLength =  (self.nPercepts*self.hiddenL1 + self.hiddenL1 + self.hiddenL1*self.hiddenL2 + self.hiddenL2 + self.hiddenL2*self.hiddenL3 + self.hiddenL3 + self.hiddenL3*len(self.actions) + len(self.actions))
        '''
        # so there are 49 inputs (percepts) and there are 3 possible outcomes per one
        # = 147 possible options
        # we want one output (3 options (biases) not just yes no ) = 150?
        #WHhy did i pick -1 0 1 initialsiation (bc i didnt know what else to do )
        #Forget all of that!
        # Theres 49 things (percepts going in)
        # I want an inner layer NN before my output of 30 nodes
        # there are 3 end choices to choose from
        '''

        lowerw = -1
        upperw = 1

        lowerb = -.5
        upperb = 0.5
        #LAYER ONE
        layerOneW = np.random.uniform(lowerw, upperw , size =( self.nPercepts , self.hiddenL1)) #49,30
        layerOneB = np.random.uniform(lowerb, upperb , size = (self.hiddenL1)) #30

        #LAYER TWO
        layerTwoW = np.random.uniform(lowerw, upperw , size = (self.hiddenL1, self.hiddenL2)) #30,3
        layerTwoB = np.random.uniform(lowerb, upperb , size = (self.hiddenL2)) #3

        #LAYER Three
        layerThreeW = np.random.uniform(lowerw, upperw, size = (self.hiddenL2, self.hiddenL3)) #30,3
        layerThreeB = np.random.uniform(lowerb, upperb , size = (self.hiddenL3)) #3

        #LAYER Four
        layerFourW = np.random.uniform(lowerw, upperw, size = (self.hiddenL3, len(self.actions))) #30,3
        layerFourB = np.random.uniform(lowerb, upperb , size = (len(self.actions))) #3


        #COMBINE
        self.chromosome = np.concatenate([layerOneW.flatten(),
                                         layerOneB.flatten(), 
                                         layerTwoW.flatten(),
                                         layerTwoB.flatten(),
                                         layerThreeW.flatten(),
                                         layerThreeB.flatten(),
                                         layerFourW.flatten(),
                                         layerFourB.flatten()
        ])
        


    def AgentFunction(self, percepts):
        '''
         You should implement a neural network-based model here that translates from 'percepts' 
         to 'actions' with the weights and biases coming from 'self.chromosome', and the
         inputs to the network coming from the 'percepts' variable.
        
         Percepts are a 7x7 Numpy Matrix. 

         The return value must be an integer, a choice of one of possible actions 
         from [-1,0,1] corresponding to turning left, moving forward, and turning right.

         .
         .
         .
        ''' 

        
        #First Division Inputs
        inputs = percepts.flatten()
        #Second Division Chromosomes
        #apply the chromosome to the percepts somehow and the output will go into the final neuron
        #Get bits oout/unpack chromosome 
        W1Size = self.nPercepts * self.hiddenL1
        B1Size = self.hiddenL1
        W2Size = self.hiddenL1 * self.hiddenL2
        B2Size = self.hiddenL2
        W3Size = self.hiddenL2 * self.hiddenL3
        B3Size = self.hiddenL3
        W4Size = self.hiddenL3 * len(self.actions)
        B4Size = len(self.actions)
        
        

       # 0 ---> OneW ----> One B ----> Two W -----> ThreeW ---> Three B ----> Four W -->  fourB|||
        # Slices
        layerOneW = self.chromosome[0 : W1Size].reshape((self.nPercepts, self.hiddenL1))
        layerOneB = self.chromosome[W1Size : W1Size+B1Size]

        layerTwoW = self.chromosome[W1Size+B1Size : W1Size+B1Size+W2Size].reshape((self.hiddenL1, self.hiddenL2))
        layerTwoB = self.chromosome[W1Size+B1Size+W2Size: W1Size+B1Size+W2Size +B2Size]

        layerThreeW = self.chromosome[W1Size+B1Size+W2Size + B2Size: W1Size+B1Size+W2Size+B2Size + W3Size].reshape((self.hiddenL2, self.hiddenL3))
        layerThreeB = self.chromosome[ W1Size+B1Size+W2Size+B2Size + W3Size:  W1Size+B1Size+W2Size+B2Size + W3Size +B3Size]

        layerFourW = self.chromosome[ W1Size+B1Size+W2Size+B2Size + W3Size +B3Size :  W1Size+B1Size+W2Size+B2Size + W3Size +B3Size + W4Size].reshape((self.hiddenL3, len(self.actions)))
        layerFourB = self.chromosome[ W1Size+B1Size+W2Size+B2Size + W3Size +B3Size +W4Size :]


        # output = X @self.W + self.b?????
        #Final neuron 
    
        #if the total convergence from the second div is closest to -1 0 or 1 make that choice
        
        #Output2 should give 3 numbers [X, X, X] -1 0 1 whichever is the biggest is what we do?
        # maybe?

        #so its not just linear
        def relu(x):
            return np.maximum(0, x)
        
        #Percepts > Layer1
        inputs = percepts.flatten()
        output1 = relu( inputs @ layerOneW+ layerOneB)

        #Layer1 > Layer2
        output2 = relu(output1 @  layerTwoW + layerTwoB)

        #Layer2 > Layer3
        output3 = relu(output2 @  layerThreeW + layerThreeB)

        #Layer3 > actions
        output4 = output3 @  layerFourW + layerFourB

        mapToActionPos = [-1,0,1]
        chosenAction = mapToActionPos[np.argmax(output4)]

        return chosenAction

def evalFitness(population):

        N = len(population)

        # Fitness initialiser for all agents
        fitness = np.zeros((N))

        '''
        This loop iterates over your agents in the population - the purpose of this boiler plate
        code is to demonstrate how to fetch information from the population
        to score fitness of each agent
        '''
        for n, snake in enumerate(population):
            '''
            snake is an instance of Snake class that you implemented above, therefore you can access 
            any attributes (such as `self.chromosome').  Additionally, the object has the following
            attributes provided by the game engine; each a list of nTurns values
            
            snake.sizes - list of snake sizes over the game turns (0 means the snake is dead)
            snake.friend_attacks - turns when this snake has bitten another snake, not including
                                    head crashes - 0 not bitten in that turn, 1 bitten friendly snake 
            snake.enemy_attacks - turns when this snake has bitten another snake, not including
                                head crashes - 0 not bitten in that turn, 1 bitten enemy snake
            snake.bitten - number of bites received in a given turn (it's possible to be bitten by
                            several snakes in one turn)
            snake.foods - turns when food was eaten by the snake, not including biting other snake
                        (0 not eaten food, food eaten)
            snake.friend_crashes - turns when crashed heads with a friendly snake (0 no crash, 1 crash) 
            snake.enemy_crashes - turns when crashed heads with an enemy snake (0 no crash, 1 crash)
            '''
            meanSize = np.mean(snake.sizes)
            # The following two lines demonstrate how to 
            # extract other information from snake.sizes
            turnsAlive = np.sum(snake.sizes > 0)
            #maxTurns = len(snake.sizes)

            '''
            This fitness functions only considers the average snake size
                MY MASTER PLAN: avoid other snakes at all costs GET FOOD
            '''
            fitness[n] = 3* np.mean(snake.sizes) # 3/5
            fitness[n] += 2* turnsAlive #?
            #fitness[n] = 1 * maxTurns #?
            fitness[n] -=  5*sum(snake.friend_attacks) #-2/5
            fitness[n] += 2* sum(snake.enemy_attacks) # 2/5
            fitness[n] -=  3 *sum(snake.bitten) # -2/5
            fitness[n]  += 3* sum(snake.foods)# 3/5 ->4
            fitness[n] -=  5* sum(snake.friend_crashes) #-4/5
            fitness[n] -= 2*sum(snake.enemy_crashes) #2/5

        return fitness


def newGeneration(old_population):
    

        '''
        This function must return a tuple consisting of:
        - a list of the new_population of snakes that is of the same length as the old_population,
        - the average fitness of the old population
        '''
        N = len(old_population)
        nPercepts = old_population[0].nPercepts
        actions = old_population[0].actions
        fitness = evalFitness(old_population)

        # Create new population list...
        new_population = list()
        for n in range(N):
            # Create a new snake
            new_snake = Snake(nPercepts, actions)
            '''
            Here you should modify the new snakes chromosome by selecting two parents (based on their
            fitness) and crossing their chromosome to overwrite new_snake.chromosome

            Consider implementing elitism, mutation and various other
            strategies for producing a new creature.

            #Tournament selection - test this diffeernt params

            #cross

            #mutation - test this different params

            .
            .
            .
            '''
            selection = randSelection(old_population, (len(old_population)//4))
            finalSelection = tournamentSelection(selection)
            snakeParent1, snakeParent2 = finalSelection
            new_snake = crossOverRandPoint(snakeParent1, snakeParent2)
            new_snake = mutateSingleChromosome(new_snake, new_snake.mutationRate ,old_population[0].chromosomeLength)
            # Add the new snake to the new population
            new_population.append(new_snake)

        # At the end you need to compute the average fitness and return it along with your new population
        avg_fitness = np.mean(fitness)

        return (new_population, avg_fitness)


def randSelection(old_population, x):
    selection =  np.random.choice(old_population, size=x, replace=None)
    return selection

def tournamentSelection(selection):

    #make return array
    finalSelection = [None] * 2
    #get fittness of all selection
    # fittnessOfSelection = evalFitness(selection_list)#returns array

    # #"save"
    # highestFittness = fittnessOfSelection[0]
    # highestFitnessSnake = selection[0]

    count = 0
    selectionList = list(selection)
    fittnessOfSelection = evalFitness(selectionList)#returns array
    #pick the highest ones
    while(count<2):
            
            #"save"
            highestFittness = fittnessOfSelection[0]
            highestFitnessSnake = selectionList[0]
            for i, fitness in enumerate(fittnessOfSelection):

                if(fittnessOfSelection[i] > highestFittness):
                    highestFittness = fittnessOfSelection[i]
                    highestFitnessSnake = selectionList[i]

            finalSelection[count]= highestFitnessSnake
            count+=1
        

       # remove the snake from selection
       #reset highest fittness and highest fittness snake
            index = selectionList.index(highestFitnessSnake)
            selectionList.pop(index)
            fittnessOfSelection = np.delete(fittnessOfSelection, index)


    return finalSelection
        
def crossOverRandPoint(snakeParent1, snakeParent2):
   # crossOverPoint = np.random.randint(0,49)
    #newChromosome = np.concatenate([snakeParent1.chromosome[:crossOverPoint],  snakeParent2.chromosome[crossOverPoint:]])

   
   # new_snake.chromosome = newChromosome  # overwrite with crossed chromosome

    W1Size = snakeParent1.nPercepts * snakeParent1.hiddenL1
    B1Size = snakeParent1.hiddenL1
    W2Size = snakeParent1.hiddenL1 * snakeParent1.hiddenL2
    B2Size = snakeParent1.hiddenL2
    W3Size = snakeParent1.hiddenL2 * snakeParent1.hiddenL3
    B3Size = snakeParent1.hiddenL3
    W4Size = snakeParent1.hiddenL3 * len(snakeParent1.actions)
    B4Size = len(snakeParent1.actions)

    #within the 8 sections
    # crossOverPointW1 = np.random.randint(0,W1Size)
    # crossOverPointB1= np.random.randint(W1Size,W1Size +B1Size)
    # crossOverPointW2 = np.random.randint(W1Size +B1Size ,W1Size +B1Size +W2Size)
    # crossOverPointB2 =  np.random.randint(W1Size +B1Size + W2Size,W1Size +B1Size +W2Size +B2Size)
    # crossOverPointW3 = np.random.randint(W1Size +B1Size + W2Size +B2Size ,W1Size +B1Size +W2Size +B2Size +W3Size)
    # crossOverPointB3 = np.random.randint(W1Size +B1Size + W2Size +B2Size +W3Size ,W1Size +B1Size +W2Size +B2Size +W3Size + B3Size)
    # crossOverPointW4 =  np.random.randint(W1Size +B1Size + W2Size +B2Size +W3Size +B3Size ,W1Size +B1Size +W2Size +B2Size +W3Size + B3Size +W4Size)
    # crossOverPointB4 = np.random.randint(W1Size +B1Size + W2Size +B2Size +W3Size +B3Size +W4Size ,W1Size +B1Size +W2Size +B2Size +W3Size + B3Size +W4Size + B4Size)


    #apply the chromosome to the percepts somehow and the output will go into the final neuron
    #Get bits oout/unpack chromosome 
   

    P1layerOneW = snakeParent1.chromosome[0 : W1Size]
    P1layerOneB = snakeParent1.chromosome[W1Size : W1Size+B1Size]

    P1layerTwoW =  snakeParent1.chromosome[W1Size+B1Size : W1Size+B1Size+W2Size]
    P1layerTwoB =  snakeParent1.chromosome[W1Size+B1Size+W2Size: W1Size+B1Size+W2Size +B2Size]

    P1layerThreeW = snakeParent1.chromosome[W1Size+B1Size+W2Size + B2Size: W1Size+B1Size+W2Size+B2Size + W3Size]
    P1layerThreeB = snakeParent1.chromosome[ W1Size+B1Size+W2Size+B2Size + W3Size:  W1Size+B1Size+W2Size+B2Size + W3Size +B3Size]

    P1layerFourW = snakeParent1.chromosome[ W1Size+B1Size+W2Size+B2Size + W3Size +B3Size :  W1Size+B1Size+W2Size+B2Size + W3Size +B3Size + W4Size]
    P1layerFourB = snakeParent1.chromosome[ W1Size+B1Size+W2Size+B2Size + W3Size +B3Size +W4Size :]

    P2layerOneW = snakeParent2.chromosome[0 : W1Size]
    P2layerOneB = snakeParent2.chromosome[W1Size : W1Size+B1Size]

    P2layerTwoW =  snakeParent2.chromosome[W1Size+B1Size : W1Size+B1Size+W2Size]
    P2layerTwoB =  snakeParent2.chromosome[W1Size+B1Size+W2Size: W1Size+B1Size+W2Size +B2Size]

    P2layerThreeW = snakeParent2.chromosome[W1Size+B1Size+W2Size + B2Size: W1Size+B1Size+W2Size+B2Size + W3Size]
    P2layerThreeB = snakeParent2.chromosome[ W1Size+B1Size+W2Size+B2Size + W3Size:  W1Size+B1Size+W2Size+B2Size + W3Size +B3Size]

    P2layerFourW = snakeParent2.chromosome[ W1Size+B1Size+W2Size+B2Size + W3Size +B3Size :  W1Size+B1Size+W2Size+B2Size + W3Size +B3Size + W4Size]
    P2layerFourB = snakeParent2.chromosome[ W1Size+B1Size+W2Size+B2Size + W3Size +B3Size +W4Size :]
 
    # crossPoint1 = np.random.randint(0,W1Size)
    # crossPoint2 = np.random.randint(0,B1Size)
    # crossPoint3 = np.random.randint(0,W2Size)
    # crossPoint4 = np.random.randint(0,B2Size)
    # crossPoint5 = np.random.randint(0,W3Size)
    # crossPoint6 = np.random.randint(0,B3Size)
    # crossPoint7 = np.random.randint(0,B4Size)
    # crossPoint8 = np.random.randint(0,W4Size)

    # newSnakeW1 = np.concatenate([ P1layerOneW[0 : crossPoint1],   P2layerOneW[crossPoint1: ] ])
    # newSnakeB1 = np.concatenate([ P1layerOneB[0 : crossPoint2],   P2layerOneB[crossPoint2: ]])
    # newSnakeW2 = np.concatenate([ P1layerTwoW[0 : crossPoint3],   P2layerTwoW[crossPoint3: ]])
    # newSnakeB2 = np.concatenate([ P1layerTwoB[0 :crossPoint4],   P2layerTwoB[ crossPoint4: ]])
    # newSnakeW3= np.concatenate([ P1layerThreeW[0 :crossPoint5],   P2layerThreeW[crossPoint5: ]])
    # newSnakeB3= np.concatenate([ P1layerThreeB[0 : crossPoint6],   P2layerThreeB[crossPoint6: ]])
    # newSnakeW4= np.concatenate([ P1layerFourW[0 :crossPoint7],   P2layerFourW[ crossPoint7: ]])
    # newSnakeB4= np.concatenate([ P1layerFourB[0 : crossPoint8],   P2layerFourB[crossPoint8: ]])

    # new_snake.chromosome = np.concatenate([newSnakeW1,newSnakeB1,newSnakeW2,newSnakeB2,newSnakeW3,newSnakeB3,newSnakeW4,newSnakeB4])

    # sections = [
    #     (0, W1Size), (W1Size, W1Size+B1Size),
    #     (W1Size+B1Size, W1Size+B1Size+W2Size), (W1Size+B1Size+W2Size, W1Size+B1Size+W2Size+B2Size),
    #     (W1Size+B1Size+W2Size+B2Size, W1Size+B1Size+W2Size+B2Size+W3Size), 
    #     (W1Size+B1Size+W2Size+B2Size+W3Size, W1Size+B1Size+W2Size+B2Size+W3Size+B3Size),
    #     (W1Size+B1Size+W2Size+B2Size+W3Size+B3Size, W1Size+B1Size+W2Size+B2Size+W3Size+B3Size+W4Size),
    #     (W1Size+B1Size+W2Size+B2Size+W3Size+B3Size+W4Size, None) 
    # ]
    # sections = [
    #     (0, W1Size+B1Size),
    #     (W1Size+B1Size, W1Size+B1Size+W2Size+B2Size),
    #     (W1Size+B1Size+W2Size+B2Size, W1Size+B1Size+W2Size+B2Size+W3Size+B3Size),
    #     (W1Size+B1Size+W2Size+B2Size+W3Size+B3Size, None) 
    # ]

    #50%eletism
    sections = [
            (0,  W1Size+B1Size+W2Size+B2Size),
            (W1Size+B1Size+W2Size+B2Size, None) 
    ]

    new_chromosome = []

    for start, end in sections:

        if np.random.rand() < 0.5:
            new_chromosome.append(snakeParent1.chromosome[start:end])
        else:
            new_chromosome.append(snakeParent2.chromosome[start:end])

    new_snake = Snake(snakeParent1.nPercepts, snakeParent1.actions) 
    new_snake.chromosome = np.concatenate(new_chromosome)
    
    return new_snake

#mutate in two random places!

def mutateSingleChromosome(new_snake, mutationRate, chromLength):
    toMutateOrNotToMutate = np.random.rand(); # 0-1
    if toMutateOrNotToMutate <= mutationRate :
        #find a random spot 0-length of chromosome and then fill with -1 to 1

        # W1Size = new_snake.nPercepts * new_snake.hidden
        # B1Size = new_snake.hidden
        # W2Size = new_snake.hidden * len(new_snake.actions)
        # B2Size = len(new_snake.actions)

        # new_snake.chromosome[0 : W1Size ] = np.random.randint(-5, 6 )
        # new_snake.chromosome[W1Size : B1Size] = np.random.randint(-5, 6 )
        # new_snake.chromosome[W2Size : B2Size] = np.random.randint(-5, 6 )
        # new_snake.chromosome[B2Size: ] = np.random.randint(-5, 6 )

        #totalChromLength = 49*30 + 30*10 + 10*20 + 20*3 + 30 + 10 + 20 + 3  # set thsi preoperly maybe
  


        mutation_indices = np.random.choice(chromLength, 10, replace=False) #20 originally

        #index1, index2, index3, index4 = mutation_indices
            
        new_snake.chromosome[mutation_indices] = np.random.uniform(-1, 1, 10) # gonna try smaller values (-3,4) 206

        

    return new_snake
            
