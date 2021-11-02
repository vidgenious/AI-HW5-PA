import random
import math
import numpy as np

def main():
    #weights will change once an epoch
    weightsL1, weightsL2 = initalizeWeights()

    #the input set used changes during an epoch
    bias = [1,1,1,1,1,1,1,1,1,1]
    aInputs = [0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]
    bInputs = [0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1]
    cInputs = [0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1]
    dInputs = [0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]
    expectedOutputs = [0,1,0,1,0,1,0,1,1,1,1,1,0,0,0,1]

    epochErrsAvg = 1
    while(epochErrsAvg >= 0.05):
        epochErrs = []

        selection = random.sample(range(0,16), 10)
        for currentSet in selection:
            inputL1 = []
            inputL1.append(1)
            inputL1.append(aInputs[currentSet])
            inputL1.append(bInputs[currentSet])
            inputL1.append(cInputs[currentSet])
            inputL1.append(dInputs[currentSet])

            result, inputL2 = calculateResult(inputL1, weightsL1, weightsL2)

            error = expectedOutputs[currentSet] - result
            epochErrs.append(error)

            weightsL1, weightsL2 = backpropogate(inputL1, inputL2, weightsL1, weightsL2, error, result)

        epochErrsAvg = np.mean(epochErrs)
        print(epochErrsAvg)



    print("looks good bucko!")

def calculateResult(inputL1, weightsL1, weightsL2):
    weightsL1 = np.array(weightsL1)
    weightsL2 = np.array(weightsL2)
    inputL1 = np.array(inputL1)

    product = weightsL1.dot(inputL1)

    inputL2 = []
    inputL2.append(1)
    for i in range(len(product)):
        inputL2.append(1/(1 + math.e ** (-product[i])))

    product = weightsL2.dot(inputL2)

    return (1/(1 + math.e ** (-product))), inputL2

def initalizeWeights():
    initialWeights = []
    temp = []
    for i in range(8):
        for j in range(5):
            temp.append((random.uniform(-1.0,1.0)))
        initialWeights.append(temp)
        temp = []

    for i in range(9):
        temp.append((random.uniform(-1.0, 1.0)))


    return initialWeights, temp

def backpropogate(inputL1, inputL2, weightsL1, weightsL2, finalError, result):
    i = 0
    finalSlope = (result * (1 - result))
    alpha = 0.1
    for weight in weightsL2:
        weight = weight + alpha * finalError * finalSlope * inputL2[i]
        i += 1

    errorTerm = finalError * finalSlope

    hiddenErrors = []
    for weight in weightsL2:
        hiddenErrors.append(weight * errorTerm)

    i = 0
    j = 0
    slope = 0
    for perceptron in weightsL1:
        slope = (inputL2[i] * (1 - inputL2[i]))
        for weight in perceptron:
            weight = weight + alpha * hiddenErrors[i] * slope * inputL1[j]
            j + 1
        i += 1
        j = 0

    return weightsL1, weightsL2

    


if __name__ == "__main__":
    main()