import random
import numpy as numpy
import matplotlib.pyplot as plt
import Utils as Utils

eps = 10e-11

polynomialDegrees = [2, 3, 4, 5]
sectionForGaussian = [1, 5]
C = [0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
p = 2  # coef to polynomial kernel
b = 1  # coef to gaussian kernel

computedKernel = [[]]

currentDatasetNumber = 0
kernels = ["linear", "polynomial", "gaussian"]
listOfNames = ["Chips", "Geyser"]
listOfKernelResults = [{"linear": 200.0, "polynomial": 200.0, "gaussian": 200.0},
                       {"linear": 200.0, "polynomial": 200.0, "gaussian": 200.0}]
listOfBestKernelParams = [{"linear": 0, "polynomial": 200.0, "gaussian": 200.0},
                          {"linear": 0, "polynomial": 200.0, "gaussian": 200.0}]
listOfBestC = [{"linear": 0, "polynomial": 200.0, "gaussian": 200.0},
               {"linear": 0, "polynomial": 200.0, "gaussian": 200.0}]

listOfBestFreeCoefs = [{"linear": 0, "polynomial": 200.0, "gaussian": 200.0},
                       {"linear": 0, "polynomial": 200.0, "gaussian": 200.0}]


def computeKernel(name, x1, x2):
    if name == "linear":
        return sum([a[0] * a[1] for a in zip(x1, x2)])
    elif name == "polynomial":
        return (1 + sum([a[0] * a[1] for a in zip(x1, x2)])) ** p
    elif name == "gaussian":
        return numpy.exp(numpy.multiply(-b, sum([(a[0] - a[1]) ** 2 for a in zip(x1, x2)])))
    return 0


def predictKernel(name, features):
    global computedKernel
    for i in range(len(features)):
        for j in range(len(features)):
            computedKernel[i][j] = computeKernel(name, features[i], features[j])


def getClass(x):
    if x == 'P':
        return 1
    return -1


def computeE(coefs, i, classes):
    return sum([getClass(classes[j]) * coefs[j] * computedKernel[i][j] for j in
                range(0, len(classes))])


def computeError(coefs, classes, freeCoef):
    wrong = 0
    for i in range(len(classes)):
        predicted = numpy.sign(computeE(coefs, i, classes) + freeCoef)
        if predicted != getClass(classes[i]):
            wrong += 1
    return wrong * 100 / len(classes)


def computeSVM(coefs, kernel, c, classes):
    global computedKernel
    datasetLen = len(classes)
    tmpCoefs = [0 for _ in range(datasetLen)]
    shuffledArray = [i for i in range(0, datasetLen)]
    for _ in range(1000):
        random.shuffle(shuffledArray)
        for step in range(0, datasetLen):
            i = shuffledArray[step]
            j = random.randint(0, datasetLen - 1)
            while i == j:
                j = random.randint(0, datasetLen - 1)

            if classes[i] != classes[j]:
                L = max(0, tmpCoefs[j] - tmpCoefs[i])
                H = min(c, c + tmpCoefs[j] - tmpCoefs[i])
            else:
                L = max(0, tmpCoefs[j] + tmpCoefs[i] - c)
                H = min(c, tmpCoefs[j] + tmpCoefs[i])

            if abs(L - H) > eps:
                eta = computedKernel[i][i] + computedKernel[j][j] - 2 * computedKernel[i][j]
                if eta > eps:
                    E1 = computeE(tmpCoefs, i, classes) - getClass(classes[i])
                    E2 = computeE(tmpCoefs, j, classes) - getClass(classes[j])
                    newCoef = tmpCoefs[j] + getClass(classes[j]) * (E1 - E2) / eta
                    newCoef = min(H, max(L, newCoef))
                    if abs(newCoef - tmpCoefs[j]) < eps:
                        continue
                    tmpCoefs[i] += getClass(classes[i]) * getClass(classes[j]) * (tmpCoefs[j] - newCoef)
                    tmpCoefs[j] = newCoef

    pos = 0
    while pos < datasetLen and (tmpCoefs[pos] < eps or tmpCoefs[pos] > c - eps):
        pos += 1
    if pos == datasetLen:
        res = 0
        cnt = 0
        for i in range(datasetLen):
            if tmpCoefs[i] >= eps:
                for j in range(datasetLen):
                    res -= tmpCoefs[j] * getClass(classes[j]) * computedKernel[i][j]
                res += getClass(classes[i])
                cnt += 1
        if cnt == 0:
            tmpFreeCoef = 0
        else:
            tmpFreeCoef = res / cnt
    else:
        res = 0
        for i in range(datasetLen):
            res -= tmpCoefs[i] * getClass(classes[i]) * computedKernel[pos][i]
        res += getClass(classes[pos])
        tmpFreeCoef = res

    err = computeError(tmpCoefs, classes, tmpFreeCoef)
    if err < listOfKernelResults[currentDatasetNumber][kernel]:
        listOfKernelResults[currentDatasetNumber][kernel] = err
        param = -1
        if kernel == "polynomial":
            param = p
        elif kernel == "gaussian":
            param = b
        listOfBestKernelParams[currentDatasetNumber][kernel] = param
        listOfBestC[currentDatasetNumber][kernel] = c
        listOfBestFreeCoefs[currentDatasetNumber][kernel] = tmpFreeCoef
        coefs[:] = tmpCoefs.copy()


def drawGraph(coefs, kernel, features, classes):
    size = 30
    backgroundSize = size * 10
    backgroundPX = []
    backgroundPY = []
    backgroundNX = []
    backgroundNY = []

    y, step = -2 if currentDatasetNumber == 0 else 0, 0.05
    yCond = 1.5 if currentDatasetNumber == 0 else 7
    xCond = 2 if currentDatasetNumber == 0 else 24
    while y < yCond:
        x = -1.5 if currentDatasetNumber == 0 else 0
        while x < xCond:
            res = listOfBestFreeCoefs[currentDatasetNumber][kernel]
            for i in range(len(classes)):
                res += getClass(classes[i]) * coefs[i] * computeKernel(kernel, [x, y], features[i])

            if res > eps:
                backgroundPX.append(x)
                backgroundPY.append(y)
            elif res < -eps:
                backgroundNX.append(x)
                backgroundNY.append(y)
            x += step
        y += step

    print(len(backgroundPX), len(backgroundPY), len(backgroundNX), len(backgroundNY))
    fig, ax = plt.subplots()

    ax.scatter(backgroundPX, backgroundPY, marker='s', c=[[0 / 255.0, 75 / 255.0, 75 / 255.0]], s=backgroundSize,
               label="P class area")
    ax.scatter(backgroundNX, backgroundNY, marker='s', c=[[1, 1, 0]], s=backgroundSize, label="N class area")

    firstClassX = [features[i][0] for i in range(len(classes)) if getClass(classes[i]) == 1]
    firstClassY = [features[i][1] for i in range(len(classes)) if getClass(classes[i]) == 1]
    secondClassX = [features[i][0] for i in range(len(classes)) if getClass(classes[i]) == -1]
    secondClassY = [features[i][1] for i in range(len(classes)) if getClass(classes[i]) == -1]

    ax.scatter(firstClassX, firstClassY, c=[[0.1, 0.63, 0.55]], s=size, label="P class point")
    ax.scatter(secondClassX, secondClassY, c='r', s=size, label="N class point")

    fig.set_figwidth(14)
    fig.set_figheight(14)
    ax.legend(
        fontsize=20,
        title=listOfNames[currentDatasetNumber] + " " + kernel,
        title_fontsize=25
    )
    plt.show()
    fig.savefig(listOfNames[currentDatasetNumber] + " " + kernel)


def solver(path):
    global p, b, computedKernel
    [features, classes] = Utils.getDataAndClasses(path)

    computedKernel = [[0] * len(classes) for _ in range(len(classes))]
    features = features.values.tolist()
    classes = classes.values.tolist()
    forLinear = []  # коэфициеты для линейного ядра
    forPolynomial = []  # коэфициенты для полиномиального ядра
    forGaussian = []  # коэфициенты для гауссового ядра

    for c in C:
        for kernel in kernels:
            if kernel == "linear":
                predictKernel(kernel, features)
                computeSVM(forLinear, kernel, c, classes)
            elif kernel == "polynomial":
                for i in polynomialDegrees:
                    p = i
                    predictKernel(kernel, features)
                    computeSVM(forPolynomial, kernel, c, classes)
            elif kernel == "gaussian":
                step = 0.1
                while b <= sectionForGaussian[1]:
                    predictKernel(kernel, features)
                    computeSVM(forGaussian, kernel, c, classes)
                    b += step
                b = sectionForGaussian[0]

    print("Done")
    f = open(listOfNames[currentDatasetNumber] + ".txt", "w")
    f.write("--" + listOfNames[currentDatasetNumber] + "--\n")
    f.write("Coefs:\n")
    f.write(str(forLinear) + "\n")
    f.write(str(forPolynomial) + "\n")
    f.write(str(forGaussian) + "\n")
    f.write("Best free coefs: \n")
    Utils.printDictionary(listOfBestFreeCoefs, currentDatasetNumber, f)
    f.write("Kernel errors:\n")
    Utils.printDictionary(listOfKernelResults, currentDatasetNumber, f)
    f.write("Kernels param:\n")
    Utils.printDictionary(listOfBestKernelParams, currentDatasetNumber, f)
    f.write("Best C param:\n")
    Utils.printDictionary(listOfBestC, currentDatasetNumber, f)

    f.close()
    drawGraph(forLinear, "linear", features, classes)
    drawGraph(forPolynomial, "polynomial", features, classes)
    drawGraph(forGaussian, "gaussian", features, classes)


def reeinitParameters():
    global p, b, computedKernel
    p = 2
    b = 1
    computedKernel.clear()


solver("datasets/chips.csv")
currentDatasetNumber += 1
reeinitParameters()
solver("datasets/geyser.csv")
