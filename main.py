import random
import numpy as numpy
import matplotlib.pyplot as plt
import Utils as Utils

eps = 0.0000000001

polynomialDegrees = [2, 3, 4, 5]
sectionForGaussian = [1, 5]
kernels = ["linear", "polynomial", "gaussian"]
C = [0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
p = 2  # coef to polynomial kernel
b = 1  # coef to gaussian kernel
freeCoef = 0
currentDatasetNumber = 0
listOfNames = ["Chips", "Geyser"]
listOfKernelResults = [{"linear": 200.0, "polynomial": 200.0, "gaussian": 200.0},
                       {"linear": 200.0, "polynomial": 200.0, "gaussian": 200.0}]
listOfBestKernelParams = [{"linear": 0, "polynomial": 200.0, "gaussian": 200.0},
                          {"linear": 0, "polynomial": 200.0, "gaussian": 200.0}]
listOfBestC = [1000, 1000]


def computeKernel(name, x1, x2):
    if name == "linear":
        return numpy.dot(x1, x2)
    elif name == "polynomial":
        return (1 + numpy.dot(x1, x2)) ** p
    elif name == "gaussian":
        return numpy.exp(numpy.multiply(b, numpy.power(numpy.linalg.norm(numpy.array(x1) - numpy.array(x2)), 2)))
    return 0


def getClass(x):
    if x == 'P':
        return 1
    return -1


def computeE(coefs, i, kernel, features, classes):
    return sum([getClass(classes[j]) * coefs[j] * computeKernel(kernel, features[j], features[i]) for j in
                range(0, len(classes))]) - getClass(classes[i])


def computeError(coefs, kernel, features, classes):
    wrong = 0
    for i in range(len(classes)):
        predicted = numpy.sign(computeE(coefs, i, kernel, features, classes) + freeCoef)
        if predicted != getClass(classes[i]):
            wrong += 1
    return wrong * 100 / len(classes)


def computeSVM(coefs, kernel, c, features, classes):
    global freeCoef
    datasetLen = len(classes)
    tmpCoefs = [0 for i in range(datasetLen)]
    for iter in range(1):
        shuffledArray = [i for i in range(0, datasetLen)]
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

            if abs(L - H) >= eps:
                eta = computeKernel(kernel, features[i], features[i]) + \
                      computeKernel(kernel, features[j], features[j]) - 2 * computeKernel(kernel, features[i],
                                                                                          features[j])
                if eta > eps:
                    newCoef = tmpCoefs[j] + getClass(classes[j]) * \
                              (computeE(tmpCoefs, i, kernel, features, classes) - computeE(tmpCoefs, j, kernel,
                                                                                           features, classes)) / eta
                    newCoef = min(H, max(L, newCoef))
                    if abs(newCoef - tmpCoefs[j]) < eps:
                        continue
                    tmpCoefs[i] += getClass(i) * getClass(j) * (tmpCoefs[j] - newCoef)
                    tmpCoefs[j] = newCoef
    pos = 0
    while pos < datasetLen and (tmpCoefs[pos] < eps or tmpCoefs[pos] > c - eps):
        pos += 1
    if pos == datasetLen:
        res = 0
        cnt = 0
        for i in range(datasetLen):
            for j in range(datasetLen):
                res -= tmpCoefs[j] * getClass(classes[j]) * computeKernel(kernel, features[i], features[j])
            res += getClass(classes[i])
            cnt += 1

        if cnt == 0:
            freeCoef = 0
        else:
            freeCoef = res / cnt
    else:
        res = 0
        for i in range(datasetLen):
            res -= tmpCoefs[i] * getClass(classes[i]) * computeKernel(kernel, features[pos], features[i])
        res += getClass(classes[pos])
        freeCoef = res

    err = computeError(tmpCoefs, kernel, features, classes)
    if err < listOfKernelResults[currentDatasetNumber][kernel]:
        listOfKernelResults[currentDatasetNumber][kernel] = err
        param = -1
        if kernel == "polynomial":
            param = p
        elif kernel == "gaussian":
            param = b
        listOfBestKernelParams[currentDatasetNumber][kernel] = param
        listOfBestC[currentDatasetNumber] = c
        coefs[:] = tmpCoefs.copy()


def drawGraph(coefs, kernel, features, classes):
    size = 30

    backgroundFX = []
    backgroundFY = []
    backgroundSX = []
    backgroundSY = []

    y, step = -2 if currentDatasetNumber == 0 else 0, 0.05
    yCond = 1.5 if currentDatasetNumber == 0 else 7
    xCond = 2 if currentDatasetNumber == 0 else 24
    while y < yCond:
        x = -1.5 if currentDatasetNumber == 0 else 0
        while x < xCond:
            res = freeCoef
            for i in range(len(classes)):
                res += getClass(classes[i]) * coefs[i] * computeKernel(kernel, [x, y], features[i])

            if res < - eps:
                backgroundFX.append(x)
                backgroundFY.append(y)
            elif res > eps:
                backgroundSX.append(x)
                backgroundSY.append(y)
            x += step
        y += step

    fig, ax = plt.subplots()

    ax.scatter(backgroundFX, backgroundFY, marker='s', c=[[0 / 255.0, 75 / 255.0, 75 / 255.0]], s=size * 10,
               label="First class area")
    ax.scatter(backgroundSX, backgroundSY, marker='s', c=[[1, 1, 0]], s=size * 10, label="Second class area")

    firstClassX = [features[i][0] for i in range(len(classes)) if getClass(classes[i]) == 1]
    firstClassY = [features[i][1] for i in range(len(classes)) if getClass(classes[i]) == 1]
    secondClassX = [features[i][0] for i in range(len(classes)) if getClass(classes[i]) == -1]
    secondClassY = [features[i][1] for i in range(len(classes)) if getClass(classes[i]) == -1]

    ax.scatter(firstClassX, firstClassY, c=[[0.1, 0.63, 0.55]], s=size, label="First class point")
    ax.scatter(secondClassX, secondClassY, c='r', s=size, label="Second class point")

    fig.set_figwidth(14)
    fig.set_figheight(14)
    ax.legend(
        fontsize=20,
        title=listOfNames[currentDatasetNumber] + " " + kernel,
        title_fontsize=25
    )
    plt.show()
    fig.savefig(listOfNames[currentDatasetNumber] + " " + kernel)


def printDictionary(dictionary, index, f):
    for key, value in dictionary[index].items():
        f.write("\t" + key + ": ")
        f.write(str(value) + "\n")


def solver(path):
    global p, b
    [features, classes] = Utils.getDataAndClasses(path)
    features = features.values.tolist()
    classes = classes.values.tolist()
    forLinear = []  # коэфициеты для линейного ядра
    forPolynomial = []  # коэфициенты для полиномиального ядра
    forGaussian = []  # коэфициенты для гауссового ядра

    for c in C:
        for kernel in kernels:
            if kernel == "linear":
                computeSVM(forLinear, "linear", c, features, classes)
            elif kernel == "polynomial":
                for i in polynomialDegrees:
                    p = i
                    computeSVM(forPolynomial, "polynomial", c, features, classes)
            elif kernel == "gaussian":
                step = 0.1
                while b <= sectionForGaussian[1]:
                    computeSVM(forGaussian, "gaussian", c, features, classes)
                    b += step
    f = open(listOfNames[currentDatasetNumber] + ".txt", "w")
    f.write("--" + listOfNames[currentDatasetNumber] + "--\n")
    f.write("Coefs:\n")
    f.write(str(forLinear) + "\n")
    f.write(str(forPolynomial) + "\n")
    f.write(str(forGaussian) + "\n")
    f.write("Kernel errors:\n")
    printDictionary(listOfKernelResults, currentDatasetNumber, f)
    f.write("Kernels param:\n")
    printDictionary(listOfBestKernelParams, currentDatasetNumber, f)
    f.write("Best C param: " + str(listOfBestC[currentDatasetNumber]) + "\n")
    f.close()
    drawGraph(forLinear, "linear", features, classes)
    drawGraph(forPolynomial, "polynomial", features, classes)
    drawGraph(forGaussian, "gaussian", features, classes)


def reeinitParameters():
    global p, b, freeCoef
    p = 2
    b = 1
    freeCoef = 0


solver("datasets/chips.csv")
currentDatasetNumber += 1
reeinitParameters()
solver("datasets/geyser.csv")
