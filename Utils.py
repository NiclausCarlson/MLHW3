import pandas as pandas


def getDataAndClasses(path):
    data = pandas.read_csv(path)
    features = data[['x', 'y']]
    classes = data['class']
    return [features, classes]


def printDictionary(dictionary, index, f):
    for key, value in dictionary[index].items():
        f.write("\t" + key + ": ")
        f.write(str(value) + "\n")
