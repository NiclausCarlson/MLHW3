import pandas as pandas


def getDataAndClasses(path):
    data = pandas.read_csv(path)
    features = data[['x', 'y']]
    classes = data['class']
    return [features, classes]
