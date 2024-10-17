

class AttributeInfo:
    def __init__(self, name, type, value, numsofvalues):
        self.name = name
        self.type = type
        self.value = value
        self.numsofvalues = numsofvalues

    def getName(self):
        return self.name

    def getType(self):
        return self.type

    def getValue(self):
        return self.value

    def getNumsofValues(self):
        return self.numsofvalues