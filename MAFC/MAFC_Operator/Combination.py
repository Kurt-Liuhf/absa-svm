import math

class Combination:

    def __init__(self, n: int, r: int):
        if r > n:
            raise Exception('IllegalArgumentException')

        if n < 1:
            raise Exception('IllegalArgumentException')

        self.n = n
        self.r = r
        self.a = r * [0]
        nFact = math.factorial(n)
        rFact = math.factorial(r)
        nminusrFact = math.factorial(n - r)
        self.total = nFact / (rFact * nminusrFact)
        self.numLeft: int = 0
        self.reset()

    def reset(self):
        self.a = list(range(len(self.a)))
        self.numLeft = self.total

    # Are there more combinations?
    def hasMore(self) -> bool:
        return self.numLeft > 0

    # Generate next combination (algorithm from Rosen p. 286)
    def getNext(self) -> list:

        if self.numLeft == self.total:
            self.numLeft = self.numLeft - 1
            return self.a

        i = self.r - 1
        while self.a[i] == (self.n - self.r + i):
            i -= 1

        self.a[i] = self.a[i] + 1
        for j in range(i + 1, self.r):
            self.a[j] = self.a[i] + j - i

        self.numLeft = self.numLeft - 1
        return self.a
