from math import pi, sin

kSIMPLE_TRAIN = [(1, False), (2, True), (4, False), (5, True), (13, False),
                 (14, True), (19, False)]

class SinClassifier:

    def __init__(self, w):
        assert isinstance(w, float)
        self.w = w

    def __call__(self, k):
        #print("w : "+str(self.w))
        #print("k : "+str(k))
        return sin(self.w * 2 ** (-k))

    def classify(self, k):
        assert isinstance(k, int), "Object to be classified must be an integer"

        if self(k) >= 0:
            return True
        else:
            return False


def train_sin_classifier(data):
    assert all(isinstance(k[0], int) and k >= 0 for k in data), \
        "All training inputs must be integers"
    assert all(isinstance(k[1], bool) for k in data), \
        "All labels must be True / False"

    x= 0.0
    w= 1.0

    for i in data:
        if not i[1]:
            x += 2.**i[0]
            #print(x)
    w += x
    print(w)
    print(pi*w)
    return SinClassifier(pi*w)


if __name__ == "__main__":
    classifier = train_sin_classifier(kSIMPLE_TRAIN)
    for kk, yy in kSIMPLE_TRAIN:
        print(kk, yy, classifier(kk), classifier.classify(kk))
