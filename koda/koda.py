import numpy as np
from matplotlib import pyplot as plt

train_raw = np.genfromtxt('../podatki/train.csv', delimiter=',', skip_header=1)
test_raw = np.genfromtxt('../podatki/test.csv', delimiter=',', skip_header=1)

train_data = []
test_data = []

#razdelitev podatkov v terke z dejanskim rezultatom in matriko (tabelo tabel) velikosti 28x28
for line in train_raw:
    train_data.append((line[1], line[1:].reshape(28, 28)))
for line in test_raw:
    test_data.append(line.reshape(28, 28))


#samo za demo
plt.imshow(train_data[0][1])
plt.show()
plt.imshow(test_data[0])
plt.show()
