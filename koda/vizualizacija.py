import numpy as np
from matplotlib import pyplot as plt
print("Getting data")
train_raw = np.genfromtxt('../podatki/train.csv', delimiter=',', skip_header=1, dtype=np.int16)
test_raw = np.genfromtxt('../podatki/test.csv', delimiter=',', skip_header=1, dtype=np.int16)
print("Got data")
train_data = []
test_data = []
rezultati = []
#razdelitev podatkov v terke z dejanskim rezultatom in matriko (tabelo tabel) velikosti 28x28
for line in train_raw:
    rezultati.append(line[0])
    train_data.append(line[1:].reshape(28, 28))
for line in test_raw:
    test_data.append(line.reshape(28, 28))

train_data = np.array(train_data)
test_data = np.array(test_data)
rezultati = np.array(rezultati)   #pricakovani rezultati...del train_data


def vizualizacija(rez,data,stevilka):
    fig, axs = plt.subplots(3, 3)
    axs[0,0].hist(rez, density=True, bins=10, edgecolor='black', linewidth=1.2)
    selection = rez == stevilka
    selected_data = data[selection]
    axs[0,1].imshow(selected_data[0])
    axs[0,2].imshow(selected_data[1])
    axs[1,0].imshow(selected_data[2])
    axs[1,1].imshow(selected_data[3])
    axs[1,2].imshow(selected_data[4])
    axs[2,0].imshow(selected_data[5])
    axs[2,1].imshow(selected_data[6])
    axs[2,2].imshow(selected_data[7])


    plt.show()

#Vpiši vizualizacija katere številke te zanima, dobis prvih 8
vizualizacija(rezultati,train_data,1)

