# Week 9 - Support Vector Machines
## Topics:
- SVM Concept: Margin Maximization
- Kernel Trick
- Hyperparameter Tuning

## Activities:
- Implement SVM using `scikit-learn`
- Compare linear and kernel SVM performance

Saatnya kita berpindah ke algoritma selanjutnya, yaitu Support Vector Machine.

### Support Vector Machine ###

Di bagian ini, kita akan menggunakan dataset bunga iris yang diambil dari [https://archive.ics.uci.edu/ml/datasets/iris](https://archive.ics.uci.edu/ml/datasets/iris). Dataset ini terdiri dari tiga class yaitu Iris Sentosa, Iris Versicolor, dan Iris Virginica. Dataset ini sudah sangat lazim dipakai untuk kasus supervised learning terutama klasifikasi.  

#### Load dataset ####
Langkah pertama, mari kita load terlebih dahulu dataset ini dengan cara berikut

```python
import pandas as pd

data = pd.read_csv('dataset/iris.data',sep=',',names=['sepal length','sepal width','petal length','petal width','class'])
data.head()
```

Di bagian kode <code>read_csv</code> ada tiga parameter yaitu lokasi dataset, kemudian <code>sep</code> yang berarti adalah separator atau pemisah data, dan <code>names</code> yang berfungsi untuk memberi header pada dataset.

#### Normalisasi Data ####
Sebelum normalisasi data, kita pisahkan dulu antara atribut dengan class, caranya adalah:

```python
# pemisahan dataset di mana X untuk menampung data atribut sedangkan y untuk data class
X = data.drop('class',axis=1)
y = data['class']
```

Sedangkan untuk normalisai, kita tetap menggunakan MinMax scaler

```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)

y = y.astype('category')
y = y.cat.codes
```

Langkah ketiga, kita pisahkan dataset untuk training dan testingnya dengan porsi masing-masing 80% dan 20% dengan cara:

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle = True, test_size=0.2, random_state = 42) 
```

#### Data Training ####
Langkah keempat adalah data training. Di sklearn, sudah ada class SVM yang dipakai untuk klasifikasi yaitu Support Vector Classification (SVC). Di SVC, sudah ada beberapa parameter kernel yaitu: linear, poly, rbf, dan sigmoid. Kita akan mencoba masing-masing kernel dan melihat performanya terhadap klasifikasi bunga Iris.
Pada bagian pertama, kita coba dulu dengan kernel **linear**.  
##### Kernel Linear #####

```python
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score

clf = make_pipeline(MinMaxScaler(), SVC(kernel='linear'))
clf.fit(X_train, y_train)
```

#### Data Testing ####
Setelah training selesai, saatnya kita memprediksi dan kita ukur performanya dengan metriks

```python
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred )

print('Akurasi: %.3f' % acc)
```

Dari hasil prediksi dengan kernel linear, kita mendapatkan akurasi prediksi sebesar 0,933. Kita coba lagi dengan kernel **poly**.

```python
clf = make_pipeline(MinMaxScaler(), SVC(kernel='poly'))
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred )

print('Akurasi: %.3f' % acc)
```

Nah, dengan kernel poly, kita mendapatkan akurasi sebesar 0,900. Sementara ini, hasil akurasi dengan kernel linear lebih bagus dibanding dengan poly. Tapi, bagaimana hasilnya jika menggunakan kernel **RBF**? RBF kepanjangan dari Radial Basis Function. Apa itu RBF? Silakan dibaca di [sini](https://en.wikipedia.org/wiki/Radial_basis_function).  
##### Kernel RBF #####

```python
clf = make_pipeline(MinMaxScaler(), SVC(kernel='rbf'))
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred )

print('Akurasi: %.3f' % acc)
```

Ternyata, hasil kernel RBF dan Linear untuk dataset bunga Iris memiliki hasil akurasi yang sama. Oke, itu dulu untuk SVM. Kita lanjutkan ke salah satu arsitektur neural network (NN) yaitu Multi-layered perceptron (MLP).

