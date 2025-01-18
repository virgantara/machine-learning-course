# Week 8 - Mid-Term Mini Projects
## Topics:
- Intro to Scikit-Learn
- Apply knowledge to a mini project
- Design, implement, and present ML solutions

## Activities:
- Group work on projects
- Present project results and challenges

## Overview ##

1. Pengenalan Scikit Learn
1. Instalasi
1. Regresi Linier
1. Regresi Logistik
1. Support Vector Machine
1. Multi-Layered Perceptron (MLP)
1. K-Fold Cross Validation

### 1. Pengenalan Scikit Learn ###

Scikit-Learn (SKLearn) adalah library python untuk machine learning yang mendukung supervised dan unsupervised learning. SKLearn merupakan produk dari project **Google Summer of Code project** oleh **David Cournapeau**. 

### 2. Instalasi ###

Instalasi scikit-learn cukup mudah, yaitu cukup dengan menjalankan perintah berikut:

$<code>pip install scikit-learn</code>  

Adapun dengan Python 3, perintahnya adalah:

$<code>pip3 install scikit-learn</code>

### 3. Regresi Linier ###

#### Import library yang diperlukan ####

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics

```

#### Membaca dataset dengan library Pandas ####
```python
data = pd.read_csv('dataset/oregon.csv')

data.columns =(['luas','jmltt','harga'])
data.drop(['jmltt'], axis=1, inplace=True)

X = data['luas']
y = data['harga']
data.head()
```

#### Plot datanya dengan matplotlib
```python
plt.scatter(X.values, y.values)
plt.show()
```

#### Data Splitting ####
```
x_train, x_test,y_train,y_test = train_test_split(X,y,test_size =0.2)
```

#### Model LinearRegression ####

Di scikit-learn terdapat beberapa class untuk machine learning, diantaranya adalah LinearRegression, LogisticRegression, MLPClassifier, dan sebagainya. Di bagian ini, kita akan menggunakan class LinearRegression. Caranya adalah:
```python
from sklearn.linear_model import LinearRegression
lr = LinearRegression()  # load class
```

#### Data Training ####
Data training dengan scikit-learn cukup mudah, hanya dengan memanggil perintah <code>fit</code>, fungsi-fungsi seperti feedforwarding dan backpro sudah berjalan. Berikut adalah kodenya:

```python
lr.fit(x_train.values.reshape(-1,1), y_train.values.reshape(-1,1))
```

#### Prediksi dan Evaluasi Model ####
Di sini, kita akan memprediksi dengan data hasil splitting dan mengevaluasi performanya dengan MSE.

```python
pred_y = lr.predict(x_test.values.reshape(-1,1))
print("Hasil prediksi 5 data teratas {}".format(pred_y[:5]))
print("Data riil 5 data teratas {}".format(y_test[:5]))

mse = lr.score(x_test.values.reshape(-1,1), y_test.values.reshape(-1,1))
print("R2 {}".format(mse))
```

### 4. Regresi Logistik ### 

Di regresi logistik, kita akan memprediksi suatu tumor apakah ganas atau jinak dari dataset di [https://www.kaggle.com/uciml/breast-cancer-wisconsin-data](https://www.kaggle.com/uciml/breast-cancer-wisconsin-data).

#### Load dataset ####

```python
import pandas as pd

df = pd.read_csv("dataset/tumor.csv")
df = df.drop('Unnamed: 32', axis=1) # kolom ini dibuang
X = df.drop(['id','diagnosis'],axis=1) # mengambil nilai variabel independen/variabel penentu nilai y/atribut
# X apakah matrix / vektor? matrix

y = df.iloc[:,1] # mengambil nilai variabel dependen/variabel yang ditentukan oleh variabel independennya

# y apakah matriks/vektor? vektor 
```

#### Normalisasi Data ####
Untuk normalisasi data, kita menggunakan min max scalling.

```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)
```

#### Data splitting dan Training ####
Data train dan test kita pisah dengan porsi 80:20

```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
# library scikit learn / sklearn

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle = True, test_size=0.2) 

clf = LogisticRegression(max_iter=1000) # max_iter = max epoch = 1000 kali epoch
clf.fit(X_train,y_train) # fit itu berisi training, validation, dan learning rate
```

#### Data Prediksi dan evaluasi performa ####

Setelah training, kini saatnya kita memprediksi dengan data test dan mengevaluasi akurasi model yang dibangung. 

```python
y_pred = clf.predict(X_test) # prediksi thd test 20%


print("Actual breast cancer : ")
print(y_test.values)

print("\nPredicted breast cancer : ")
print(y_pred)

print("\nAccuracy score : %f" %(accuracy_score(y_test, y_pred) * 100))

```

Setiap kelompok diminta untuk merancang, membangun, dan mengimplementasikan sebuah model prediktif berbasis dataset pilihan mereka dengan memanfaatkan teknik machine learning yang telah dipelajari dari Week 1 hingga Week 7. Tugas ini bertujuan untuk mengintegrasikan teori dan praktik dalam satu proyek nyata.

## Instruksi Detail:

- Pilih Dataset
Setiap kelompok harus memilih dataset publik yang relevan, seperti data prediksi harga rumah, klasifikasi email spam, atau dataset lain yang sesuai dengan minat mereka. Dataset harus memiliki fitur numerik atau kategorikal, dan minimal 500 baris data.

- Eksplorasi dan Persiapan Data
	1. Bersihkan data dari nilai yang hilang atau tidak relevan.
	2. Lakukan analisis eksplorasi data (EDA) untuk memahami distribusi dan pola data.
	3. Bagi dataset menjadi data latih dan data uji (contoh: 80% latih, 20% uji).

- Setiap kelompok harus mengimplementasikan minimal tiga teknik machine learning yang telah dipelajari, seperti:
		1. Linear Regression untuk prediksi variabel kontinu.
		2. Naive Bayes untuk klasifikasi.
		3. Decision Tree untuk analisis keputusan.
        4. Implementasikan Neural Network (Feedforward atau Backpropagation) untuk membandingkan performa.

- Evaluasi Model
        1. Gunakan metrik evaluasi seperti MAE/MSE untuk regresi atau akurasi/F1-score untuk klasifikasi.
        2. Bandingkan kinerja model yang berbeda dan beri argumen mengapa satu model lebih baik dari yang lain untuk dataset tersebut.

- Buat laporan mini (maksimal 5 halaman) berisi:
    1. Deskripsi dataset.
    2. Proses eksplorasi data.
    3. Metode yang digunakan.
    4. Hasil evaluasi model.
    5. Kesimpulan.
