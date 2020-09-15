# soal no 1

Apa itu machine learning?
Jawab : machine learning adalah komputer yang memiliki kemampuan melakukan belajar dari pengalaman tugasnya di masa lalu,
        dan mengalami peningkatan kinerjanya di masa depan.

Apa itu data feature dan data target?
Jawab : data feature adalah variabel yang memprediksi suatu kejadian, 
        sedangkan data target adalah hasil dari prediksi tersebut.

Apa Perbedaan Supervised Learning dan Unsupervised Learning?
Jawab : Supervised Learning memiliki column / data label, outputnya jelas, 
        sedangkan Unsupervised Learning tidak memiliki label /column, outputnya tidak jelas

Apa Jenis2 yang ada di dalam Supervised Learning? Jelaskan Perbedaannya!
Jawab : - klasifikasi -> outputnya berupa class label
        - regresi -> outputnya berupa quantity yang continues

Apa perbedaan Hyperparameter dan Parameter?
Jawab : - hyperparameter -> didefinisikan di awal pembuatan objek model 
        - paramater -> dihasilkan diakhir

Sebutkan Langkah-langkah dalam mengaplikasikan algoritma apapun dalam machine learning!
Jawab : 1. pilih model atau algoritma yang akan digunakan
        2. pilih hyperparameter ketika membuat objek model
        3. pisahkan antara data feature dan data target
        4. perintahkan model untuk mempelajari data dengan menggunakan method .fit()
        5. applikasikan model ke dalam test data dengan menggunakan method .predict() untuk supervised learning atau
            .transform() untuk unsupervised learning

# soal no 2 

from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import numpy as np

rng = np.random.RandomState(42)
x = 20 * rng.rand(50)
y = x**2 + 2 * x + - 1 + rng.randn(50)

model_dtr = DecisionTreeRegressor(max_depth = 4, max_features = None, max_leaf_nodes = 7, min_samples_leaf = 5)

x_matriks = x[:, np.newaxis]
model_dtr.fit(x_matriks, y)

x_new = np.arange(20, 30, 0.5)
x_new = x_new[:, np.newaxis]

y_test = model_dtr.predict(x_new)

y_train = model_dtr.predict(x_matriks)

fig, ax = plt.subplots(figsize=(12, 6))
ax.scatter(x, y, label = 'Data Asli')
plt.plot(x, y_train, c = 'y', label = 'Prediksi Terhadap Data Training', linestyle = 'none', marker = 'o')
plt.plot(x_new, y_test, c = 'g', label = 'Prediksi Terhadap Data Baru')
ax.set_xlabel('X-Value')
ax.set_ylabel('Y-Value')
plt.legend()
plt.show()
tree.plot_tree(model_dtr)
