# -*- coding: utf-8 -*-
"""
# **Business Understanding**

Pemerintah Kota Bandung memiliki program kerja untuk memberikan bantuan sosial kepada fakir miskin, sehingga untuk mengefektifkan serta mengefesiensikan dana tersebut agar tepat sasaran, maka Pemerintah Kota Bandung memerlukan data yang valid mengenai kondisi perekonomian masyarakat di setiap kelurahannya. Hal ini tentunya relevan dengan kondisi Indonesia saat ini (Adanya Covid 19), yang menuntut Pemerintah Kota Bandung untuk memprioritaskan masyarakat dengan Penghasilan Rendah.

1. Berapa banyak kependudukan di suatu wilayah berdasarkan kecamatan di Kota Bandung pada tahun 2017-2020?
2. Daerah mana yang memiliki kepadatan penduduk yang paling tinggi dan rendah?
3. Bagaimana korelasi Korelasi Penduduk MBR dan Penduduk Miskin dengan Lulusan SD di kecamatan yang ada di Kota Bandung?
4. Prediksi jumlah penduduk miskin dengan berdasarkan banyaknya masyarakat dengan tamatan SD

# **Data Understanding**

## `Kepadatan Penduduk`
"""

import pandas as pd

data_kepadatan1 = pd.read_csv('KepadatanPenduduk2017.csv')
data_kepadatan2 = pd.read_csv('KepadatanPenduduk2018.csv')
data_kepadatan3 = pd.read_csv('KepadatanPenduduk2019.csv')
data_kepadatan4 = pd.read_csv('KepadatanPenduduk2020.csv')

data_kepadatan1.head()

# #tambah kolom tahun di setiap data
data_kepadatan1.insert(3, "Tahun", "2017")
data_kepadatan2.insert(3, "Tahun", "2018")
data_kepadatan3.insert(3, "Tahun", "2019")
data_kepadatan4.insert(3, "Tahun", "2020")

kpd_pertahun = []
kpd_pertahun.append(data_kepadatan1)
kpd_pertahun.append(data_kepadatan2)
kpd_pertahun.append(data_kepadatan3)
kpd_pertahun.append(data_kepadatan4)

#tes data kpd tahun 2017
kpd_pertahun[0].head()

"""## `Data Masyarakat Berpenghasilan Rendah dan Non Rendah`"""

data = pd.read_csv('MbrNon2017Fix.csv')

data.head()

#Menghapus nilai kosong
data = data.drop(labels = ['No.'], axis = 1)
data = data.dropna()
data = data.reset_index(drop=True)
data.head()

data = data.rename(columns={'Jumlah Kepala Rumah Tangga MBR': 'Jumlah_KRT_MBR', 'Jumlah Kepala Rumah Tangga Non MBR': 'Jumlah_KRT_NONMBR'})
data.info()

#memilih data mbr
data_mbr = data[['Kecamatan','Jumlah_KRT_MBR']]
data_mbr.info()

data_mbr

"""# **Data Cleaning and Preparation**"""

#rename kolom pada dataset Kepadatan Penduduk
kpd_tahun = []
for i in range(4):
    data = kpd_pertahun[i].rename(columns={'Jumlah Penduduk': 'jml_pend', 'Luas Wilayah': 'luas_wil'})
    kpd_tahun.append(data)

# cek rename
kpd_tahun[1].info()

kpd_tahun[1].to_csv("Hasil1.csv", index=False) #Menyimpan file pada Hasil1.csv

"""## `Tingkat Pendidikan Masyarakat Miskin`"""

#Load data
data2 = pd.read_csv('PendidikanMiskin2017.csv')

data2.head()

#Menghapus nilai kosong
data2 = data2.drop(labels = ['No'], axis = 1)
data2 = data2.dropna()
data2 = data2.reset_index(drop=True)
data2.head()

#memilih data sd
data_sd = data2[['Kecamatan','SD']]
data_sd.info()

"""# **Exploratory Data Analysis**

## `Persebaran Penduduk`
"""

#Persebaran penduduk di setiap kecamatan (dari tahun 2017-2020)
import seaborn as sns
data_kpd = {'Kecamatan': kpd_tahun[0]['Kecamatan'],
        '2017':kpd_tahun[0]['jml_pend'],
        '2018':kpd_tahun[1]['jml_pend'],
        '2019':kpd_tahun[2]['jml_pend'],
        '2020':kpd_tahun[3]['jml_pend'],}

data_df = pd.DataFrame.from_dict(data_kpd)
vis_df = pd.melt(data_df, id_vars="Kecamatan", var_name="Tahun", value_name="Jumlah Penduduk")

sns.factorplot(x='Jumlah Penduduk', y='Kecamatan', hue='Tahun', data=vis_df, kind='bar', height=7, aspect=1, legend_out=True)

"""## `Korelasi Penduduk MBR dan Penduduk Miskin dengan Lulusan SD`"""

df_mbr = {'Kecamatan': data_mbr['Kecamatan'],
        '2017':data_mbr['Jumlah_KRT_MBR']}

data_df = pd.DataFrame.from_dict(df_mbr)

df_sd = {'Kecamatan': data_sd['Kecamatan'],
        '2017':data_sd['SD']}

sd_df = pd.DataFrame.from_dict(df_sd)

import matplotlib.pyplot as plt

plt.scatter(data_mbr['Jumlah_KRT_MBR'], data_sd['SD'])
plt.legend(loc='best', fontsize=16)
plt.xlabel('Masyarakat Berpenghasilan Rendah')
plt.ylabel('Penduduk Lulusan SD')

plt

#korelasi per kecamatan
from scipy.stats import pearsonr
corr_perkecamatan = []

for col in data_df.columns:
  if col != 'Kecamatan':
    corr, _ = pearsonr(data_df[col],df_sd[col])
    corr_perkecamatan.append(corr)

print(corr_perkec)

"""## `Tahap Regresi`

**Menggabungkan Data Penghasilan MBR dan Tingkat Pendidikan**
"""

mis = []
put = []
for col in data_df.columns:
  if col != 'Kecamatan':
    mis.extend(data_df[col])
    put.extend(sd_df[col])

whole_data = {'mbr':mis, 'sd':put}
whole_data_df = pd.DataFrame.from_dict(whole_data)

whole_data_df.info()

"""**Menghapus Outlier**"""

import numpy as np
from scipy import stats

z_scores = stats.zscore(whole_data_df)
abs_z_scores = np.abs(z_scores)
filtered_entries = (abs_z_scores < 3).all(axis=1)
new_df = whole_data_df[filtered_entries]
new_df.info()

X = np.array([new_df['sd']]).reshape((29,1))
y = np.array([new_df['mbr']]).reshape((29,1))
X.shape

y.shape

"""**Split Data (Train dan Test)**"""

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

X_train.shape

X_test.shape

"""**Training dan Evaluasi Model dengan Regresi Linear** ✅"""

#REGRESI LINEAR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import math

model = LinearRegression()
model.fit(X_train, y_train)

model.score(X_test, y_test)

y_predicted = model.predict(X_test)

print('Nilai MAE : '+str(mean_absolute_error(y_test, y_predicted)))
print('Nilai MSE : '+str(mean_squared_error(y_test, y_predicted)))
print('Nilai RMSE : '+str(math.sqrt(mean_squared_error(y_test, y_predicted))))

plt.scatter(X_train, y_train, color = "red")
plt.plot(X_train, model.predict(X_train), color = "purple")
plt.title("Hasil Prediksi dengan data Training")
plt.xlabel("Penduduk Lulusan SD")
plt.ylabel("Jumlah Masyarakat Berpenghasilan Rendah")
plt.show()

plt.scatter(X_test, y_test, color = "green")
plt.plot(X_test, model.predict(X_test), color = "purple")
plt.title("Hasil Prediksi dengan data Test")
plt.xlabel("Penduduk Lulusan SD")
plt.ylabel("Jumlah Masyarakat Berpenghasilan Rendah")
plt.show()

"""**Model menggunakan Ridge Regression dengan GridSearchCV**✅"""

from sklearn.linear_model import Ridge
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV

ridge = Ridge()
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

space = dict()
space['solver'] = ['svd', 'cholesky', 'lsqr', 'sag']
space['alpha'] = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]
space['fit_intercept'] = [True, False]
space['normalize'] = [True, False]

search = GridSearchCV(ri, space, scoring='neg_mean_absolute_error', n_jobs=-1, cv=cv)

result = search.fit(X, y)

"""**Evaluasi Model pada Ridge Regression dengan GridSearchCV**"""

print('Best Score: %s' % result.best_score_)
print('Best Hyperparameters: %s' % result.best_params_)

clf_best = result.best_estimator_
clf_best.score(X_test,y_test)

y_predicted = clf_best.predict(X_test)

print('Nilai MAE : '+str(mean_absolute_error(y_test, y_predicted)))
print('Nilai MSE : '+str(mean_squared_error(y_test, y_predicted)))
print('Nilai RMSE : '+str(math.sqrt(mean_squared_error(y_test, y_predicted))))

"""**Visualiasasi Model pada Ridge Regression dengan GridSearchCV**"""

plt.scatter(X_train, y_train, color = "red")
plt.plot(X_train, clf_best.predict(X_train), color = "green")
plt.title("Hasil Prediksi (Training set)")
plt.xlabel("Penduduk Lulusan SD")
plt.ylabel("Jumlah Masyarakat Berpenghasilan Rendah")
plt.show()

plt.scatter(X_test, y_test, color = "red")
plt.plot(X_test, clf_best.predict(X_test), color = "green")
plt.title("Hasil Prediksi (Test set)")
plt.xlabel("Penduduk Lulusan SD")
plt.ylabel("Jumlah Masyarakat Berpenghasilan Rendah")
plt.show()

"""# **Conclusion**

1. Program ini dapat mengetahui banyak kependudukan di suatu wilayah berdasarkan kecamatan di Kota Bandung yang diimplementasikan pada Grafik Persebaran penduduk di setiap kecamatan (dari tahun 2017-2020)
2. Daerah yang memiliki kepadatan penduduk yang paling tinggi ialah Kecamatan "Babakan Ciparay"  dan kepadatan penduduk yang rendah ada pada Kecamatan "Cinambo"
3. Hasil Korelasi Penduduk MBR dan Penduduk Miskin dengan Lulusan SD di kecamatan yang ada di Kota Bandung ialah bernilai positif dengan nilai koefisien korelasi Pearson yaitu sebesar 0.6942072305666471.
4. Prediksi jumlah penduduk miskin dengan berdasarkan banyaknya masyarakat dengan tamatan SD dengan menggunakan model Lasso regression yang telah di fine tuning menggunakan GridSearchCV menghasilkan score r2 sebesar 0.6792875608228346 dan nilai RMSE sebesar 1190.670238990295.
"""