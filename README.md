# **Laporan Proyek Machine Learning - Usamah Putra Firdaus**

## **Domain Proyek**

Diabetes merupakan salah satu penyakit kronis yang paling banyak diderita di seluruh dunia. Berdasarkan data dari World Health Organization (WHO), jumlah penderita diabetes terus meningkat setiap tahunnya, baik di negara maju maupun berkembang. Penyakit ini seringkali tidak menunjukkan gejala pada tahap awal, sehingga banyak penderita yang tidak menyadari bahwa mereka mengidap diabetes hingga komplikasi muncul. Kondisi ini menyebabkan keterlambatan dalam penanganan yang dapat memperburuk kesehatan pasien [[1]](https://www.who.int/news-room/fact-sheets/detail/diabetes).

Dalam era digital saat ini, kemajuan teknologi di bidang data science dan machine learning membuka peluang besar untuk mendukung deteksi dini penyakit, termasuk diabetes. Dengan memanfaatkan data kesehatan seperti kadar glukosa darah, tekanan darah, indeks massa tubuh (BMI), dan faktor risiko lainnya, kita dapat membangun sistem prediksi yang membantu tenaga medis maupun individu dalam mengidentifikasi risiko diabetes secara lebih cepat dan akurat.

Proyek ini bertujuan untuk membangun model predictive analytics yang mampu mendeteksi kemungkinan seseorang menderita diabetes berdasarkan [data kesehatan](https://www.kaggle.com/datasets/nanditapore/healthcare-diabetes). Dengan pendekatan ini, diharapkan dapat mendukung pengambilan keputusan yang lebih tepat dalam pencegahan maupun penanganan awal diabetes, sehingga menekan angka penderita dan meningkatkan kualitas hidup masyarakat.

## Business Understanding
### Problem Statements
Dari latar belakang diatas, maka rumusan masalah yang akan dibahas pada proyek ini sebagai berikut:
- fitur apa saja yang berpengaruh dalam proses modeling predictive analytics terhadap penderita diabetes?
- Model machine learning apa yang memiliki akurasi tertinggi dan tingkat kesalahan prediksi paling rendah dalam mendeteksi penderita diabetes?

### Goals
Berdasarkan Problem Statement yang telah disebutkan, berikut adalah tujuan/goals dari proyek ini sebagai berikut:
- Menganalisis pentingnya setiap fitur terhadap penderita diabetes.
- Membandingkan performa beberapa model *machine learning* untuk menemukan model dengan akurasi terbaik dan kesalahan prediksi paling minim.

### Solution Statements
- Melakukan Exploratory Data Analysis (EDA) dengan matriks korelasi untuk melihat fitur-fitur yang berpengaruh
- Membandingkan 3 performa model *Machine Learning* yaitu Logistic Regression, Decision Tree, dan Random Forest
- Melakukan Evaluasi model menggunakan Confusion Matrix untuk melihat mana model yang paling sedikit melakukan kesalahan prediksi

## **Data Understanding**
Dataset yang digunakan untuk memprediksi seseorang yang beresiko mengalami diabetes. Dataset diambil dari kaggle yang dapat diakses [disini](https://www.kaggle.com/datasets/nanditapore/healthcare-diabetes), dataset ini dipubilkasi oleh [Nandita Pore](https://www.kaggle.com/nanditapore) pada tahun 2023. Dataset ini berisi berbagai atribut yang berkaitan dengan kesehatan, yang dikumpulkan secara cermat untuk mendukung pengembangan model prediktif dalam mengidentifikasi individu yang berisiko menderita diabetes.

  ![img alt](https://github.com/UsamahPutraFirdaus/Submission_MLTerapan/blob/main/Submission_1_PredictiveAnalytics/img/variable.png?raw=true)

Data Healthcare Diabetes yang digunakan terdapat 10 kolom dan 2768 baris data.

### Variable Description
- Id: Identitas unik untuk setiap entri data.
- Pregnancies: Jumlah kehamilan yang pernah dialami.
- Glucose: Konsentrasi glukosa plasma selama 2 jam dalam tes toleransi glukosa oral.
- BloodPressure: Tekanan darah diastolik (mm Hg).
- SkinThickness: Ketebalan lipatan kulit trisep (mm).
- Insulin: Kadar insulin serum selama 2 jam (mu U/ml).
- BMI: Indeks massa tubuh (berat dalam kg / tinggi dalam m²).
- DiabetesPedigreeFunction: Skor genetik risiko diabetes berdasarkan silsilah keluarga.
- Age: Usia dalam tahun.
- Outcome: Klasifikasi biner yang menunjukkan adanya (1) atau tidak adanya (0) diabetes.


### Exploratory Data Analysis (EDA)
**A. Menangani Missing Valye & Duplicate Data**
Pada tahap ini, dilakukan pemeriksaan terhadap data yang tidak valid dalam dataset. Hasil pemeriksaan menunjukkan bahwa tidak terdapat nilai null pada kolom mana pun, dan tidak ditemukan data yang duplikat. Oleh karena itu, data dinyatakan siap untuk dianalisis pada tahap berikutnya.

**B. EDA - Univariete Analysis**
1. Distribusi Usia

  ![img alt](https://github.com/UsamahPutraFirdaus/Submission_MLTerapan/blob/main/Submission_1_PredictiveAnalytics/img/distribusi%20usia.png?raw=true)
   
   **Interpretasi Visualisasi Distribusi Usia**
   
   - Dataset ini didominasi oleh orang-orang berusia muda, terutama awal 20-an.
   - cukup sedikit orang yang berusia di atas 60 tahun.

3. Distribusi Jumlah Orang perKelompok Usia

  ![img alt](https://github.com/UsamahPutraFirdaus/Submission_MLTerapan/blob/main/Submission_1_PredictiveAnalytics/img/distribusi%20jumlah%20orang%20per%20kelompok.png?raw=true)

   >```Ruby
   ># Buat kategori umur
   >bins = [20, 30, 40, 50, 60, 70, 80]
   >labels = ['20-30', '31-40', '41-50', '51-60', '61-70', '71-85']
   >df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels, right=True)
   >```
   Kode diatas digunakan untuk membuat kelompok umur
   
   **Interpretasi Visualisasi Distribusi Jumlah Orang perKelompok Usia**
   
   - Kelompok usia muda (20–30) adalah mayoritas populasi.
   - Populasi mengecil seiring bertambahnya usia, terutama di atas 50 tahun.
     
4. Perbandingan Jumlah Penderita Diabetes

   ![img alt](https://github.com/UsamahPutraFirdaus/Submission_MLTerapan/blob/main/Submission_1_PredictiveAnalytics/img/perbandingan%20penderita%20diabetes.png?raw=true)

   **Interpretasi Visualisasi Perbandingan Jumlah Penderita Diabetes**
   
   - Mayoritas orang dalam dataset **tidak** menderita diabetes.
   - Namun, sekitar 1 dari 3 orang ternyata menderita diabetes, yang masih cukup signifikan secara proporsi.
   
5. Perbandingan Jumlah Kehamilan paling banyak dengan Paling Sedikit

   ![img alt](https://github.com/UsamahPutraFirdaus/Submission_MLTerapan/blob/main/Submission_1_PredictiveAnalytics/img/distribusi%20kehamilan%20berdasarkan%20kelompok%20usia.png?raw=true)

   **Interpretasi Visualisasi Perbandingan Jumlah Kehamilan Paling Banyak dengan Paling Sedikit**
   
   - Distribusi jumlah kehamilan condong ke angka kecil: Mayoritas orang memiliki kehamilan 0–4 kali.
   - Jumlah kehamilan lebih dari 10 berpotensi sebagai outlier
   
**B. EDA - Multivariete Analysis**
1. Rata-rata Kehamilan Berdasarkan perKelompok Usia

   ![img alt](https://github.com/UsamahPutraFirdaus/Submission_MLTerapan/blob/main/Submission_1_PredictiveAnalytics/img/rata%20rata%20kehamilan%20per%20kelompok%20usia.png?raw=true)

   **Interpretasi Visualisasi Rata-rata Kehamilan Berdasarkan perKelompok Usia**
   
   - Puncak rata-rata kehamilan terjadi pada usia 41–50 tahun, dan menurun setelahnya.
   - Data ini menunjukkan bahwa sebagian besar perempuan mengalami jumlah kehamilan tertinggi di usia pertengahan hingga awal lanjut usia.

2. Jumlah Penderita Diabetes berdasarkan perKelompok Usia

   ![img alt](https://github.com/UsamahPutraFirdaus/Submission_MLTerapan/blob/main/Submission_1_PredictiveAnalytics/img/jumlah%20penderita%20diabetes%20perkelompok%20usia.png?raw=true)

   **Interpretasi Visualisasi Jumlah Penderita Diabetes berdasarkan perKelompok Usia**

   - Jika dilihat langsung dari visualisasi diatas, usia 20 hingga 30 memiliki penderita diabetes yang sangat banyak. Namun jika melihat dari visualisasi Distribusi Jumlah Orang perKelompok Usia, kelompok usia 20 - 30 memiliki jumlah yang sangat banyak dibandingkan dengan kelompok usia lainnya. Kelompok usia 20-30 hanya sekitar -+ 17% penderita diabetes. Dibandingkan dengan kelompok usia 41-50, tingkat penderita diabetes mencapai hampir 55%

3. Matriks Korelasi antar Kolom

   ![img alt](https://github.com/UsamahPutraFirdaus/Submission_MLTerapan/blob/main/Submission_1_PredictiveAnalytics/img/matriks%20korelasi.png?raw=true)
   
   **Interpretasi Visualisasi Jumlah Penderita Diabetes berdasarkan perKelompok Usia**
   - Glucose adalah fitur paling signifikan dalam menentukan kemungkinan diabetes.
   - BMI, Age, dan jumlah kehamilan (Pregnancies) juga berkontribusi tetapi tidak sekuat Glucose.

**C. Indentify Outliers**
Untuk mengatasi outlier, salah satu metode yang umum digunakan adalah metode IQR (Interquartile Range) dengan visualisasi menggunakan boxplot. Berikut penjelasan mengenai metode IQR dan visualisasi boxplot:

1. Apa itu IQR?
Interquartile Range (IQR) adalah selisih antara kuartil ketiga (Q3) dan kuartil pertama (Q1) dari suatu data. Q1 adalah nilai yang membagi 25% data pertama (bagian bawah), sedangkan Q3 adalah nilai yang membagi 75% data (bagian atas). IQR digunakan untuk menggambarkan sebaran nilai yang berada di tengah 50% data.

2. Visualisasi Boxplot
Checking Outliers

Langkah-langkah Deteksi Outlier dengan IQR:
- Hitung Q1 dan Q3
   - Q1 adalah nilai pada persentil ke-25 dari data.
   - Q3 adalah nilai pada persentil ke-75 dari data.
- Hitung IQR
   - IQR = Q3 − Q1
- Tentukan Batas Deteksi Outlier
   - Batas Bawah (Lower Bound) = Q1 − 1.5 × IQR
   - Batas Atas (Upper Bound) = Q3 + 1.5 × IQR
- Identifikasi Outlier
   - Nilai yang lebih kecil dari batas bawah atau lebih besar dari batas atas dikategorikan sebagai outlier, yaitu data yang menyimpang jauh dari nilai mayoritas.
 
   ![img alt](https://github.com/UsamahPutraFirdaus/Submission_MLTerapan/blob/main/Submission_1_PredictiveAnalytics/img/indentify%20outliers.png?raw=true)

   Pada kolom `Pregnancies`, `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`, dan `DiabetesPedigreeFunction` terdeteksi cukup banyak outlier. Sementara itu, kolom `Age` juga menunjukkan indikasi adanya outlier. Namun, setelah ditinjau lebih lanjut, nilai-nilai pada kolom Age masih berada dalam rentang yang wajar sehingga tidak dihapus dari data.

## **Data Preparation**
Pada tahap ini, dilakukan proses transformasi data agar sesuai dan siap digunakan dalam proses pemodelan, berikut adalah beberapa persiapan yang perlu dilakukan sebagai berikut:

### Drop Kolom
Pada tahap ini terdapat kolom yang tidak digunakan dalam modeling yaitu kolom `Id`,  Kolom ini akan dihapus menggunakan fungsi `drop()`.

### Handling Outliers
Karena saat dilakukan pengecekan outlier ditemukan cukup banyak outlier, maka outliers tersebut perlu dihapus agar dataset menjadi lebih bersih dan siap digunakan untuk proses modeling. Berikut adalah tahapan untuk menghapus outliers :

1. Menentukan Kolom yang Akan Dicek
- Pertama, kita tentukan kolom-kolom mana saja dalam data yang berisi angka (data numerik) dan akan diperiksa apakah mengandung outlier atau tidak.

2. Menghitung Kuartil dan IQR Untuk setiap kolom:
   - Hitung Q1 (kuartil pertama) → batas bawah 25% data terendah.
   - Hitung Q3 (kuartil ketiga) → batas atas 25% data tertinggi.
   - Lalu hitung IQR (Interquartile Range), yaitu selisih antara Q3 dan Q1. IQR menunjukkan rentang "normal" dari data tersebut.

3. Menentukan Batas Outlier
- Setelah IQR diketahui, kita tentukan batas bawah dan batas atas dari nilai yang dianggap wajar:
   - Batas bawah = Q1 dikurangi 1.5 × IQR.
   - Batas atas = Q3 ditambah 1.5 × IQR.
   - Nilai yang berada di luar kedua batas ini dianggap sebagai outlier, yaitu data yang terlalu jauh berbeda dari data lainnya.

4. Mengidentifikasi Outlier
- Untuk setiap kolom, kita periksa apakah ada nilai yang berada di luar batas bawah atau batas atas tersebut. Jika ada, maka baris data tersebut dicatat sebagai data yang mengandung outlier.

5. Menggabungkan Semua Outlier
- Setelah seluruh kolom diperiksa, semua baris yang mengandung outlier dari salah satu kolom atau lebih dikumpulkan dalam satu daftar.

6. Menghapus Baris Outlier
- Semua baris data yang mengandung outlier kemudian dihapus dari dataset. Ini dilakukan agar data menjadi lebih bersih dan hasil analisis atau pemodelan tidak terganggu oleh nilai-nilai ekstrem.

   ![img alt](https://github.com/UsamahPutraFirdaus/Submission_MLTerapan/blob/main/Submission_1_PredictiveAnalytics/img/outliers%20setelah%20dihapus.png?raw=true)

   Visualiasi diatas menampilkan hasil setelah outliers dihapus

### Melakukan Split Dataset
Karena fokus prediksi terletak pada variabel Outcome sebagai target untuk menentukan tingkat akurasi dalam mengklasifikasikan apakah seseorang mengalami menderita diabetes atau tidak, maka kolom tersebut dipisahkan dari dataset utama dan disimpan dalam variabel terpisah. Dataset kemudian dibagi menjadi dua bagian: data training yang digunakan untuk melatih model, dan data testing yang digunakan untuk menguji performa model terhadap data yang belum pernah dilihat sebelumnya. Pembagian ini dilakukan dengan rasio 75% untuk training dan 25% untuk testing, menggunakan fungsi train_test_split dari pustaka sklearn.

```Ruby
x = df.drop(["Outcome"],axis=1)
y = df["Outcome"]
x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.25,random_state=42,stratify = y)
```

## Modeling
Pada tahap modeling, dilakukan pemilihan algoritma yang akan digunakan dalam membangun model machine learning, serta proses pengembangan dan pelatihan model tersebut agar dapat dimanfaatkan dalam analisis prediksi. Tiga algoritma berikut akan diuji terlebih dahulu untuk mengevaluasi performa dan menentukan model dengan hasil terbaik, yaitu:

**1. Algoritma Logistic Regression**

  Regresi logistik adalah teknik analisis data yang menggunakan matematika untuk menemukan hubungan antara dua faktor data. Kemudian menggunakan hubungan ini untuk memprediksi nilai dari salah satu faktor tersebut berdasarkan faktor yang lain. Prediksi biasanya memiliki jumlah hasil yang terbatas, seperti ya atau tidak.

Model ini bekerja dengan memodelkan hubungan antara satu atau lebih variabel independen dan variabel dependen biner (dua kelas) menggunakan fungsi logistik (sigmoid). Pada algoritma ini digunakan parameter `solver='lbfgs'` untuk optimasi, `max_iter=500` sebagai batas maksimum iterasi, dan `random_state=42` untuk memastikan hasil model tetap konsisten.
```Ruby
lr_model = LogisticRegression(solver='lbfgs', max_iter=500, random_state=42)
```

**2. Algoritma Decision Tree**

  Decision trees adalah algoritme pembelajaran yang diawasi dan bersifat non-parametrik, yang digunakan untuk tugas klasifikasi dan regresi. Memiliki struktur pohon hierarkis, yang terdiri dari simpul akar, cabang, simpul internal dan simpul daun.

Decision Tree membagi data berdasarkan fitur yang paling baik memisahkan kelas target menggunakan metrik seperti Gini Impurity. Pada implementasinya, model ini menggunakan parameter `max_depth=5` untuk menghindari overfitting,` criterion='gini'` sebagai metode pemilihan split, dan `random_state=42` untuk menjaga konsistensi hasil.
```Ruby
dt_model = DecisionTreeClassifier(max_depth=5, criterion='gini', random_state=42)
```

**3. Algoritma Random Forest**

  Random Forest adalah algoritma dalam machine learning yang digunakan untuk pengklasifikasian dataset. Karena fungsinya bisa digunakan untuk banyak dimensi dengan berbagai skala dan performa yang tinggi. Klasifikasi ini dilakukan melalui penggabungan tree dalam decision tree dengan cara training dataset.

Random Forest merupakan algoritma ensemble learning yang menggabungkan banyak Decision Tree untuk meningkatkan akurasi dan stabilitas prediksi. Model ini membangun beberapa pohon keputusan dan menggabungkan hasil prediksinya. Parameter yang digunakan adalah `n_estimators=50` (jumlah pohon), `max_depth=12` (kedalaman maksimum pohon), `random_state=42` untuk replikasi hasil, dan `n_jobs=-1` yang berarti proses training dilakukan secara paralel menggunakan seluruh core CPU.
```Ruby
rf_model = RandomForestClassifier(n_estimators=50, max_depth=12, random_state=42, n_jobs=-1)
```

## Evaluation
Metrik evaluasi yang digunakan dalam proyek ini ialah sebagai berikut:

### A. Classification Report

   ![img alt](https://github.com/UsamahPutraFirdaus/Submission_MLTerapan/blob/main/Submission_1_PredictiveAnalytics/img/tp_tn_fp_fn.png?raw=true)

Terdapat 4 label pada matriks confusion seperti yang terlihat di gambar, yaitu TP, TN, FP, dan FN. a. True Positive (TP) merupakan jumlah data pada positif yang ditebak dengan benar. b. True Negative (TN) merupakan jumlah data pada negatif yang ditebak dengan benar. c. False Positive (FP) merupakan jumlah data yang ditebak dengan salah karena diprediksi positif, sedangkan aslinya adalah negatif. d. False Negative (FN) merupakan jumlah data yang ditebak dengan salah karena diprediksi negatif, sedangkan aslinya adalah positif.

1. Precision

   ![img alt](https://github.com/UsamahPutraFirdaus/Submission_MLTerapan/blob/main/Submission_1_PredictiveAnalytics/img/precision.png?raw=true)

   Dari seluruh prediksi positif, berapa yang benar-benar positif.

2. Recall
   
   ![img alt](https://github.com/UsamahPutraFirdaus/Submission_MLTerapan/blob/main/Submission_1_PredictiveAnalytics/img/recal.png?raw=true)

   Dari semua kasus positif aktual, berapa banyak yang berhasil diprediksi dengan benar.

3. F1-Score

   ![img alt](https://github.com/UsamahPutraFirdaus/Submission_MLTerapan/blob/main/Submission_1_PredictiveAnalytics/img/f1.png?raw=true)

   Harmoni antara precision dan recall. Semakin tinggi, semakin baik keseimbangan keduanya.

4. Average (Macro vs Weighted)

   ![img alt](https://github.com/UsamahPutraFirdaus/Submission_MLTerapan/blob/main/Submission_1_PredictiveAnalytics/img/avg.png?raw=true)

   - Macro Average : Menghitung metrik (precision, recall, f1) secara rata-rata antar kelas, tanpa memperhatikan jumlah sampel (support) pada tiap kelas.
   - Weighted Average : Sama seperti macro, tetapi diberi bobot sesuai jumlah sampel (support) di tiap kelas.

5. Hasil Classficiation Report

   ![img alt](https://github.com/UsamahPutraFirdaus/Submission_MLTerapan/blob/main/Submission_1_PredictiveAnalytics/img/akurasi.png?raw=true)

  - Logistic Regression: Akurasi 78.2%, performa terendah dari ketiga model, dengan recall kelas 1 hanya 51.9% yang menunjukkan banyak false negative.
  - Decision Tree: Akurasi 84.5%, lebih baik dari Logistic Regression dengan F1-Score kelas 1 sebesar 71.7%, menandakan model ini cukup seimbang.
  - Random Forest: Akurasi tertinggi 99.1%, dengan precision dan recall kelas 1 nyaris sempurna (1.000 dan 0.973), menunjukkan performa sangat baik dan overfitting bisa menjadi perhatian.

Kesimpulan: Random Forest memberikan performa terbaik secara keseluruhan, namun perlu dipastikan bahwa model tidak overfit terhadap data pelatihan. Inilah model yang paling andal berdasarkan metrik yang ditampilkan.
   
### B. Confusion Matrix

   ![img alt](https://github.com/UsamahPutraFirdaus/Submission_MLTerapan/blob/main/Submission_1_PredictiveAnalytics/img/Confusion_Matrix.png?raw=true)

Berdasarkan hasil Confusion Matrix dari ketiga model yang diuji, model Random Forest menunjukkan performa terbaik, dengan hanya 5 kesalahan prediksi pada kelas positif (1), dan tidak ada kesalahan pada kelas negatif (0). Sebaliknya, model Logistic Regression memberikan hasil terburuk, dengan 40 kesalahan prediksi pada kelas negatif (0) dan 88 kesalahan pada kelas positif (1). Hal ini menunjukkan bahwa Logistic Regression kurang efektif dalam mendeteksi penderita diabetes

### Perbandingan Hasil Evaluasi Model

Dari hasil kedua skema evaluasi model (Classification Report dan Confusion Matrix) hasil menunjukkan bahwa Random Forest memberikan hasil performa terbaik. Dimana akurasi mencapai 99.1% dan kesalahan prediksi hanya 5 kesalahan pada kelas positif dan tidak ada kesalahan prediksi pada kelas negatif.

## **Kesimpulan**
Berdasarkan serangkaian proses analisis data dan pembangunan model machine learning terhadap dataset diabetes dari Kaggle, diperoleh beberapa poin utama sebagai berikut:
1. Fitur Penting yang Mempengaruhi Diabetes
Berdasarkan analisis korelasi dan visualisasi, fitur Glucose, BMI, dan Age memiliki pengaruh paling signifikan terhadap status diabetes seseorang. Fitur Glucose menempati posisi tertinggi dalam korelasi dengan target variabel.

2. Distribusi Usia Penderita Diabetes
Analisis usia menunjukkan bahwa kelompok usia 41–50 tahun merupakan kelompok dengan jumlah penderita diabetes terbanyak. Hal ini mengindikasikan pentingnya pencegahan dan deteksi dini di kelompok usia tersebut.

3. Perbandingan Model
Tiga model telah dibangun dan dibandingkan, yaitu:
   - Logistic Regression
   - Decision Tree
   - Random Forest
Pada dataset ini model Random Forest menghasilkan kesalahan prediksi terendah berdasarkan analisis confusion matrix, menjadikannya model dengan performa terbaik di antara ketiganya.

4. Model Terbaik untuk Prediksi Diabetes
Dengan hasil confusion matrix model Random Forest menunjukkan jumlah kesalahan paling sedikit, serta mendapatkan akurasi paling tinggi dibandingkan dengan model lainnya. Hal ini menunjukkan bahwa Model Random Forest merupakan model yang paling baik performanya dalam mendeteksi Penderita Diabetes


# Reference:
[1] World Health Organization, "Diabetes", 2024. https://www.who.int/news-room/fact-sheets/detail/diabetes (accessed May. 8, 2025)
