# Laporan Proyek Machine Learning - Auliya Sabrina Vyantika

## Domain Proyek

erkembangan teknologi informasi berbasis komputer telah mempengaruhi berbagai sektor, termasuk dalam dunia perbankan. Salah satu permasalahan yang sering dihadapi oleh Bank ABC adalah kesalahan dalam pengambilan keputusan pemberian kredit. Hal ini dapat mengakibatkan kerugian bagi pihak bank jika kredit yang diberikan tidak dapat dibayar kembali oleh nasabah, atau jika terjadi kredit macet. Untuk itu, bank perlu mengevaluasi kelayakan nasabah dalam memenuhi kewajiban pembayaran pinjaman sebelum memberikan persetujuan kredit.

Masalah ini dapat diatasi dengan menggunakan pendekatan machine learning, khususnya dengan metode klasifikasi. Algoritma klasifikasi seperti decision tree, random forest, support vector machine, dan lainnya yang dapat digunakan untuk menganalisis data nasabah, seperti status pembayaran cicilan kredit sebelumnya. Setiap algoritma memiliki keunggulan tersendiri dalam menangani jenis data yang berbeda dan memprediksi apakah nasabah berpotensi mengalami kesulitan dalam pembayaran pinjaman. Dengan membandingkan kinerja berbagai algoritma, bank dapat memilih model yang paling tepat untuk meminimalkan risiko kesalahan dalam pengambilan keputusan kredit, sehingga dapat memberikan keputusan yang lebih akurat dan efisien. Proses evaluasi kredit yang lebih berbasis data ini dapat mengurangi risiko kerugian akibat kredit macet.
  
  [PENERAPAN METODE PERBANDINGAN EKSPONENSIAL PADA SISTEM PENDUKUNG KEPUTUSAN PEMBERIAN KREDIT PADA BANK XYZ](http://jurnal.borneo.ac.id/index.php/borneo_saintek/article/view/911)

  [Perkembangan Teknologi Informasi Terhadap Peningkatan Bisnis Online](http://interdisiplin.my.id/index.php/i/article/view/5)


## Business Understanding

Pada bagian ini, akan dibahas mengenai klarifikasi masalah yang ada serta tujuan dan solusi yang dapat dicapai dengan menggunakan pendekatan machine learning dalam proses pengambilan keputusan pemberian kredit di Bank ABC.

### Problem Statements

1. Bank ABC sering menghadapi kesalahan dalam pengambilan keputusan pemberian kredit kepada nasabah, yang dapat mengakibatkan kredit macet dan kerugian finansial. 
2. Tidak adanya sistem yang efisien dan berbasis data untuk mengevaluasi kelayakan nasabah dalam membayar kembali pinjaman, sehingga keputusan pemberian kredit kurang tepat.
3. Kurangnya pengolahan data historis yang relevan, seperti riwayat pembayaran cicilan kredit, yang dapat digunakan untuk memprediksi kemampuan nasabah dalam memenuhi kewajiban kreditnya.

### Goals

1. Menerapkan model machine learning untuk menganalisis data nasabah secara lebih akurat dan mengurangi kesalahan dalam pengambilan keputusan pemberian kredit, sehingga mengurangi risiko kredit macet.
2. Mengembangkan sistem berbasis machine learning yang dapat mengevaluasi kelayakan nasabah dalam membayar pinjaman secara otomatis dan efisien, dengan mempertimbangkan data historis dan karakteristik nasabah.
3. Menggunakan algoritma klasifikasi seperti decision tree, random forest, dan support vector machine untuk menganalisis data nasabah dan memprediksi kemungkinan terjadinya kredit macet berdasarkan riwayat pembayaran cicilan sebelumnya.

### Solution Statements

1. Menggunakan beberapa algoritma klasifikasi seperti decision tree, random forest, dan support vector machine untuk membandingkan kinerjanya dalam memprediksi nasabah yang berpotensi mengalami kesulitan pembayaran. Model yang memberikan akurasi tertinggi akan dipilih untuk diterapkan pada sistem evaluasi kredit.
2. Melakukan tuning hyperparameter pada model baseline yang telah diterapkan untuk meningkatkan kinerja model. Dengan teknik seperti grid search atau random search, parameter yang optimal dapat ditemukan untuk meningkatkan akurasi dan performa model dalam memprediksi kredit macet.

### Metrik Evaluasi

- **Accuracy**: Untuk mengukur seberapa banyak prediksi yang benar dibandingkan dengan total prediksi yang dibuat oleh model.
- **Precision**: Mengukur seberapa tepat model dalam memprediksi nasabah yang akan mengalami kredit macet (positif).
- **Recall**: Mengukur seberapa baik model dalam menangkap semua nasabah yang berpotensi mengalami kredit macet.
- **F1-Score**: Menyediakan keseimbangan antara precision dan recall, untuk memastikan bahwa model tidak hanya tepat, tetapi juga tidak melewatkan banyak nasabah yang berisiko.


## Data Understanding

Dataset yang digunakan berfokus pada data nasabah bank, yang mencakup informasi terkait status pembayaran kredit dan berbagai karakteristik lainnya yang berhubungan dengan kelayakan pemberian kredit. Dataset ini digunakan untuk menganalisis dan memprediksi kemungkinan kredit macet berdasarkan data historis nasabah.

Dataset yang digunakan dalam proyek ini dapat diunduh melalui [GitHub](https://github.com/Salmanab16/kredit-macet). Dataset ini terdiri dari beberapa fitur, seperti status pembayaran kredit sebelumnya, umur, penghasilan, jumlah pinjaman, dan lainnya. Dengan data ini, model machine learning akan dilatih untuk memprediksi apakah seorang nasabah berpotensi mengalami kredit macet atau tidak.

Data ini dapat digunakan untuk pelatihan model klasifikasi dengan berbagai algoritma seperti decision tree, random forest, dan support vector machine, untuk membantu pihak bank dalam pengambilan keputusan pemberian kredit yang lebih akurat dan efisien.

### Variabel-variabel pada kredit-macet dataset adalah sebagai berikut:
- **`jenis_kelamin`**: Jenis kelamin nasabah yang terdiri dari dua kategori, yaitu P (Perempuan) dan L (Laki-laki).
- **`umur`**: Usia nasabah yang menunjukkan usia saat aplikasi kredit diajukan.
- **`jml_pinjaman`**: Jumlah pinjaman yang diajukan oleh nasabah.
- **`jkw`**: Jangka waktu pinjaman dalam satuan bulan.
- **`jml_angsuran_per_bulan`**: Jumlah angsuran yang harus dibayar nasabah setiap bulan.
- **`type_pinjaman`**: Tipe pinjaman yang diberikan, seperti konsumsi, investasi, atau lainnya.
- **`jenis_pinjaman`**: Jenis pinjaman yang diajukan nasabah, misalnya pinjaman untuk rumah, kendaraan, atau pendidikan.
- **`bi_sektor_ekonomi`**: Kode sektor ekonomi yang digunakan oleh Bank Indonesia (BI) untuk mengklasifikasikan sektor ekonomi nasabah.
- **`bi_golongan_debitur`**: Golongan debitur yang diberikan oleh Bank Indonesia berdasarkan tingkat risiko.
- **`bi_gol_penjamin`**: Golongan penjamin yang digunakan oleh Bank Indonesia untuk menilai jenis jaminan yang diberikan.
- **`saldo_nominatif`**: Saldo nominatif nasabah, yang menunjukkan saldo utang yang terdaftar pada bank.
- **`tunggakan_pokok`**: Tunggakan pokok yang harus dibayar oleh nasabah, yaitu jumlah pokok yang tertunggak dalam pembayaran.
- **`tunggakan_bunga`**: Tunggakan bunga yang harus dibayar oleh nasabah, yaitu bunga yang belum dibayar hingga saat ini.
- **`status kredit`**: Status kredit nasabah, apakah nasabah tersebut lancar atau mengalami masalah seperti kredit macet.

Variabel-variabel ini digunakan untuk menganalisis dan memprediksi kemungkinan terjadinya kredit macet pada nasabah, yang akan membantu bank dalam pengambilan keputusan pemberian kredit yang lebih tepat dan efisien.

### Exploratory Data Analysis (EDA)

Pada tahap awal, dilakukan beberapa analisis untuk memahami data lebih dalam. Hasil analisis menunjukkan beberapa hal penting sebagai berikut:

1. **Ukuran Dataset**:
   Dataset terdiri dari 766 baris data dengan 16 kolom.

   ![Deskripsi Gambar](https://drive.google.com/uc?export=view&id=1SHY3ez8cRwliTN2zxmZjb8_KYGI3VWF0)

2. **Mengetahui Type Data, Missing Value, dan Data Duplikat pada Dataset**: 

   **Type Data**
   
    ![Deskripsi Gambar](https://drive.google.com/uc?export=view&id=1hSDu0oX_Blc2SaWKQDFJ5JbocZUjpc-3)

   **Missing Value**
   
   ![Deskripsi Gambar](https://drive.google.com/uc?export=view&id=12IfgPNvex-D4zPsHUMHEkTcmI-PQeW86)

   **Duplikat**

   ![Deskripsi Gambar](https://drive.google.com/uc?export=view&id=1dOHsBLHN3h-zAV2Eof3Cg7fw-iIJ3V-c)
   
   Tidak terdapat data yang duplikat. Hanya saja, beberapa kolom memiliki nilai yang hilang (missing values). 

    - Kolom `umur` memiliki 9 nilai yang hilang.
    - Kolom `jkw` memiliki 8 nilai yang hilang.
    - Kolom `bi_sektor_ekonomi` memiliki 1 nilai yang hilang.

    Setelah dilakukan penanganan, dataset memiliki 753 baris data yang lengkap tanpa nilai yang hilang.

    ![Deskripsi Gambar](https://drive.google.com/uc?export=view&id=1DyZpdZEGjgn_LYyl6lgCWkZl6czGRHbK)

4. **Kolom dengan Tipe Data Tidak Tepat**:

   Terdapat beberapa kolom yang memiliki tipe data object padahal seharusnya menggunakan tipe data category agar lebih efisien, yaitu:

    ![Deskripsi Gambar](https://drive.google.com/uc?export=view&id=1p0_4eon3xm9rvwAX50m62j62lDnl9djs)

    - Kolom jenis_kelamin, yang memiliki nilai unik seperti 'P', 'L', 'WANITA', 'LAKI-LAKI', 'PRIA', 'PEREMPUAN'.
    - Kolom status kredit, yang memiliki dua kategori: 'MACET' dan 'LANCAR'.
    
    Setelah perbaikan tipe data menjadi category, kolom-kolom ini menjadi :
   
   ![Deskripsi Gambar](https://drive.google.com/uc?export=view&id=1OO2HjEWGPLvIQqT_0gM8VFm9TTuOOMpH)

   **Type Data Terbaru**
   
   ![Deskripsi Gambar](https://drive.google.com/uc?export=view&id=1ZgsF8NDyoqyf1pOG7Jdo05rxuNY7v5fd)


## **Data Preparation**

Pada bagian ini, dilakukan beberapa tahapan untuk mempersiapkan data yang digunakan dalam model. Berikut adalah langkah-langkah data preparation yang diterapkan:

1. **Pengecekan Struktur Data**
   - Dilakukan pengecekan jumlah baris dan kolom pada dataset untuk mengetahui dimensi data.
   - Memeriksa informasi tipe data dan jumlah nilai non-null pada setiap kolom.

2. **Pengecekan Duplikasi**
   - Mengecek apakah terdapat data yang duplikat dalam dataset.
   
3. **Pengecekan Missing Values**
   - Memeriksa jumlah nilai yang hilang (missing values) pada setiap kolom untuk memastikan data bersih dan siap digunakan.

4. **Penghapusan Missing Values**
   - Baris dengan missing values dihapus dari dataset untuk memastikan bahwa model tidak terpengaruh oleh data yang hilang.

5. **Deskriptif Statistik dan Tipe Data**
   - Melakukan eksplorasi deskriptif untuk memahami distribusi dan statistik dasar setiap kolom.
   - Mengidentifikasi kolom yang memiliki tipe data yang tidak sesuai, seperti kolom `jenis_kelamin` dan `status kredit` yang seharusnya bertipe kategori, bukan `object`.

6. **Perubahan Tipe Data**
   - Mengubah tipe data kolom `jenis_kelamin` dan `status kredit` menjadi tipe kategori untuk meningkatkan efisiensi pemrosesan.

7. **Pembersihan Nilai Kolom `jenis_kelamin`**
   - Melakukan standarisasi nilai pada kolom `jenis_kelamin` dengan mengganti beberapa representasi nilai (seperti "WANITA" menjadi "P" dan "LAKI-LAKI" menjadi "L") agar konsisten.

8. **Label Encoding**
   - Menerapkan Label Encoding pada kolom `jenis_kelamin` untuk mengubah nilai kategori menjadi representasi numerik.

9. **Pemisahan Fitur dan Target**
   - Memisahkan dataset menjadi dua bagian: fitur (X) yang akan digunakan untuk melatih model dan target (y) yang merupakan variabel yang diprediksi.

10. **Normalisasi Data**
    - Menggunakan MinMaxScaler untuk melakukan normalisasi pada kolom-kolom numerik agar skala data menjadi seragam.


## **Modeling**

Pada tahap ini, beberapa algoritma machine learning diterapkan untuk menyelesaikan masalah klasifikasi berdasarkan dataset yang telah diproses. Berikut adalah algoritma yang digunakan beserta tahapan dan parameter yang diterapkan:

### **1. K-Nearest Neighbors (KNN)**
- **Kelebihan**:
  - Mudah dipahami dan diimplementasikan.
  - Tidak memerlukan asumsi distribusi data tertentu.
  - Sangat efektif untuk dataset kecil hingga menengah.
- **Kekurangan**:
  - Performa menurun pada dataset yang sangat besar atau memiliki dimensi tinggi.
  - Sensitif terhadap noise dalam data.
- **Proses Pemodelan**:
  - Menggunakan **jumlah tetangga (k)** yang berbeda untuk mencari nilai terbaik.
  - Parameter lainnya yang digunakan adalah **metrik jarak** seperti Euclidean.

### **2. Decision Tree (DT)**
- **Kelebihan**:
  - Mudah diinterpretasikan dan divisualisasikan.
  - Dapat menangani data numerik dan kategori.
- **Kekurangan**:
  - Rentan terhadap overfitting pada data dengan banyak fitur.
- **Proses Pemodelan**:
  - Menggunakan parameter **maksimum kedalaman pohon (max_depth)** untuk membatasi kedalaman dan mencegah overfitting.
  - Parameter **min_samples_split** dan **min_samples_leaf** digunakan untuk mengontrol pembentukan pohon.

### **3. Random Forest (RF)**
- **Kelebihan**:
  - Dapat menangani overfitting dengan baik.
  - Akurat dan stabil.
- **Kekurangan**:
  - Model lebih kompleks dan membutuhkan lebih banyak waktu untuk pelatihan.
- **Proses Pemodelan**:
  - Menggunakan beberapa **decision trees** dengan parameter **n_estimators** untuk menentukan jumlah pohon yang akan digunakan.
  - Parameter **max_features** digunakan untuk mengontrol fitur yang dipilih setiap pohon.

### **4. Support Vector Machine (SVM)**
- **Kelebihan**:
  - Sangat efektif pada data dengan margin pemisahan yang jelas.
  - Efektif untuk dimensi tinggi.
- **Kekurangan**:
  - Memerlukan pemilihan kernel yang tepat.
  - Pelatihan pada dataset besar bisa memakan waktu.
- **Proses Pemodelan**:
  - Menggunakan parameter **kernel** (linear, polynomial, atau radial basis function - RBF).
  - Parameter **C** digunakan untuk mengontrol trade-off antara margin dan kesalahan.

### **5. Naive Bayes (NB)**
- **Kelebihan**:
  - Cepat dan efisien pada dataset besar.
  - Sangat baik untuk data yang memiliki fitur independen.
- **Kekurangan**:
  - Tidak efektif jika ada ketergantungan antar fitur.
- **Proses Pemodelan**:
  - Menggunakan probabilitas untuk setiap fitur dalam klasifikasi.
  - Model ini mengasumsikan bahwa setiap fitur independen.


## **Evaluation**

Pada bagian ini, kami menggunakan beberapa metrik evaluasi untuk mengukur performa setiap model yang diterapkan. Metrik yang digunakan adalah **Akurasi, Precision, Recall, dan F1-Score**. Berikut adalah penjelasan masing-masing metrik:

### **Akurasi (Accuracy)**
Akurasi mengukur proporsi prediksi yang benar dibandingkan dengan jumlah total prediksi. Formula yang digunakan adalah:

![Deskripsi Gambar](https://drive.google.com/uc?export=view&id=1AAWJjKG7NqVnf3HRuZxld7NwIEVsUd6m)

Akurasi memberikan gambaran umum tentang seberapa baik model dalam mengklasifikasikan data, namun tidak selalu memberikan gambaran yang baik ketika ada ketidakseimbangan kelas.

### **Precision**
Precision mengukur proporsi prediksi positif yang benar dibandingkan dengan total prediksi positif yang dilakukan oleh model. Formula yang digunakan adalah:

![Deskripsi Gambar](https://drive.google.com/uc?export=view&id=1z1RcIFZh6Db9BYSfDNpYf2A_FGezZV8_)

Di mana:
- **TP** (True Positives) adalah jumlah prediksi positif yang benar.
- **FP** (False Positives) adalah jumlah prediksi positif yang salah.

Precision sangat penting ketika false positives harus dihindari, seperti dalam deteksi penipuan.

### **Recall**
Recall mengukur proporsi kasus positif yang berhasil dideteksi oleh model dibandingkan dengan total kasus positif yang ada. Formula yang digunakan adalah:

![Deskripsi Gambar](https://drive.google.com/uc?export=view&id=1UmLqvYMJOLmwRMp4DiJX8-RZV36KB8mN)

Di mana:
- **FN** (False Negatives) adalah jumlah prediksi negatif yang salah.

Recall sangat penting ketika kita ingin meminimalkan jumlah false negatives, seperti dalam deteksi penyakit atau ancaman keamanan.

### **F1-Score**
F1-Score adalah rata-rata harmonis dari Precision dan Recall. Formula yang digunakan adalah:

![Deskripsi Gambar](https://drive.google.com/uc?export=view&id=1e4sd5FTj-d6RPck_4LaS7aESApWPuFpg)

F1-Score memberikan gambaran yang lebih baik tentang keseimbangan antara Precision dan Recall, terutama ketika ada ketidakseimbangan antara keduanya.

---

### **Hasil Evaluasi Model**

Berdasarkan hasil evaluasi, berikut adalah analisis performa masing-masing model:

![Deskripsi Gambar](https://drive.google.com/uc?export=view&id=1ySL0IDeuTiqq-xHhUZ9XvkX0OWeWZpJJ)

### **Analisis Hasil Evaluasi**
- **K-Nearest Neighbors (KNN)**: Memiliki akurasi **90.07%**, precision **94.59%**, recall **92.11%**, dan F1-score **93.33%**. Model ini stabil namun tidak sebaik model lainnya.
- **Decision Tree (DT)**: Menunjukkan akurasi **97.35%** dan F1-score **98.25%**, sangat baik dalam klasifikasi.
- **Random Forest (RF)**: Mencatatkan performa terbaik dengan **akurasi 98.01%**, recall **100%**, dan F1-score **98.70%**. Model ini paling stabil dan efektif.
- **Support Vector Machine (SVM)**: Meskipun memiliki recall **100%**, akurasi dan precision cukup rendah (**75.50%**).
- **Naive Bayes (NB)**: Memiliki akurasi terendah (**64.90%**) dengan precision yang tinggi (**94.20%**) namun recall yang rendah (**57.02%**).

### **Kesimpulan**
- **Random Forest (RF)** adalah model terbaik berdasarkan **akurasi**, **recall**, dan **F1-score** yang sangat baik.
- **Decision Tree (DT)** dapat menjadi pilihan yang lebih sederhana namun tetap efektif.
- **SVM** cocok jika recall menjadi prioritas utama meskipun akurasi keseluruhan lebih rendah.
- **Naive Bayes (NB)** dapat dipilih jika **precision** lebih diutamakan, meskipun memiliki kekurangan pada recall.

---

### **Rekomendasi**
- Jika mengutamakan **akurasi dan kestabilan klasifikasi**, pilihlah **Random Forest (RF)**.
- **Decision Tree (DT)** bisa dipilih sebagai alternatif yang lebih sederhana.
- **SVM** dapat digunakan jika lebih memprioritaskan **recall**, meskipun dengan akurasi lebih rendah.
- **Naive Bayes (NB)** lebih cocok jika precision menjadi prioritas utama meskipun ada trade-off dengan recall.

