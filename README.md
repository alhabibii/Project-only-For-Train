# Laporan Proyek Machine Learning - Auliya Sabrina Vyantika

## Project Overview

Dalam era digital saat ini, jumlah buku yang tersedia secara daring sangatlah besar. Platform seperti Amazon menyajikan jutaan judul buku dari berbagai genre, penulis, dan tahun terbit. Hal ini menciptakan tantangan tersendiri bagi pengguna dalam menemukan buku yang relevan dan sesuai dengan preferensi mereka. Oleh karena itu, sistem rekomendasi buku menjadi komponen penting dalam meningkatkan pengalaman pengguna dan efisiensi pencarian informasi.

Dataset yang digunakan dalam proyek ini berasal dari Amazon Web Services, yang mencakup informasi penting mengenai buku seperti ISBN, judul buku, penulis, tahun terbit, dan penerbit. ISBN yang tidak valid telah dihapus, sehingga data yang tersedia sudah bersih dari kesalahan identifikasi. Selain itu, tersedia juga tautan ke gambar sampul buku dalam tiga ukuran berbeda, yang mengarah langsung ke situs Amazon.

Permasalahan yang diangkat dalam proyek ini adalah bagaimana memanfaatkan informasi konten buku yang tersedia untuk membangun sistem yang mampu memahami karakteristik buku dan mengelompokkan atau merekomendasikan buku dengan cara yang relevan. Dengan kata lain, proyek ini bertujuan untuk mengimplementasikan pendekatan berbasis konten (content-based filtering) atau clustering untuk mengelompokkan buku berdasarkan kesamaan atribut.

Menurut Ricci, Rokach, dan Shapira (2011), sistem rekomendasi memainkan peran penting dalam berbagai domain e-commerce dan informasi digital karena dapat membantu mengurangi kelebihan informasi (information overload) dan meningkatkan kepuasan pengguna. Oleh karena itu, membangun model rekomendasi atau analisis berbasis atribut buku menjadi solusi strategis yang perlu ditelusuri lebih lanjut.

> **Referensi**:  
> Ricci, F., Rokach, L., & Shapira, B. (2011). *Introduction to Recommender Systems Handbook*. Springer. https://doi.org/10.1007/978-0-387-85820-3

---

## Business Understanding

Pada bagian ini dilakukan proses klarifikasi terhadap permasalahan yang ingin diselesaikan dalam proyek. Dataset yang digunakan mencakup informasi mengenai buku (ISBN, judul, penulis, tahun terbit, penerbit, dan URL gambar), serta data pengguna dan rating yang diberikan terhadap buku. Informasi ini sangat berpotensi untuk digunakan dalam membangun sistem rekomendasi yang dapat membantu pengguna menemukan buku yang relevan sesuai preferensi mereka.

### Problem Statements

- **Pernyataan Masalah 1**  
  Pengguna mengalami kesulitan dalam menemukan buku yang sesuai dengan preferensi mereka karena jumlah pilihan buku yang sangat banyak di platform daring.

- **Pernyataan Masalah 2**  
  Belum tersedia sistem rekomendasi yang memanfaatkan data atribut konten buku maupun interaksi pengguna untuk menyarankan buku yang relevan.

- **Pernyataan Masalah 3**  
  Informasi dan metadata buku yang tersedia belum dimanfaatkan secara optimal untuk pengelompokan atau analisis rekomendasi yang lebih personal.

### Goals

- **Jawaban Pernyataan Masalah 1**  
  Membangun sistem rekomendasi buku yang dapat membantu pengguna menemukan buku yang sesuai dengan minat mereka menggunakan pendekatan *Content-based Filtering* dan *Collaborative Filtering*.

- **Jawaban Pernyataan Masalah 2**  
  Mengimplementasikan pendekatan *Content-based Filtering* untuk merekomendasikan buku berdasarkan kemiripan atribut konten seperti penulis, tahun terbit, dan judul.

- **Jawaban Pernyataan Masalah 3**  
  Menggunakan pendekatan *Collaborative Filtering* untuk merekomendasikan buku berdasarkan interaksi pengguna, seperti rating yang diberikan oleh pengguna lain dengan preferensi serupa.

---

## Data Understanding

Dataset yang digunakan dalam proyek machine learning ini adalah Book Recommendation System 📚📚 yang diperoleh dari situs [Kaggle](https://www.kaggle.com/code/fahadmehfoooz/book-recommendation-system/input). Dataset ini terdiri dari tiga file utama, yaitu Books.csv, Ratings.csv, dan Users.csv. Dataset ini memuat informasi buku, pengguna, serta rating yang diberikan, yang sangat relevan untuk pengembangan sistem rekomendasi berbasis pembelajaran mesin.

![BOOK](https://github.com/user-attachments/assets/964b165d-e1e4-48cf-a2b6-9b585f74b0f9)

### 1. `Books.csv`
- **Ukuran Data**: 271.360 baris, 8 kolom

#### ⚠️ Kondisi Data:
- Kolom `Book-Author`, `Publisher`, dan `Image-URL-L` memiliki **missing values**.
- Kolom `Year-Of-Publication` bertipe `object`, seharusnya bertipe numerik → perlu pembersihan dan konversi tipe data.
- Potensi **duplikat** ISBN atau buku dengan ejaan berbeda.
- Kemungkinan terdapat **outlier** pada tahun terbit (misal < 1000 atau > 2025).

![RATINGS](https://github.com/user-attachments/assets/1abe8cf0-83b5-439f-84f6-13975ed9ef56)

### 2. `Ratings.csv`
- **Ukuran Data**: 1.149.780 baris, 3 kolom

#### ⚠️ Kondisi Data:
- Tidak terdapat **missing value**.
- Tidak ditemukan **duplikat baris** saat inspeksi awal.
- Nilai `Book-Rating` = 0 biasanya berarti **tidak memberikan rating nyata** → bisa dianggap *noise* dalam analisis preferensi pengguna.

![USER](https://github.com/user-attachments/assets/2fded83a-3db6-401d-88a6-910570eddf8e)

### 3. `Users.csv`
- **Ukuran Data**: 278.858 baris, 3 kolom

#### ⚠️ Kondisi Data:
- Kolom `Age` hanya memiliki **168.096 nilai non-null**, sisanya kosong.
- Terdapat **outlier** pada `Age`, seperti usia 0 atau > 100 → perlu dibersihkan.
- Kolom `Location` berformat **tidak konsisten**, mengandung gabungan kota, negara, dan kode pos → perlu normalisasi jika dianalisis berdasarkan lokasi.

### Variabel-variabel dalam dataset:

#### Books.csv
- `ISBN` : Kode unik identifikasi untuk buku
- `Book-Title` : Judul buku
- `Book-Author` : Nama penulis buku (hanya penulis pertama jika lebih dari satu)
- `Year-Of-Publication` : Tahun buku diterbitkan
- `Publisher` : Nama penerbit buku
- `Image-URL-S` : URL gambar sampul ukuran kecil
- `Image-URL-M` : URL gambar sampul ukuran sedang
- `Image-URL-L` : URL gambar sampul ukuran besar

#### Users.csv
- `User-ID` : Identifikasi unik untuk pengguna
- `Location` : Lokasi geografis pengguna (biasanya dalam format City, State, Country)
- `Age` : Umur pengguna (dapat mengandung nilai kosong atau tidak masuk akal)

#### Ratings.csv
- `User-ID` : Identifikasi pengguna yang memberikan rating
- `ISBN` : ISBN dari buku yang diberi rating
- `Book-Rating` : Nilai rating yang diberikan pengguna (rentang 0–10, dengan 0 berarti tidak ada rating eksplisit)

### Exploratory Data Analysis (EDA)

Beberapa tahapan eksplorasi dilakukan terhadap dataset ini, di antaranya:

### Exploratory Data Analysis (EDA)

Beberapa tahapan eksplorasi dilakukan terhadap dataset ini, di antaranya:

![Deskripsi Gambar](https://drive.google.com/uc?export=view&id=1d3MdhtgxVtSkXVvyBIga7NITEOTR9CFd)

**Jumlah Data Buku**: 
  Terdapat total 271,360 data buku, yang masing-masing memiliki ISBN unik.

**Jumlah Buku Unik berdasarkan ISBN**: 
  Jumlah buku unik berdasarkan ISBN adalah 271,360, menunjukkan bahwa setiap buku memiliki identifikasi yang unik.

**Jumlah Penulis Unik**:
  Terdapat 102,022 penulis unik dalam dataset ini, yang menunjukkan keragaman dalam jumlah penulis buku.

**Jumlah Penerbit Unik**:
  Dataset ini mencakup 16,807 penerbit unik yang turut mempublikasikan buku-buku dalam sistem.

**Missing Values**:
  Terdapat beberapa kolom dengan missing values, di antaranya:
  - `Book-Author`: 2 missing values
  - `Publisher`: 2 missing values
  - `Image-URL-L`: 3 missing values
  Kolom lainnya tidak mengandung nilai yang hilang.

**Kualitas Data**:
  Kolom ISBN, Book-Title, Year-Of-Publication, dan Image-URL-S/M tidak memiliki missing values, sehingga data pada kolom-kolom tersebut cukup lengkap.

![Deskripsi Gambar](https://drive.google.com/uc?export=view&id=1fypKiH-vTV_8arYeAgWj6k-FhtYHN7Ke)

**Jumlah Data Rating**: 
  Terdapat 1.149.780 data rating yang diberikan oleh pengguna.

**Rata-Rata Rating**: 
  Rata-rata rating (mean) yang diberikan adalah sekitar 2.87 dengan standar deviasi sebesar 3.85, menunjukkan penyebaran rating yang cukup luas.

**Rating Minimum dan Maksimum**: 
  Rating minimum adalah 0, yang biasanya menandakan tidak adanya rating eksplisit dari pengguna. Rating maksimum adalah 10, yang merupakan skor tertinggi yang bisa diberikan pada sebuah buku.

**Distribusi Rating**:
  Kuartil ke-1 (25%), ke-2 (median/50%), dan ke-3 (75%) masing-masing bernilai 0, 0, dan 7, mengindikasikan bahwa sebagian besar data rating yang ada adalah 0. Hal ini berarti banyak interaksi pengguna tidak disertai dengan rating eksplisit.

**Kesimpulan**:
  Sebagian besar entri pada kolom `Book-Rating` bernilai 0, yang berarti banyak interaksi pengguna tidak disertai dengan rating eksplisit. Oleh karena itu, untuk membangun sistem rekomendasi berbasis rating, disarankan untuk memfilter hanya data dengan `Book-Rating` > 0.

![Deskripsi Gambar](https://drive.google.com/uc?export=view&id=1mTNyDkxqCWCY0dvbSUS05NOKCa-3Q_zq)

**Jumlah Data Rating**:
  Terdapat 1.149.780 data rating yang diberikan oleh pengguna.

**Jumlah Pengguna Unik**:
  Terdapat 105.283 pengguna unik yang memberikan rating.

**Jumlah Buku Unik yang Diberi Rating**:
  Terdapat 340.556 buku unik yang diberikan rating.

**Missing Values**:
  Tidak ada missing value pada dataset `ratings` karena semua kolom (`User-ID`, `ISBN`, dan `Book-Rating`) memiliki nilai yang lengkap (0 missing value).


![Deskripsi Gambar](https://drive.google.com/uc?export=view&id=1lvUiBhbjuwYdlREGIliirpOb4w9NzkYV)

**Jumlah Data Pengguna**:
  Dataset `users` berisi 278.858 entri pengguna.

**Jumlah Data yang Tidak Null**:
  - `User-ID`: 278.858 data tidak null.
  - `Location`: 278.858 data tidak null.
  - `Age`: 168.096 data tidak null, yang menunjukkan adanya nilai kosong pada kolom usia.

**Tipe Data**:
  - `User-ID`: integer (`int64`).
  - `Location`: string (`object`).
  - `Age`: angka desimal (`float64`).

![Deskripsi Gambar](https://drive.google.com/uc?export=view&id=1-fIKYX0EP7A_BN_m-037xygJ0T6-pzX0)

**Statistik Deskriptif Kolom 'Age':**
- Jumlah Data: 168.096
- Rata-rata Umur: 34.75 tahun
- Standar Deviasi: 14.43 tahun
- Umur Minimum: 0 tahun (kemungkinan data yang tidak valid)
- Kuartil:
- 25%: 24 tahun
- 50% (Median): 32 tahun
- 75%: 44 tahun
- Umur Maksimum: 244 tahun (kemungkinan data yang tidak valid)

**Jumlah Pengguna Unik:**
- 278.858 pengguna unik.

**Jumlah Lokasi Unik:**
- 57.339 lokasi unik.

---

## Data Preparation

### Handling Missing Value

Langkah pertama dalam tahapan pembersihan data adalah menangani missing value. Pada dataset yang digunakan, terdapat beberapa kolom yang memiliki missing values, terutama pada kolom `Book-Title`, `Book-Author`, `Year-Of-Publication`, `Publisher`, serta kolom gambar (`Image-URL-S`, `Image-URL-M`, `Image-URL-L`). Selain itu, kolom `Age` juga memiliki banyak missing value.

- **Solusi**: Untuk menjaga kualitas data, dilakukan pembuangan baris yang memiliki missing values menggunakan `dropna()`. Langkah ini memastikan bahwa hanya data yang lengkap dan valid yang digunakan dalam analisis selanjutnya.

### Handling Outliers

Dalam dataset ini, terdapat beberapa nilai yang dianggap sebagai outlier, seperti data usia pengguna (`Age`) yang lebih rendah dari 5 tahun atau lebih tinggi dari 99 tahun. Hal ini dapat menyebabkan distorsi dalam analisis dan model.

- **Solusi**: Usia yang tidak realistis difilter untuk memastikan hanya usia yang berada dalam rentang 5 hingga 99 tahun yang dipertimbangkan. Langkah ini penting untuk menjaga validitas data yang akan digunakan dalam pembuatan model.

### Handling Duplicates

Sebelum melanjutkan ke tahapan analisis lebih lanjut, perlu diperiksa apakah ada duplikat dalam dataset yang dapat mempengaruhi hasil analisis.

- **Solusi**: Dilakukan pengecekan dan penghapusan baris duplikat menggunakan metode `drop_duplicates()`. Ini memastikan bahwa tidak ada data yang terhitung lebih dari sekali, yang dapat memengaruhi hasil model rekomendasi.

### Standarisasi dan Normalisasi Data

Setelah menangani missing values dan outliers, penting untuk memastikan bahwa data berada dalam format yang standar dan siap digunakan dalam model.

- **Normalisasi Rating**: Rating buku yang diterima oleh pengguna memiliki rentang nilai yang berbeda-beda. Oleh karena itu, dilakukan normalisasi pada nilai rating dengan metode **Min-Max Normalization**, mengubah nilai rating ke dalam rentang 0–1 untuk memastikan bahwa model dapat beroperasi dengan lebih stabil.

### Content-Based Filtering Preparation

#### 1. **Menggabungkan Semua Data**
Proses penggabungan ketiga dataset utama (`Books.csv`, `Ratings.csv`, dan `Users.csv`) dilakukan berdasarkan kolom kunci seperti `ISBN` untuk buku dan `User-ID` untuk pengguna. Hal ini dilakukan agar informasi tentang pengguna, buku, dan rating terintegrasi dalam satu dataset yang utuh.

#### 2. **Persiapan Kolom Buku**
Kolom `ISBN`, `Book-Title`, dan `Book-Author` dikonversi menjadi list dan dibentuk menjadi dictionary. Struktur ini akan digunakan dalam pembuatan sistem rekomendasi berbasis konten, yang mengandalkan metadata buku seperti judul dan pengarang.

### Collaborative Filtering Preparation

#### 1. **Encode Label**
Untuk memungkinkan penggunaan data kategori dalam model TensorFlow, dilakukan encoding untuk kolom `User-ID` dan `ISBN` ke dalam format numerik. Hal ini penting karena model tidak dapat memproses data kategori langsung, dan encoding menjadi kunci untuk mengonversi data menjadi format yang dapat diproses oleh model berbasis embedding.

#### 2. **Pembuatan Variabel Input dan Target**
Setelah data berhasil diencoding, variabel input (`x`) yang berisi pasangan `(user, book)` dibuat, sementara target (`y`) berisi rating yang telah dinormalisasi. Format ini memungkinkan model untuk mempelajari hubungan antara pengguna dan buku.

#### 3. **Split Data**
Dataset dibagi menjadi dua bagian, 80% untuk training dan 20% untuk validasi. Pembagian ini dilakukan untuk memastikan model dapat diuji untuk mengukur generalisasi dan kemampuannya untuk bekerja pada data yang tidak terlihat sebelumnya.

### Ekstraksi Fitur TF-IDF
Ekstraksi fitur menggunakan **TF-IDF (Term Frequency-Inverse Document Frequency)** dilakukan untuk mengukur pentingnya kata-kata dalam deskripsi buku relatif terhadap kumpulan dokumen lainnya. Fitur ini penting untuk memahami kecocokan antara buku berdasarkan deskripsi teks, yang digunakan dalam rekomendasi berbasis konten.

### Balancing Data
Jika dataset yang digunakan menunjukkan adanya ketidakseimbangan antara rating atau kelas, dilakukan teknik balancing data untuk memastikan model dapat belajar dengan lebih adil tanpa bias terhadap kelas yang lebih banyak.

---

## Modeling

Tahapan ini menjelaskan proses pembuatan dan evaluasi model sistem rekomendasi yang digunakan dalam proyek. Dua pendekatan utama yang digunakan adalah **Content-Based Filtering** dan **Collaborative Filtering**. Selain penjelasan, hasil top-N rekomendasi juga dilampirkan sebagai bukti keberhasilan model dalam memberikan saran buku kepada pengguna.

## 1. Content-Based Filtering
- Menggunakan informasi dari fitur konten buku, seperti **nama author**.
- Sistem mencari kemiripan antar buku menggunakan teknik **TF-IDF** dan **Cosine Similarity** berdasarkan fitur `author`.
- Rekomendasi yang dihasilkan adalah buku yang memiliki **kemiripan tinggi** dengan buku yang pernah disukai pengguna.

### Cara Kerja
1. Membersihkan data `author` dengan menghapus karakter aneh.
2. Menggunakan `TfidfVectorizer` untuk membentuk matriks TF-IDF dari kolom `author`.
3. Menghitung **cosine similarity** antar vektor buku.
4. Mengambil top-N buku paling mirip dari similarity matrix.

### Kelebihan
- Tidak membutuhkan data dari pengguna lain.
- Cocok untuk kondisi dengan data pengguna terbatas (cold-start user).

### Kekurangan
- Cenderung memberikan rekomendasi yang mirip terus-menerus (kurang variasi).
- Tidak bisa merekomendasikan buku yang belum pernah disukai siapapun.

### 📊 Contoh Output Rekomendasi
Rekomendasi berdasarkan buku **Passenger to Frankfurt: An extravaganza,**:

![7](https://github.com/user-attachments/assets/f2ba62c0-d321-48dd-9874-61d0f23d17f2)

## 2. Collaborative Filtering

Collaborative Filtering adalah pendekatan sistem rekomendasi yang memanfaatkan **interaksi historis pengguna**, seperti **rating terhadap item (buku)**. Sistem ini mencoba mempelajari pola preferensi pengguna berdasarkan kesamaan perilaku dengan pengguna lain.

Dalam proyek ini, digunakan model **Neural Collaborative Filtering (NCF)** dengan arsitektur kustom bernama **RecommenderNet**, yang dibangun menggunakan TensorFlow dan Keras. Model ini melakukan pembelajaran representasi (embedding) untuk pengguna dan buku, kemudian memprediksi **seberapa besar kemungkinan seorang pengguna menyukai sebuah buku**.

### Cara Kerja Model
- Data interaksi pengguna dan buku disiapkan dalam bentuk pasangan `(user_id, isbn)` dengan label rating.
- Setiap user dan buku diubah menjadi vektor embedding berdimensi tetap.
- Vektor tersebut digabungkan dan diproses melalui lapisan dense untuk mempelajari hubungan non-linear antara user dan buku.
- Model dilatih dengan **loss function Binary Crossentropy** dan **metrik evaluasi RMSE**.
- Setelah pelatihan, sistem dapat memprediksi skor rating dari setiap kombinasi user-buku yang belum pernah dilihat sebelumnya.

### Kelebihan
- Mampu menangkap pola kompleks antar user dan item.
- Dapat memberikan rekomendasi yang lebih **personal** karena mempertimbangkan interaksi pengguna lainnya.
- Menghasilkan **top-N rekomendasi** berdasarkan prediksi rating tertinggi.

### Kekurangan
- Membutuhkan data rating yang cukup banyak, rentan terhadap **sparsity**.
- Tidak optimal untuk pengguna/item baru (**cold-start problem**).

### 🏆 Hasil Top-N Rekomendasi (Sample Output)

📌 **Rekomendasi untuk satu pengguna secara acak**:

![8](https://github.com/user-attachments/assets/3e3ed645-d3f3-4a2d-b205-0ca866f8486c)

Dengan pendekatan ini, sistem dapat memberikan saran buku secara cerdas dan dinamis berdasarkan preferensi pembaca yang serupa, bukan hanya berdasarkan isi buku.

---
## Evaluation

### Metrik Evaluasi yang Digunakan

Dalam proyek ini, digunakan dua pendekatan sistem rekomendasi, yaitu:

1. **Collaborative Filtering (menggunakan RecommenderNet berbasis deep learning)**
2. **Content-Based Filtering (menggunakan cosine similarity)**

Setiap pendekatan menggunakan metrik evaluasi yang berbeda, disesuaikan dengan karakteristik masalah dan data.

---

#### 1. Root Mean Squared Error (RMSE) – untuk Collaborative Filtering

- **Definisi:**  
  RMSE digunakan untuk mengukur sejauh mana prediksi model mendekati nilai rating aktual. Metrik ini sangat peka terhadap kesalahan prediksi yang besar karena menghitung akar dari rata-rata kuadrat selisih prediksi dan nilai sebenarnya.

- **Formula:**

![R](https://github.com/user-attachments/assets/47948cde-13f4-494e-94f7-b69843e3ec69)

- **Alasan Penggunaan:**  
  Cocok digunakan dalam masalah regresi seperti prediksi rating buku karena mempertimbangkan akurasi absolut dari prediksi secara keseluruhan.

---

#### 2. Precision@N – untuk Content-Based Filtering

- **Definisi:**  
  Precision@N mengukur berapa banyak dari N item yang direkomendasikan yang benar-benar relevan dengan minat pengguna.

- **Formula:**
  
![P](https://github.com/user-attachments/assets/6e31d9ca-4cca-470a-b817-35d6ebf45bbe)

- **Alasan Penggunaan:**  
  Sesuai untuk mengevaluasi skenario top-N recommendation seperti pada pendekatan berbasis konten, terutama untuk mengecek relevansi rekomendasi yang diberikan.

---

### Hasil Evaluasi

#### Collaborative Filtering (RecommenderNet)

- **Visualisasi RMSE:**

  ![RMSE Graph](https://drive.google.com/uc?export=view&id=183pG59ea2YWJiEElaH8rwI6X4E7IY2kp)

- **Train RMSE akhir:** ~0.55  
- **Validation RMSE akhir:** ~0.49  

Model menunjukkan kecenderungan konvergen selama proses pelatihan hingga epoch ke-100. Namun terdapat sedikit perbedaan antara error pelatihan dan validasi yang mengindikasikan potensi overfitting.

---

#### Content-Based Filtering

- **Precision@10 (simulasi):** ~0.60  
  Berdasarkan uji coba manual terhadap 10 rekomendasi teratas, sekitar 6 buku sesuai dengan minat pengguna, menghasilkan precision sebesar 60%.

---

### Interpretasi dan Hubungan dengan Business Understanding

- **Problem Statement:**  
  Sistem ini dirancang untuk membantu pengguna menemukan buku yang relevan dari jumlah koleksi yang sangat besar.

- **Apakah tujuan proyek tercapai?**  
  ✅ Ya  
  - *Collaborative Filtering* mampu memprediksi rating pengguna dengan RMSE < 0.6  
  - *Content-Based Filtering* memberikan rekomendasi yang cukup relevan dengan precision@10 sebesar 60%

- **Dampak terhadap bisnis dan pengguna:**  
  Sistem rekomendasi yang baik akan meningkatkan **kepuasan pengguna**, mempercepat proses pencarian buku, serta mendorong **interaksi lebih lanjut**, yang semuanya berdampak positif terhadap **retensi dan konversi pengguna** dalam konteks bisnis.

---

### Saran Perbaikan Model

#### Collaborative Filtering:
- Tambahkan teknik regularisasi seperti **dropout** atau **L2 regularization**
- Gunakan **early stopping** untuk mencegah overfitting
- Eksperimen dengan **arsitektur model** yang lebih kompleks atau sebaliknya lebih sederhana

#### Content-Based Filtering:
- Gunakan **fitur tambahan** seperti genre, publisher, tahun terbit, deskripsi buku
- Gabungkan dengan **Collaborative Filtering** menjadi model hybrid untuk hasil yang lebih baik
