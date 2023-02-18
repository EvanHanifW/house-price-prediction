# Laporan Proyek Machine Learning - Evan Hanif Widiatama

## Domain Proyek

Rumah sebagai tempat tinggal adalah salah satu kebutuhan pokok manusia selain pakaian dan makanan. Setiap manusia membutuhkan rumah untuk tempat berlindung dan sebagai tempat berkumpul dan berlangsungnya aktivitas keluarga, sekaligus sebagai sarana investasi. Menentukan harga jual rumah bisa menjadi tugas yang sulit dan memerlukan beberapa pertimbangan. Beberapa fitur seperti luas rumah, luas tanah, kondisi rumah, letak rumah, dll dapat membuat bingung bagi pemilik rumah yang masih awam dan bingung untuk menjual rumahnya di harga tertentu.

Berlandaskan dari masalah ini, kita dapat menggatasinya dengan menggunakan algoritma machine learning. Supervised learning solusinya, kita dapat menggunakan dataset rumah berdasarkan fitur-fiturnya yang telah dilabeli dengan harga yang pantas untuk membuat model yang sesuai. Dengan ini, diharapkan masalah tentang kesulitan menentukan harga rumah dapat diatasi dengan model machine learning kita.

## Business Understanding

Sebagai tujuan awal, proyek ini dibangun untuk:
- Masyarakat umum yang sedang kebingungan untuk menentukan harga jual rumahnya
- Perusahaan jual-beli properti

### Problem Statements

1. Bagaimana cara memproses data agar dapat diterima oleh model?
2. Model apa yang paling baik untuk digunakan?

### Goals

1. Melakukan Data Preprocessing dengan tepat
2. Menggunakan MAE untuk mengetahui model yang paling baik.

### Solution statements

1. Melakukan Handling Outliers dengan IQR
2. Melakukan One Hot Encoding untuk categorical feature
3. Membuang fitur dengan korelasi kecil (|Correlation| <= 0.15)
4. Menggunakan Model Algoritma: KNN, AdaBoost, RandomForest, SVR

## Data Understanding
Dataset yang digunakan adalah dataset House Price Prediction yang didapatkan dari [Kaggle](https://www.kaggle.com/datasets/shree1992/housedata). Dataset ini memiliki 4600 sampel dengan 18 Fitur.

### Variabel-variabel pada dataset sebagai berikut:
- Price: Merepresentasikan harga dalam USD dan merupakan fitur target
- Bedroom: Merepresentasikan jumlah kamar tidur
- Bathroom: Merepresentasikan jumlah kamar mandi
- sqft_living: Merepresentasikan ukuran luas rumah
- sqft_lot: Merepresentasikan ukuran luas tanah
- waterfront: Merepresentasikan variabel dummmy apakah rumah berada dekat dengan akses perairan
- view: Merepresentasikan variabel dummy tentang view di sekitar rumah
- condition: Merepresentasikan kondisi rumah
- sqft_above: Merepresentasikan luas diatas tanah
- sqft_basement: Merepresentasikan luas basement
- yr_built: Merepresentasikan tahun kapan rumah dibuat
- yr_renovated: Merepresentasikan tahun kapan rumah terakhir direnovasi
- street: Merepresntasikan jalan letak rumah
- city: Merepresentasikan kota letak rumah

### EDA - Handling Outliers:


## Data Preparation
Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan proses data preparation yang dilakukan
- Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut.

## Modeling
Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan. Anda perlu menjelaskan tahapan dan parameter yang digunakan pada proses pemodelan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan kelebihan dan kekurangan dari setiap algoritma yang digunakan.
- Jika menggunakan satu algoritma pada solution statement, lakukan proses improvement terhadap model dengan hyperparameter tuning. **Jelaskan proses improvement yang dilakukan**.
- Jika menggunakan dua atau lebih algoritma pada solution statement, maka pilih model terbaik sebagai solusi. **Jelaskan mengapa memilih model tersebut sebagai model terbaik**.

## Evaluation
Pada bagian ini anda perlu menyebutkan metrik evaluasi yang digunakan. Lalu anda perlu menjelaskan hasil proyek berdasarkan metrik evaluasi yang digunakan.

Sebagai contoh, Anda memiih kasus klasifikasi dan menggunakan metrik **akurasi, precision, recall, dan F1 score**. Jelaskan mengenai beberapa hal berikut:
- Penjelasan mengenai metrik yang digunakan
- Menjelaskan hasil proyek berdasarkan metrik evaluasi

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.
