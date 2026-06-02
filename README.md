Final Project: Deteksi Kendaraan dan Estimasi Kecepatan Jalan Tol(ETLE)

Proyek *Computer Vision* ini bertujuan untuk mendeteksi, melacak (*tracking*), dan mengestimasi kecepatan kendaraan secara otomatis dari rekaman video cctv jalan tol. Proyek ini dibangun menggunakan **YOLOv8** dengan model training custom menggunakan frame-frame yang telah di annotate menggunakan roboflow untuk deteksi objek dan algoritma **ByteTrack** untuk pelacakan kendaraan lintas *frame*, mensimulasikan sistem tilang elektronik (ETLE).

Fitur Utama
* **Real-Time Tracking:** Menggunakan YOLOv8 dan ByteTrack untuk mempertahankan ID unik pada setiap kendaraan yang melintas.
* **Speed Estimation:** Menghitung kecepatan kendaraan berdasarkan waktu tempuh (Δ Frame) antara dua garis virtual (Garis Atas dan Garis Bawah) dengan jarak dunia nyata yang telah dikalibrasi (18 meter).
* **Speed Violation Alert:** Indikator visual otomatis (mengubah warna bounding box menjadi **merah**) untuk kendaraan yang melaju di atas batas kecepatan (> 60 km/h).
* **Live HUD Statistics:** Menampilkan total ID unik yang terdeteksi, jumlah yang berhasil dihitung kecepatannya, dan yang belum/gagal dihitung langsung pada layar video.
* **Automated Reporting:** Mencetak ringkasan akurasi dan *success rate* (tingkat keberhasilan perhitungan dari id-id kendaraam yang terdeteksi) di terminal saat program selesai dijalankan.

Struktur Repository (GitHub)
* `ETLEmodern.py` dan `Source Code Final Project Deteksi Kendaraan di Jalan Tol Kelompok 5.py`: Skrip utama sistem estimasi kecepatan menggunakan YOLOv8 yang sudah di train secara custom.
* `ETLEtradisional.py`: Skrip pendekatan tradisional/awal untuk perbandingan (menggunakan metode tracking klasik / non deeplearning).
* `Training Model YOLOV8.ipynb` : *Jupyter Notebook* yang berisi *source code* proses *training* model kustom YOLOv8.
* `best.pt` : *Weights* model YOLOv8 kustom hasil *training*.
* `Background tetap Objek Gerak.mp4/.mkv` : Video sampel pengujian initial (tidak dipakai)
* `Moving Background And Foreground.mp4` : Video sampel pengujian initial (tidak dipakai)
* `Moving Foreground Morning.mp4` : Video sampel pengujian kondisi awal (video pendek untuk mengetes model secara singkat)

Aset Tambahan (Google Drive)
Beberapa *file* berukuran besar dan dokumen laporan tidak diunggah ke GitHub, melainkan disimpan di Google Drive. 
**https://drive.google.com/drive/folders/1REVXoFJjvHfZUHSgRKPwERgJBBNfJML9?usp=sharing**

Isi Google Drive meliputi:
1. `based_video.mp4` (455.1 MB) - Video rekaman utama yang dipakai berdurasi sekitar 12 menit yang digunakan untuk pengujian algoritma di `ETLEmodern.py` dan atau `Source Code Final Project Deteksi Kendaraan di Jalan Tol Kelompok 5.py`.
2. `Laporan_CV.pdf` & `CVL_Final Project.pdf` - Laporan PPT dan juga Dokumen mengenai project ini
3. `Video Presentasi Final Project Deteksi Kecepatan.mp4` (125.3 MB) - Video pemaparan hasil akhir proyek.
4. Salinan *source code* dan *weights* (`best.pt`) untuk *backup*.
