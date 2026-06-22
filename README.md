# Sürücü Analiz Sistemi

Bu proje, araç içi kamera görüntüsünden sürücü davranışlarını analiz eden bir Python uygulamasıdır. Sistem; esneme, uyuklama, arkaya bakma, etrafa bakınma, telefonla konuşma ve sigara/su içme gibi durumları tespit etmeye çalışır. Tespit edilen ihlaller `ihlal_loglari.json` dosyasına kaydedilir.

## Projede Bulunan Dosyalar

- `surucu_test1.py`: Ana uygulama ve analiz motoru.
- `requirements.txt`: Gerekli Python paketleri.
- `yolov8n-pose.pt`: Pose tespiti için kullanılan YOLO model dosyası.
- `face_landmarker.task`: Yüz noktalarını okumak için kullanılan model dosyası.
- `ihlal_loglari.json`: Tespit edilen ihlallerin yazıldığı günlük dosyası.

## Gereksinimler

Bu proje Python ortamında çalışır. Genel olarak aşağıdaki bileşenler gerekir:

- Python 3.10 veya daha yeni bir sürüm
- Bir kamera veya webcam
- Model dosyaları: `yolov8n-pose.pt` ve `face_landmarker.task`

Paketler `requirements.txt` üzerinden kurulabilir. Proje, diğerlerinin yanında şu kütüphaneleri kullanır:

- `opencv-python`
- `numpy`
- `torch`
- `ultralytics`
- `cvzone`
- `mediapipe`

## Kurulum

1. Proje klasörünü bilgisayarınıza kopyalayın.
2. Bir sanal ortam oluşturun:

```powershell
python -m venv .venv
```

3. Sanal ortamı etkinleştirin:

```powershell
.\.venv\Scripts\Activate.ps1
```

4. Bağımlılıkları yükleyin:

```powershell
pip install -r requirements.txt
```

## Çalıştırma

Uygulamayı başlatmak için proje klasöründe şu komutu çalıştırın:

```powershell
python surucu_test1.py
```

Uygulama açıldığında varsayılan olarak bilgisayar kamerasını kullanır. Pencere açıkken `q` tuşuna basarak programı kapatabilirsiniz.

## Nasıl Çalışır?

Program, kamera görüntüsünü kare kare işler ve iki farklı analiz yapar:

- Yüz bölgesinden ağız ve göz konumlarını kontrol eder.
- `yolov8n-pose.pt` modeli ile vücut/kol/omuz noktalarını değerlendirir.

Bu kontroller sonucunda bir ihlal tespit edilirse ekranda uyarı gösterilir ve olay `ihlal_loglari.json` dosyasına yazılır. Aynı ihlalin çok sık kaydedilmemesi için bekleme süresi (cooldown) uygulanır.

## GPU / CPU Davranışı

Kod, uygun bir NVIDIA CUDA ortamı varsa modeli GPU üzerinde çalıştırmayı dener. GPU uyumsuzluğu veya CUDA hatası oluşursa otomatik olarak CPU moduna düşer. Bu nedenle aynı proje farklı bilgisayarlarda ek ayar yapmadan çalışabilir.(rtx 5070 ekran kartıma bulunan cuda mimarisi kullanılan pytorch kutuphanesinin güncel olmaması nedeniyle olmadı kendi bilgisayarımda cpu üzerinden kullanım ve test etmeye devam ettim.)

## Log Dosyası

İhlal kayıtları `ihlal_loglari.json` içinde JSON liste yapısında tutulur. Dosya yoksa program ilk kayıt sırasında otomatik oluşturur.

## Sık Karşılaşılan Sorunlar

- Kamera açılmıyorsa başka bir uygulama kamerayı kullanıyor olabilir.
- Model dosyaları eksikse uygulama başlarken hata alabilirsiniz. `yolov8n-pose.pt` ve `face_landmarker.task` dosyalarının proje kökünde olduğundan emin olun.
- `pip install -r requirements.txt` sırasında hata alırsanız önce sanal ortamın aktif olduğundan emin olun.
- GPU ile ilgili hata alırsanız bu normal olabilir; uygulama CPU moduna geçmeyi dener.

## Notlar

Bu proje bir yardım sistemi olarak tasarlanmıştır. Sonuçlar ortam ışığı, kamera kalitesi ve oturma pozisyonuna göre değişebilir. Kritik kullanım senaryolarında çıktılar mutlaka insan kontrolü ile doğrulanmalıdır.