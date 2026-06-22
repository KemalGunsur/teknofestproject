import cv2
import math
import json
import os
import torch
import numpy as np # Sahte test karesi için eklendi
from datetime import datetime
from ultralytics import YOLO
from cvzone.FaceMeshModule import FaceMeshDetector 

class SurucuAnalizMotoru:
    def __init__(self, log_dosyasi="ihlal_loglari.json", cooldown=3.0, arac_id="KriptoTrafik_X7T_01"):
        print("Kripto-Trafik: Sürücü Analiz Modülü Yükleniyor...")
        
        # İlk olarak modeli hafızaya alıyoruz
        self.model = YOLO('yolov8n-pose.pt')
        self.detector = FaceMeshDetector(maxFaces=1)
        
        # --- AKILLI DONANIM VE MİMARİ DOĞRULAMA (WARMUP) ---
        if torch.cuda.is_available():
            try:
                # Modeli GPU'ya taşımayı dene
                self.model.to('cuda')
                
                # Sürücü hatasını tetiklemek için sahte bir test karesi oluşturup çalıştırıyoruz
                sahte_kare = np.zeros((100, 100, 3), dtype=np.uint8)
                self.model(sahte_kare, verbose=False)
                
                # Eğer yukarıdaki satır çökmediyse GPU mimarin uyumludur
                self.cihaz = 'cuda'
                print(f"✅ GPU Algılandı ve Sürücü Uyumlu: {torch.cuda.get_device_name(0)}")
            except Exception as e:
                # GPU var ama sm_120 mimari hatası veya başka bir CUDA hatası alındıysa buraya düşer
                self.cihaz = 'cpu'
                self.model.to('cpu')
                print("⚠️ GPU mevcut fakat kütüphane mimari uyumsuzluğu nedeniyle otomatik olarak güvenli CPU moduna geçildi!")
        else:
            self.cihaz = 'cpu'
            print("⚠️ GPU Algılanamadı, CPU modunda devam ediliyor.")
        
        # 5G Loglama Ayarları
        self.LOG_DOSYASI = log_dosyasi
        self.COOLDOWN_SANIYE = cooldown
        self.ARAC_ID = arac_id
        self.son_log_zamanlari = {}
        
        # --- ZAMANA BAĞLI FİLTRELEME (DEBOUNCING) AYARLARI ---
        self.IHLAL_ESIKLERI = {
            "TEHLIKE: ESNEME (YORGUNLUK)!": 1.0,
            "TEHLIKE: UYUKLAMA!": 1.0,
            "TEHLIKE: ARKAYA BAKIYOR!": 0.5,
            "TEHLIKE: ETRAFA BAKINMA!": 2.0,
            "TEHLIKE: TELEFONLA KONUSMA": 1.5,
            "DIKKAT: SIGARA / SU ICME": 1.5
        }
        
        self.mevcut_ihlal = None
        self.ihlal_baslangic_zamani = None
        print(f"Sistem Hazır! Aktif Çalışma Birimi: [{self.cihaz.upper()}]")

    @staticmethod
    def mesafe_hesapla(nokta1, nokta2):
        return math.sqrt((nokta2[0] - nokta1[0])**2 + (nokta2[1] - nokta1[1])**2)

    @staticmethod
    def yatay_mesafe(nokta1, nokta2):
        return abs(nokta1[0] - nokta2[0])

    def ihlal_kaydet(self, eylem_adi):
        simdi = datetime.now()
        if eylem_adi in self.son_log_zamanlari:
            gecen_sure = (simdi - self.son_log_zamanlari[eylem_adi]).total_seconds()
            if gecen_sure < self.COOLDOWN_SANIYE: return 

        yeni_log = {
            "arac_id": self.ARAC_ID,
            "tarih_saat": simdi.strftime("%Y-%m-%d %H:%M:%S"),
            "ihlal_turu": eylem_adi,
            "durum": "ONAYLANDI"
        }

        loglar = []
        if os.path.exists(self.LOG_DOSYASI):
            with open(self.LOG_DOSYASI, "r", encoding="utf-8") as dosya:
                try: loglar = json.load(dosya)
                except json.JSONDecodeError: loglar = [] 
                    
        loglar.append(yeni_log)
        with open(self.LOG_DOSYASI, "w", encoding="utf-8") as dosya:
            json.dump(loglar, dosya, ensure_ascii=False, indent=4)
            
        self.son_log_zamanlari[eylem_adi] = simdi
        print(f"📡 [5G MERKEZE İLETİLDİ] -> {eylem_adi}")

    def kare_isle(self, frame):
        yolo_results = self.model(frame, stream=True, verbose=False)
        frame, faces = self.detector.findFaceMesh(frame, draw=False)

        annotated_frame = frame.copy()
        aktif_eylem = "NORMAL SEYIR"
        uyari_rengi = (0, 255, 0) 

        esniyor_mu = False
        if faces: 
            face = faces[0] 
            ust_dudak, alt_dudak = face[13], face[14]
            alin, cene = face[10], face[152]
            dudak_acikligi = self.mesafe_hesapla(ust_dudak, alt_dudak)
            yuz_uzunlugu = self.mesafe_hesapla(alin, cene)

            if yuz_uzunlugu > 0:
                if (dudak_acikligi / yuz_uzunlugu) > 0.12:
                    esniyor_mu = True
                    aktif_eylem = "TEHLIKE: ESNEME (YORGUNLUK)!"
                    uyari_rengi = (0, 0, 255)

        for r in yolo_results:
            annotated_frame = r.plot(img=annotated_frame)
            if r.keypoints is not None and len(r.keypoints.data) > 0:
                keypoints = r.keypoints.data[0].numpy() 
                if len(keypoints) >= 11:
                    burun = keypoints[0]
                    sol_goz, sag_goz = keypoints[1], keypoints[2]
                    sol_kulak, sag_kulak = keypoints[3], keypoints[4]
                    sol_omuz, sag_omuz = keypoints[5], keypoints[6]
                    sol_bilek, sag_bilek = keypoints[9], keypoints[10]

                    omuz_genisligi = 200 
                    if sol_omuz[2] > 0.5 and sag_omuz[2] > 0.5:
                        omuz_genisligi = self.mesafe_hesapla((sol_omuz[0], sol_omuz[1]), (sag_omuz[0], sag_omuz[1]))

                    if not esniyor_mu:
                        if sol_omuz[2] > 0.5 and sag_omuz[2] > 0.5:
                            if sol_goz[2] < 0.5 and sag_goz[2] < 0.5 and burun[2] < 0.5:
                                aktif_eylem = "TEHLIKE: ARKAYA BAKIYOR!"
                                uyari_rengi = (0, 0, 255)

                        if aktif_eylem == "NORMAL SEYIR":
                            if burun[2] > 0.5 and sol_omuz[2] > 0.5 and sag_omuz[2] > 0.5:
                                omuz_y_ortalama = (sol_omuz[1] + sag_omuz[1]) / 2
                                if (omuz_y_ortalama - burun[1]) < (omuz_genisligi * 0.20):
                                    aktif_eylem = "TEHLIKE: UYUKLAMA!"
                                    uyari_rengi = (0, 0, 255) 

                            if aktif_eylem == "NORMAL SEYIR" and burun[2] > 0.5:
                                if sol_goz[2] < 0.4 or sag_goz[2] < 0.4:
                                    aktif_eylem = "TEHLIKE: ETRAFA BAKINMA!"
                                    uyari_rengi = (0, 165, 255)
                                elif sol_goz[2] > 0.5 and sag_goz[2] > 0.5:
                                    oran = max(self.yatay_mesafe(burun, sol_goz), self.yatay_mesafe(burun, sag_goz)) / (min(self.yatay_mesafe(burun, sol_goz), self.yatay_mesafe(burun, sag_goz)) + 0.1)
                                    if oran > 3.0:
                                        aktif_eylem = "TEHLIKE: ETRAFA BAKINMA!"
                                        uyari_rengi = (0, 165, 255)

                            if aktif_eylem == "NORMAL SEYIR":
                                esik = omuz_genisligi * 0.65 if omuz_genisligi != 200 else 130
                                sol_el_h = sol_bilek[1] < sol_omuz[1] if sol_omuz[2] > 0.5 else True
                                sag_el_h = sag_bilek[1] < sag_omuz[1] if sag_omuz[2] > 0.5 else True
                                if (sol_kulak[2]>0.5 and sol_bilek[2]>0.5 and sol_el_h and self.mesafe_hesapla(sol_kulak[:2], sol_bilek[:2]) < esik) or \
                                   (sag_kulak[2]>0.5 and sag_bilek[2]>0.5 and sag_el_h and self.mesafe_hesapla(sag_kulak[:2], sag_bilek[:2]) < esik):
                                    aktif_eylem = "TEHLIKE: TELEFONLA KONUSMA"
                                    uyari_rengi = (0, 0, 255)

                            if aktif_eylem == "NORMAL SEYIR":
                                if (sol_bilek[2]>0.5 and sol_omuz[2]>0.5 and sol_bilek[1]<sol_omuz[1]) or \
                                   (sag_bilek[2]>0.5 and sag_omuz[2]>0.5 and sag_bilek[1]<sag_omuz[1]):
                                    aktif_eylem = "DIKKAT: SIGARA / SU ICME"
                                    uyari_rengi = (128, 0, 128)

        if aktif_eylem != "NORMAL SEYIR":
            if aktif_eylem == self.mevcut_ihlal:
                if (datetime.now() - self.ihlal_baslangic_zamani).total_seconds() >= self.IHLAL_ESIKLERI.get(aktif_eylem, 1.0):
                    self.ihlal_kaydet(aktif_eylem)
            else:
                self.mevcut_ihlal, self.ihlal_baslangic_zamani = aktif_eylem, datetime.now()
        else:
            self.mevcut_ihlal, self.ihlal_baslangic_zamani = None, None

        cv2.putText(annotated_frame, aktif_eylem, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, uyari_rengi, 3)
        return annotated_frame, aktif_eylem

if __name__ == "__main__":
    motor = SurucuAnalizMotoru()
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        islenmis_kare, durum = motor.kare_isle(frame)
        cv2.imshow("Kripto-Trafik: Final Hata Korumali Motor", islenmis_kare)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cap.release()
    cv2.destroyAllWindows()