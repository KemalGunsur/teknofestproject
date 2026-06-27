"""
Modul  : surucu_analiz.py
Kaynak : KemalGunsur / teknofestproject (surucu_test1.py icindeki SurucuAnalizMotoru sinifi)
Degisiklikler:
  - Canli kamera (cv2.VideoCapture(0)) destegi kaldirildi; video dosyasi modu eklendi.
  - ihlal_kaydet() JSON log yazimi kaldirildi; sonuclar dogrudan tespitler listesine
    aktarilacak sekilde kare_isle() donusu degistirildi:
      Eski: (annotated_frame, eylem_str)
      Yeni: (eylem_str | None, conf_score)
  - cv2.imshow / waitKey cagrilari kaldirildi (headless Docker ortami icin).
  - Tum ic Turkce buyuk harfli mesajlar aynen korundu;
    src/predict.py icindeki _eylemi_donustur() bunlari yarişma etiketine ceviriyor.
  - GPU/CPU otomatik secim mekanizmasi korundu.
"""

import math
import numpy as np
import torch

from ultralytics import YOLO
from cvzone.FaceMeshModule import FaceMeshDetector


class SurucuAnalizMotoru:
    """
    Pose + FaceMesh tabanli surucu davranis analiz motoru.
    Headless (ekransiz) calisacak sekilde duzenlenmistir.
    """

    def __init__(self, weights_path: str):
        print("[SurucuAnaliz] Model yukleniyor...")

        self.model    = YOLO(weights_path)
        self.detector = FaceMeshDetector(maxFaces=1)

        # GPU / CPU secimi
        if torch.cuda.is_available():
            try:
                self.model.to("cuda")
                dummy = np.zeros((100, 100, 3), dtype=np.uint8)
                self.model(dummy, verbose=False)
                self.cihaz = "cuda"
                print(f"[SurucuAnaliz] GPU aktif: {torch.cuda.get_device_name(0)}")
            except Exception:
                self.model.to("cpu")
                self.cihaz = "cpu"
                print("[SurucuAnaliz] GPU mimarisi uyumsuz, CPU moduna gecildi.")
        else:
            self.cihaz = "cpu"
            print("[SurucuAnaliz] GPU bulunamadi, CPU modu.")

        # Zamanlama / debounce
        self.mevcut_ihlal        = None
        self.ihlal_baslangic_san = None   # baslangic zamani saniye cinsinden (video fps bazli)
        self.son_basarili_zaman  = None   

        self.IHLAL_ESIKLERI = {
            "esneme"          : 1.0,
            "uyuklama"        : 1.0,
            "arkaya_bakma"    : 0.5,
            "etrafa_bakinma"  : 2.0,
            "telefonla_konusma": 1.5,
            "sigara_su_icme"  : 1.5,
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _mesafe(p1, p2):
        return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

    @staticmethod
    def _yatay_mesafe(p1, p2):
        return abs(p1[0] - p2[0])

    # ------------------------------------------------------------------
    def kareyi_isle(self, frame, zaman_saniye: float):
        """
        Tek bir kareyi analiz eder.

        Parametreler
        ------------
        frame         : BGR numpy dizisi
        zaman_saniye  : Videonun su anki zaman noktasi (kare_no / fps)

        Donus
        -----
        (eylem_str | None, conf_score: float)
          eylem_str  -> Tespit edilen eylemin ic adi (asagida tanimli).
                        Hicbir ihlal yoksa None.
          conf_score -> 0.0 – 1.0 arasi guven skoru (pose keypoint ozeti).
        """
        yolo_results = self.model(frame, stream=True, verbose=False)
        frame_copy, faces = self.detector.findFaceMesh(frame, draw=False)

        aktif_eylem   = None
        conf_score    = 0.0
        esniyor_mu    = False

        # ---------- YUZ ANALIZI ----------
        if faces:
            face            = faces[0]
            ust_dudak       = face[13]
            alt_dudak       = face[14]
            alin            = face[10]
            cene            = face[152]
            dudak_acikligi  = self._mesafe(ust_dudak, alt_dudak)
            yuz_uzunlugu    = self._mesafe(alin, cene)

            if yuz_uzunlugu > 0 and (dudak_acikligi / yuz_uzunlugu) > 0.12:
                esniyor_mu  = True
                aktif_eylem = "esneme"
                conf_score  = min(1.0, (dudak_acikligi / yuz_uzunlugu) / 0.20)

        # ---------- POSE ANALIZI ----------
        for r in yolo_results:
            if r.keypoints is None or len(r.keypoints.data) == 0:
                continue

            kp = r.keypoints.data[0].numpy()
            if len(kp) < 11:
                continue

            burun       = kp[0]
            sol_goz     = kp[1];  sag_goz  = kp[2]
            sol_kulak   = kp[3];  sag_kulak = kp[4]
            sol_omuz    = kp[5];  sag_omuz  = kp[6]
            sol_bilek   = kp[9];  sag_bilek = kp[10]

            omuz_genisligi = 200
            if sol_omuz[2] > 0.5 and sag_omuz[2] > 0.5:
                omuz_genisligi = self._mesafe(sol_omuz[:2], sag_omuz[:2])

            # Arkaya bakma
            if not esniyor_mu and aktif_eylem is None:
                if sol_omuz[2] > 0.5 and sag_omuz[2] > 0.5:
                    if sol_goz[2] < 0.5 and sag_goz[2] < 0.5 and burun[2] < 0.5:
                        aktif_eylem = "arkaya_bakma"
                        conf_score  = 0.80

            # Uyuklama
            if aktif_eylem is None:
                if burun[2] > 0.5 and sol_omuz[2] > 0.5 and sag_omuz[2] > 0.5:
                    omuz_y_ort = (sol_omuz[1] + sag_omuz[1]) / 2
                    if (omuz_y_ort - burun[1]) < (omuz_genisligi * 0.20):
                        aktif_eylem = "uyuklama"
                        conf_score  = 0.82

            # Etrafa bakinma
            if aktif_eylem is None and burun[2] > 0.5:
                if sol_goz[2] < 0.4 or sag_goz[2] < 0.4:
                    aktif_eylem = "etrafa_bakinma"
                    conf_score  = 0.70
                elif sol_goz[2] > 0.5 and sag_goz[2] > 0.5:
                    d1 = self._yatay_mesafe(burun[:2], sol_goz[:2])
                    d2 = self._yatay_mesafe(burun[:2], sag_goz[:2])
                    oran = max(d1, d2) / (min(d1, d2) + 0.1)
                    if oran > 3.0:
                        aktif_eylem = "etrafa_bakinma"
                        conf_score  = min(1.0, oran / 4.5)

            # Telefonla konusma
            if aktif_eylem is None:
                esik     = omuz_genisligi * 0.65 if omuz_genisligi != 200 else 130
                sol_el_h = sol_bilek[1] < sol_omuz[1] if sol_omuz[2] > 0.5 else True
                sag_el_h = sag_bilek[1] < sag_omuz[1] if sag_omuz[2] > 0.5 else True

                sol_hit = (sol_kulak[2] > 0.5 and sol_bilek[2] > 0.5 and sol_el_h and
                           self._mesafe(sol_kulak[:2], sol_bilek[:2]) < esik)
                sag_hit = (sag_kulak[2] > 0.5 and sag_bilek[2] > 0.5 and sag_el_h and
                           self._mesafe(sag_kulak[:2], sag_bilek[:2]) < esik)

                if sol_hit or sag_hit:
                    aktif_eylem = "telefonla_konusma"
                    conf_score  = 0.88

            # Sigara / su icme (el omuz ustunde)
            if aktif_eylem is None:
                sol_yukari = sol_bilek[2] > 0.5 and sol_omuz[2] > 0.5 and sol_bilek[1] < sol_omuz[1]
                sag_yukari = sag_bilek[2] > 0.5 and sag_omuz[2] > 0.5 and sag_bilek[1] < sag_omuz[1]
                if sol_yukari or sag_yukari:
                    aktif_eylem = "sigara_su_icme"
                    conf_score  = 0.72

       # ---------- DEBOUNCE (Kayıp Kare Toleranslı Güncelleme) ----------
        TOLERANS_SANIYE = 0.5  

        if aktif_eylem is None:
            if self.mevcut_ihlal is not None and self.son_basarili_zaman is not None:
                if (zaman_saniye - self.son_basarili_zaman) < TOLERANS_SANIYE:
                    aktif_eylem = self.mevcut_ihlal
                else:
                    self.mevcut_ihlal        = None
                    self.ihlal_baslangic_san = None
                    return None, 0.0
            else:
                self.mevcut_ihlal        = None
                self.ihlal_baslangic_san = None
                return None, 0.0

        if aktif_eylem == self.mevcut_ihlal:
            self.son_basarili_zaman = zaman_saniye
            gecen = zaman_saniye - (self.ihlal_baslangic_san or zaman_saniye)
            esik  = self.IHLAL_ESIKLERI.get(aktif_eylem, 1.0)
            if gecen >= esik:
                return aktif_eylem, conf_score
            return None, 0.0
        else:
            self.mevcut_ihlal        = aktif_eylem
            self.ihlal_baslangic_san = zaman_saniye
            self.son_basarili_zaman  = zaman_saniye
            return None, 0.0
