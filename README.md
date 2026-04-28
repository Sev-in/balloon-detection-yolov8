# 🎈 Balloon Detection with YOLOv8

Bu projede YOLOv8 kullanılarak balon tespiti yapan bir model geliştirilmiştir.

---

## 📂 Veri Seti Süreci

1. Roboflow üzerinden sadece **parti balonları içeren 7 farklı veri seti** buldum.  
   Toplamda **7229 adet görüntü** içeriyor.

2. Tüm veri setlerinde farklı etiketler olduğu için:
   - Tüm label dosyalarını düzenledim
   - Tek sınıf olacak şekilde **tüm etiketleri 0 (balloon)** yaptım

3. Sonrasında:
   - Tüm veri setlerini birleştirdim
   - Duplicate (tekrar eden) görüntüleri kaldırdım
   - Tek bir veri seti haline getirdim
   - `dataset_merged.yaml` dosyasını oluşturdum

---

## 🧠 Model Eğitimi

YOLOv8n modeli ile eğitim gerçekleştirdim.

```python
model.train(
    data=str(DATASET_YAML),
    epochs=50,
    imgsz=640,
    batch=8,
    device=0,
    workers=2,
    project=str(RUNS_DIR),
    name="balloon_yolov8n"
)
```
---
📊 Eğitim Sonuçları
```python
mAP50: 0.967
mAP50-95: 0.798
```
---
🧪 Test Sonuçları
```python
Precision: 0.92
Recall: 0.907
mAP50: 0.926
mAP50-95: 0.76
```
---
⚡ Performans

Ortalama inference süresi: ~5–25 ms
Gerçek zamanlı çalışmaya uygundur
---
🚀 ONNX Dönüşümü

Model ONNX formatına çevrilmiştir:
```python
yolo export model=best.pt format=onnx
```
---
🎯 Sonuç

Model, balon tespiti için yüksek doğruluk elde etmiştir ve:

Gerçek zamanlı uygulamalarda kullanılabilir
Tek sınıflı detection senaryoları için uygundur
---
📌 Not

Model yalnızca balon tespiti için eğitilmiştir.
Farklı nesneler için yeniden eğitim gereklidir.
---
👨‍💻 Geliştirici

Sev-in
