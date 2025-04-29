# CNN
# Giriş
Bu projede derin öğrenme tabanlı evrişimli sinir ağları (CNN) yaklaşımları kullanılarak el yazısı rakam sınıflandırma ve görüntü özellik çıkarımı üzerine çalışmalar yapılmıştır. Amaç, farklı CNN mimarilerinin performanslarını karşılaştırmak ve CNN ile klasik makine öğrenmesi modellerini (SVM) birleştiren hibrit bir yapının etkinliğini değerlendirmektir.

# Yöntem
## Veri Setleri ve Ön İşleme
- **MNIST**: Tek kanallı (grayscale) el yazısı rakamlar (60.000 eğitim, 10.000 test örneği). Görüntüler `[0,1]` aralığında normalize edildi.
- **CIFAR-10**: RGB renkli 10 sınıflı nesne verisi (50.000 eğitim, 10.000 test örneği). Pretrained modeller için `224×224` boyutuna yeniden boyutlandırıldı ve ImageNet istatistikleriyle normalize edildi.

## Model 1: LeNet-5 Benzeri Mimari
- İki `Conv2d` katmanı (6 ve 16 filtre, çekirdek 5×5)
- `ReLU` aktivasyon, `MaxPool2d` (2×2)
- Tam bağlantılı katmanlar: 120, 84, 10 nöron

## Model 2: Geliştirilmiş LeNet (BatchNorm + Dropout)
- Model 1’in katman yapısı korunarak:
  - `BatchNorm2d` her konvolüsyon sonrası
  - `Dropout(0.5)` tam bağlantılı katmanlar arası

## Model 3: Pretrained VGG11
- `torchvision.models.vgg11(pretrained=True)` kullanıldı
- Son sınıflandırıcı katmana `Linear(4096,10)` eklendi
- CIFAR-10 üzerinde ince ayar (fine-tuning)

## Model 4: Hibrit CNN + SVM
1. Model 1 (LeNet) ile ara katmanlardan özellik çıkarımı
2. Elde edilen `.npy` dosyalarındaki özellikler SVM (`sklearn.svm.SVC`) ile sınıflandırıldı

## Eğitim Protokolü
- **Loss**: `nn.CrossEntropyLoss()`
- **Optimizer**: Adam (LeNet lr=0.001, VGG11 lr=0.0001)
- **Epoch Sayısı**: 10 (LeNet modelleri), 5 (VGG11)
- **Batch Size**: 64 (eğitim), 1000 (test)

# Sonuçlar
| Model                | Doğruluk (MNIST) | Doğruluk (CIFAR-10) | SVM Doğruluk (MNIST) |
|----------------------|------------------|---------------------|----------------------|
| LeNet                | %98.xx           | -                   | -                    |
| Improved LeNet       | %98.yy           | -                   | -                    |
| VGG11 (fine-tuned)   | -                | %85.zz              | -                    |
| LeNet + SVM          | -                | -                   | %97.ww               |

- **Şekil 1**: Eğitim ve doğrulama kayıp eğrileri
- **Şekil 2**: Eğitim ve doğrulama doğruluk eğrileri
- **Şekil 3**: Karmaşıklık matrisleri

# Tartışma
- Geliştirilmiş LeNet’te `BatchNorm` ve `Dropout` eklemesi, temel LeNet’e göre öğrenme kararlılığını artırdı ve aşırı uyum riskini azalttı.
- Pretrained VGG11, CIFAR-10 üzerinde iyi sonuç verse de hesaplama maliyeti ve eğitim süresi daha yüksekti.
- Hibrit LeNet+SVM yaklaşımı, hızlı özellik çıkarımı ile SVM’in güçlü sınıflandırma kabiliyetini birleştirerek rekabetçi performans gösterdi.

# Referanslar
1. LeCun, Y., Bottou, L., Bengio, Y., ve Haffner, P. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*. 86(11):2278–2324.
2. Simonyan, K. ve Zisserman, A. (2014). Very Deep Convolutional Networks for Large-Scale Image Recognition. *arXiv:1409.1556*.
3. Paszke, A., Gros, J., Chintala, S., ve diğerleri (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. *NeurIPS*.
4. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*. 12:2825–2830.

