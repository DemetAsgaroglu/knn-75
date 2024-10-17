# KNN ile Pima Indians Diyabet Tahmini

Bu proje, **K-Nearest Neighbors (KNN)** algoritmasını kullanarak Pima Indian kadınlarının sağlık ölçümlerine dayalı diyabet tahmini yapmayı amaçlamaktadır. Veri seti, çeşitli sağlık ölçümlerini ve bu kadınların diyabet durumunu (1: diyabet, 0: diyabet değil) içermektedir. Projede veri ön işleme, keşifsel veri analizi, görselleştirme ve modelleme adımları bulunmaktadır.

## Veri Seti

Bu projede kullanılan veri seti: [Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database).

### Veri Setindeki Değişkenler:

- `Pregnancies`: Hamilelik sayısı
- `Glucose`: Plazma glukoz konsantrasyonu
- `BloodPressure`: Diyastolik kan basıncı (mm Hg)
- `SkinThickness`: Triceps deri kıvrım kalınlığı (mm)
- `Insulin`: 2 saatlik serum insülin seviyesi (mu U/ml)
- `BMI`: Vücut kitle indeksi (kg/m^2)
- `DiabetesPedigreeFunction`: Diyabet soyağacı fonksiyonu
- `Age`: Yaş (yıl)
- `Outcome`: Sınıf değişkeni (0: diyabet yok, 1: diyabet var)

## Proje Adımları

1. **Kütüphanelerin İçe Aktarılması**

    Veri işlemleri ve görselleştirme için gerekli kütüphaneler olan NumPy, pandas, matplotlib ve seaborn projeye dahil edilmiştir.

    ```python
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    ```

2. **Veri Setinin Yüklenmesi**

    Kaggle'dan alınan veri seti projeye yüklenmiştir.

    ```python
    df = pd.read_csv("/kaggle/input/pima-indians-diabetes-database/diabetes.csv")
    ```

3. **Veri Keşfi ve Görselleştirme**

    Veri setinin ilk 10 satırı gösterilmiş, temel bilgileri incelenmiş, eksik veri olup olmadığı kontrol edilmiş ve değişkenler arasındaki korelasyon hesaplanmıştır. Dağılım grafikleri ve `Outcome` değişkeninin dağılımı çizilmiştir.

    - Özelliklerin dağılımı:
      
      ![Özellik Dağılımı](images/feature_distribution.png)

    - Hedef değişkenin dağılımı (0: Diyabet Yok, 1: Diyabet Var):
      
      ![Outcome Dağılımı](images/outcome_distribution.png)

4. **Veri Ön İşleme**

    Veriler, **StandardScaler** kullanılarak ölçeklendirilmiştir.

    ```python
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df.drop(columns=['Outcome']))
    data_scaled = pd.DataFrame(data_scaled, columns=df.columns[:-1])
    data_scaled['Outcome'] = df['Outcome']
    ```

5. **Veri Setinin Eğitim ve Test Olarak Bölünmesi**

    Veri seti, %70 eğitim ve %30 test olacak şekilde bölünmüştür.

    ```python
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=15)
    ```

6. **Modelin Eğitilmesi**

    **K-Nearest Neighbors (KNN)** sınıflandırıcısı `n_neighbors=3` ile eğitilmiştir.

    ```python
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(x_train, y_train)
    ```

7. **Modelin Değerlendirilmesi**

    Modelin performansı doğruluk, karışıklık matrisi ve sınıflandırma raporu ile değerlendirilmiştir.

    ```python
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

    y_pred = knn.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred)

    print(f"Doğruluk Oranı: {accuracy}")
    print(f"Karışıklık Matrisi:\n {cm}")
    print(f"Sınıflandırma Raporu:\n {cr}")
    ```

    - **Doğruluk Oranı**: Modelin doğruluk oranı %75.
    - **Karışıklık Matrisi**: 
    
      ```
      [[130  30]
       [ 26  45]]
      ```

    - **Sınıflandırma Raporu**:

      ```
                precision    recall  f1-score   support
           0       0.83      0.81      0.82       160
           1       0.59      0.63      0.61        71
      ```

## Gelecekte Yapılabilecek İyileştirmeler

Modelin performansını artırmak için şunlar yapılabilir:
- Veri dengesizliği ile başa çıkmak için **SMOTE** veya **class weight dengeleme** gibi teknikler kullanılabilir.
- KNN modelinin parametrelerini iyileştirmek için **hiperparametre optimizasyonu** yapılabilir.
- **Random Forest**, **SVM** veya **XGBoost** gibi diğer sınıflandırma algoritmaları denenebilir.

## Sonuç

Bu projede, KNN algoritması kullanılarak sağlık verilerine dayalı diyabet tahmini başarıyla gerçekleştirilmiştir. Ancak, veri dengesizliğini gidermek ve model parametrelerini iyileştirmek, özellikle diyabet sınıfındaki hataları azaltmak için daha iyi sonuçlar elde edilmesini sağlayabilir.
