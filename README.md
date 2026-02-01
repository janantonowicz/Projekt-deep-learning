# Klasyfikacja obrazów Fashion MNIST

### Projekt Python Web & Deep Learning

## Website demo


https://github.com/user-attachments/assets/006cd1d9-c479-4982-be2d-e6a27c273dc5



## Co realizują sieci?

Klasyfikacja obrazu i przypasowanie go do jednej z 10 kategorii:

`t-shirt`, `trouser`, `pullover`, `dress`, `coat`, `sandal`, `shirt`, `sneaker`, `bag`, `ankle boot`

## Opis datasetu

### Pochodzenie danych

Wykorzystano zbiór danych Fashion MNIST.

### Fashion MNIST

- 70,000 czarnobiałych obrazów przdstawiających ubrania z 10 kategorii
    
- Wymiary obrazu: **28 × 28 pixels**

### Przetwarzanie wstępne:
    
- Dane wejściowe zostały znormalizowane do zakresu **[0, 1]**
    
- Kształt obrazów został zmieniony na format 28x28x1 (dodanie kanału głębi)
    

### Podział na zbiory

|Zbiór|Liczba próbek|Procent [%]|
|---|---|---|
|Treningowy|48,000|68.5%|
|Walidacyjny|12,000|17%|
|Testowy|10,000|14.5%|

## Opis architektur sieci oraz procesu uczenia

### Bloki konwolucyjne

**Blok 1**

- Dwie warstwy Conv2D (32 filters, kernel 3×3)
    
- MaxPooling2D
    

**Blok 2**

- Dwie warstwy Conv2D (64 filters, kernel 3×3)
    
- MaxPooling2D
    

Warstwa konwolucyjna skanuje obraz poszukując wzorców. Pierwszy blok szuka 32 wzorców (np. pionowe linie, kropki) Drugi blok szuka 64 bardziej złożonych wzorców.
Kernel 3 x 3 to rozmiar okna skanującego.

    

Po blokach konwolucyjnych zastosowano warstwę Flatten, która przekształca mapy cech do postaci wektora, a następnie warstwy w pełni połączone (Dense).

python

```
X = tf.keras.layers.Flatten()(X)
X = tf.keras.layers.Dense(128, activation="relu")(X)
```

Warstwa ukryta składa się z 128 neuronów z funkcją aktywacji ReLU, natomiast warstwa 
wyjściowa posiada 10 neuronów z funkcją Softmax, odpowiadających liczbie klas w zbiorze danych.

### Funkcja Straty

Jako funkcję straty wykorzystano Sparse Categorical Crossentropy, odpowiednią dla problemu wieloklasowej klasyfikacji z etykietami zapisanymi w postaci liczb całkowitych.


### Optymalizator

python

```
optimizer = tf.keras.optimizers.Adam(1e-3)
```

Wykorzystano optymalizator Adam.

Algorytm decyduje jak zmienić wagi sieci neuronowej na podstawie obliczonej straty – błędu, tak aby w następnym kroku wynik był lepszy.


### 🔹 Batch Size

```
BATCH_SIZE = 128
```

Model aktualizuje swoje wagi po przeanalizowaniu każdych z 128 obrazków.

### Liczba epok i Early Stopping

Aby osiągnąć jak najlepszy wynik wykorzystałem mechanizm **Early Stopping**. Dzięki temu proces uczenia zostanie automatycznie przerwany jeśli przez 10 kolejnych epok (`patience=10`) strata na zbiorze walidacyjnym nie ulegnie poprawie.
Dzięki `restore_best_weights` finalny model to ten który osiągnął najlepszy wynik, a nie ten z ostatniej epoki.
    

## Model z Augmentacją

- Losowe odbicie obrazu (RandomFlip)
- Losowe przesunięcie, obrót, przybliżenie oraz zmiana kontrastu
Dzięki temu zwiększamy różnorodność danych treningowych i zapobiegamy overfittingowi. Występuje mniejsza szansa że model nauczy się zbioru testowego „na pamięć”

## Porównanie modeli

Model z Augmentacją potrzebował znacznie więcej epok przy szkoleniu. Dzięki temu radzi sobie lepiej od prostego modelu. Oba modele jednak mają ograniczenia. Ponieważ są nauczone na prostym zbiorze fashion MNIST osiągną znacznie gorsze wyniki przy spotkaniu z obrazami w innym formacie (np. przy obrazach o większej rozdzielczości). Jako że jest to prosty model, przyjmuje tylko rozdzielczość 28x28 pikseli, zdjęcie wrzucone w większej rozdzielczości zostaje „ściśnięte” do wymaganych wymiarów przez co może stracić początkowe cechy charakterystyczne. 

### Augmented Model

<img width="945" height="357" alt="image" src="https://github.com/user-attachments/assets/257d1165-df07-477e-80c6-9dc8b6315507" />


```
313/313 ━━━━━━━━━━━━━━━━━━━━ 2s 5ms/step - accuracy: 0.9054 - loss: 0.2717
Test accuracy: 0.9064
```

### Simple Model

<img width="945" height="354" alt="image" src="https://github.com/user-attachments/assets/9fb6dc12-c0e9-40ff-95cd-fc2fb61968c8" />


```
313/313 ━━━━━━━━━━━━━━━━━━━━ 3s 6ms/step - accuracy: 0.9231 - loss: 0.2275
Test accuracy: 0.9226
```

### Wyniki

- **Prosty model osiągnął ~1.5% wyższy test accuracy**.
    
- W obu modelach występuje **overfitting**, pomimo funkcji early stopping, która pomogła ograniczyć zjawisko.
    
- Model z Augmentacją był szkolony na znacznie większej ilości epok.
    
- Oba modele napotykają problemy przy obrazach innych niż te ze zbioru treningowego (m.in., inny kolor tła lub rozdzielczość).
    

## Podsumowanie

Zgodnie z Testem to model prosty osiągnął lepszy wynik o 1,5% jednak w obu modelach występuje overfitting – dzięki zastosowaniu funkcji early stop udało się zminimalizować zjawisko (nadal jednak występuje). Można zauważyć znacznie gorsze wyniki przy obrazach z tłem w innym kolorze niż ze zbioru treningowego
