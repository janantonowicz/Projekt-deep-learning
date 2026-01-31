import os
import io
import base64
import numpy as np
from PIL import Image, ImageOps
from flask import Flask, request, render_template
import tensorflow as tf

app = Flask(__name__)

# Konfiguracja ścieżek do modeli
MODEL_PATH_1 = os.path.join('models', 'model_with_aug.keras')
MODEL_PATH_2 = os.path.join('models', 'simple_fashion_mnist_model(1).keras')

# Ładowanie modeli
model1 = None
model2 = None

try:
    if os.path.exists(MODEL_PATH_1):
        model1 = tf.keras.models.load_model(MODEL_PATH_1)
        print(f"Załadowano model: {MODEL_PATH_1}")
    else:
        print(f"Nie znaleziono modelu: {MODEL_PATH_1}")

    if os.path.exists(MODEL_PATH_2):
        model2 = tf.keras.models.load_model(MODEL_PATH_2)
        print(f"Załadowano model: {MODEL_PATH_2}")
    else:
        print(f"Nie znaleziono modelu: {MODEL_PATH_2}")
        
except Exception as error:
    print(f"Błąd podczas ładowania modeli: {error}")

# Ładowanie danych Fashion MNIST (do funkcji losowania)
x_test = None
y_test = None

try:
    (_, _), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    print("Załadowano dane Fashion MNIST (zbiór testowy)")
except Exception as error:
    print(f"Błąd ładowania danych Fashion MNIST: {error}")

# Nazwy klas dla Fashion MNIST
CLASS_NAMES = ["t-shirt","trouser","pullover","dress","coat","sandal","shirt","sneaker","bag","ankle boot"]

# Statystyki sesji (działają tylko przy użyciu funkcji losowania, gdzie znamy poprawne etykiety)
STATS = {
    'model1': {'correct': 0, 'incorrect': 0},
    'model2': {'correct': 0, 'incorrect': 0}
}

def prepare_image(image):
    """
    Wersja uproszczona z detekcją tła:
    1. Konwersja do skali szarości (L)
    2. Resize do 28x28 (bez zachowania proporcji)
    3. Inteligentna inwersja: sprawdzamy czy obraz trzeba odwrócić
    4. Normalizacja
    """
    img = image.convert('L')
    img = img.resize((28, 28))
    
    # Dodajemy autokontrast - pomaga oddzielić obiekt od tła na prawdziwych zdjęciach
    img = ImageOps.autocontrast(img)
    
    # Sprawdzamy średnią jasność pikseli
    # Jeśli > 127, to znaczy że obraz jest jasny (np. białe tło) -> robimy inwersję
    # Jeśli < 127, to znaczy że obraz jest ciemny (np. Fashion MNIST) -> zostawiamy
    img_array_temp = np.array(img)
    if np.mean(img_array_temp) > 127:
        img = ImageOps.invert(img)
    
    img_array = np.array(img).astype('float32') / 255.0
    
    img_array = np.expand_dims(img_array, axis=0)
    img_array = np.expand_dims(img_array, axis=-1)
    
    return img_array

@app.route('/', methods=['GET', 'POST'])
def index():
    predictions = None
    img_b64 = None
    error = None
    
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error='Nie przesłano pliku')
        
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error='Nie wybrano pliku')
            
        if file:
            try:
                image = Image.open(file.stream)
                
                # Przygotowanie obrazu dla modelu
                input_tensor = prepare_image(image)
                
                preds_data = {}
                
                # Inferencja Model 1
                if model1:
                    pred1 = model1.predict(input_tensor)
                    class_idx1 = np.argmax(pred1)
                    class1 = CLASS_NAMES[class_idx1]
                    conf1 = f"{np.max(pred1)*100:.2f}%"
                    preds_data['model1'] = {'class': class1, 'confidence': conf1}
                else:
                    preds_data['model1'] = {'class': 'Błąd modelu', 'confidence': '-'}

                # Inferencja Model 2
                if model2:
                    pred2 = model2.predict(input_tensor)
                    class_idx2 = np.argmax(pred2)
                    class2 = CLASS_NAMES[class_idx2]
                    conf2 = f"{np.max(pred2)*100:.2f}%"
                    preds_data['model2'] = {'class': class2, 'confidence': conf2}
                else:
                    preds_data['model2'] = {'class': 'Błąd modelu', 'confidence': '-'}

                predictions = preds_data
                
                # Przygotowanie obrazu do wyświetlenia (base64)
                # Reset streamu obrazu, bo był czytany
                # Ale image object mamy już w pamięci
                buffered = io.BytesIO()
                # Zapisujemy oryginalny obraz do wyświetlenia (lub przetworzony?)
                # Wyświetlmy oryginalny, żeby użytkownik widział co wysłał
                image.save(buffered, format="PNG")
                img_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                
            except Exception as e:
                error = f"Błąd przetwarzania: {str(e)}"

    return render_template('index.html', predictions=predictions, image_data=img_b64, error=error, stats=STATS)

@app.route('/random', methods=['POST'])
def random_sample():
    predictions = None
    img_b64 = None
    error = None
    true_label_name = None

    if x_test is None or y_test is None:
        error = "Dane testowe nie zostały załadowane."
        return render_template('index.html', error=error)
    
    try:
        # Losowanie obrazu
        idx = np.random.randint(len(x_test))
        sample_image = x_test[idx] # (28, 28)
        true_label = y_test[idx]
        true_label_name = CLASS_NAMES[true_label]
        
        # Przygotowanie do modelu (normalizacja i wymiary)
        img_array = sample_image.astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0) # (1, 28, 28)
        input_tensor = np.expand_dims(img_array, axis=-1) # (1, 28, 28, 1)
        
        preds_data = {}
        
        # Inferencja Model 1
        if model1:
            pred1 = model1.predict(input_tensor)
            class_idx1 = np.argmax(pred1)
            class1 = CLASS_NAMES[class_idx1]
            conf1 = f"{np.max(pred1)*100:.2f}%"
            preds_data['model1'] = {'class': class1, 'confidence': conf1}
            
            # Aktualizacja statystyk dla Modelu 1
            if class_idx1 == true_label:
                STATS['model1']['correct'] += 1
            else:
                STATS['model1']['incorrect'] += 1
                
        else:
            preds_data['model1'] = {'class': 'Błąd modelu', 'confidence': '-'}

        # Inferencja Model 2
        if model2:
            pred2 = model2.predict(input_tensor)
            class_idx2 = np.argmax(pred2)
            class2 = CLASS_NAMES[class_idx2]
            conf2 = f"{np.max(pred2)*100:.2f}%"
            preds_data['model2'] = {'class': class2, 'confidence': conf2}
            
            # Aktualizacja statystyk dla Modelu 2
            if class_idx2 == true_label:
                STATS['model2']['correct'] += 1
            else:
                STATS['model2']['incorrect'] += 1
                
        else:
            preds_data['model2'] = {'class': 'Błąd modelu', 'confidence': '-'}

        predictions = preds_data
        
        # Przygotowanie obrazu do wyświetlenia
        # sample_image to numpy array (28, 28) Musimy go zamienić na obrazek PIL
        img_display = Image.fromarray(sample_image.astype('uint8'), mode='L')
        # powiększenie w celu lepszego wyświetlania
        img_display = img_display.resize((200, 200), resample=Image.Resampling.NEAREST)
        
        buffered = io.BytesIO()
        img_display.save(buffered, format="PNG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    except Exception as e:
        error = f"Błąd podczas losowania: {str(e)}"
        
    return render_template('index.html', predictions=predictions, image_data=img_b64, error=error, true_label=true_label_name, stats=STATS)

if __name__ == '__main__':
    # Uruchomienie aplikacji
    app.run(debug=True, port=5000)
