import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import time  # Importar utilidades para medir el rendimiento (tiempo de cómputo).

# Importamos nuestros módulos locales
from data_loader import get_data_loaders
from model import SimpleNN

# --- Parámetros Globales del Experimento ---
NUM_EPOCHS = 10
LEARNING_RATE = 0.01  # Tasa de aprendizaje base (algunos optimizadores la adaptarán)
BATCH_SIZE = 32
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Establecer directorios para artefactos de entrenamiento (modelos) y visualizaciones.
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# --- Función de Bucle de Entrenamiento ---
def train_model(model, optimizer, train_loader, criterion, num_epochs=NUM_EPOCHS):
    """
    Itera sobre el dataset de entrenamiento durante 'num_epochs' para un 
    modelo y optimizador dados, actualizando los pesos.
    """
    # Activa el modo 'entrenamiento' del modelo. Esto es crucial para capas
    # como Dropout o BatchNorm, que se comportan de manera diferente durante la inferencia.
    model.train() 
    
    epoch_losses = []
    
    print(f"--- Entrenando con {optimizer.__class__.__name__} ---")
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        steps = 0
        for i, (inputs, labels) in enumerate(train_loader):
            
            # 1. Transfiere el lote de datos (imágenes y etiquetas) al dispositivo de cómputo (GPU/CPU).
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            # 2. Resetea los gradientes acumulados de la iteración anterior.
            # PyTorch acumula gradientes por defecto; no hacerlo llevaría a cómputos incorrectos.
            optimizer.zero_grad()

            # 3. Forward pass: Propaga la entrada a través de la red para obtener las predicciones (logits).
            outputs = model(inputs)
            
            # 4. Calcula la pérdida (ej. Cross-Entropy) comparando logits con etiquetas verdaderas.
            loss = criterion(outputs, labels)

            # 5. Backward pass: Calcula el gradiente de la pérdida con respecto a cada parámetro (w) del modelo.
            # Esta es la esencia de la retropropagación (backpropagation).
            loss.backward()

            # 6. Actualiza los pesos del modelo.
            # Aquí es donde el optimizador (ej. Adam, SGD) aplica su regla de actualización específica
            # usando los gradientes (calculados en .backward()) y su estado interno.
            optimizer.step()

            # --- Recolección de métricas ---
            running_loss += loss.item()
            steps += 1
            
            if (i+1) % 250 == 0:
                epoch_losses.append(running_loss / 250)
                running_loss = 0.0
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss promedio: {sum(epoch_losses[-steps//250:]) / (steps//250):.4f}')
        
    print("Entrenamiento finalizado.")
    return epoch_losses

# --- Función de Bucle de Evaluación ---
def evaluate_model(model, test_loader):
    """
    Evalúa la precisión (accuracy) del modelo sobre un conjunto de datos (ej. test set).
    No realiza actualizaciones de pesos.
    """
    # Activa el modo 'evaluación'. Desactiva Dropout y fija los promedios de BatchNorm.
    model.eval() 
    correct = 0
    total = 0
    
    # Desactiva el cálculo de gradientes.
    # Esto es crucial para la inferencia: reduce el uso de memoria y acelera el cómputo.
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            
            # Obtiene la predicción final (la clase con el logit más alto)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy

# --- Bloque principal de experimentación ---
if __name__ == "__main__":
    
    print(f"Usando dispositivo de cómputo: {DEVICE}")
    train_loader, test_loader = get_data_loaders(BATCH_SIZE)
    
    # Define la función de pérdida (Loss Function).
    # CrossEntropyLoss es estándar para clasificación multiclase, ya que combina Softmax y NLLLoss.
    criterion = nn.CrossEntropyLoss()
    
    # Define el 'Experimento': un diccionario que mapea los optimizadores a comparar.
    # Cada optimizador tiene su propia clase y un conjunto de hiperparámetros (ej. 'lr').
    optimizers_to_test = {
        # SGD (Stochastic Gradient Descent): El optimizador fundamental. 
        # 'momentum' le ayuda a acelerar en la dirección correcta y superar mínimos locales.
        "SGD": (optim.SGD, {"lr": 0.01, "momentum": 0.9}),
        
        # Adam (Adaptive Moment Estimation): Combina 'momentum' (primer momento) y 'RMSprop' (segundo momento).
        # Es un estándar de facto por su rápida convergencia y menor sensibilidad al 'lr' inicial.
        "Adam": (optim.Adam, {"lr": 0.001}),
        
        # RMSprop: Optimización adaptativa. Escala el learning rate por la magnitud de gradientes pasados
        # (divide por un promedio móvil de gradientes cuadrados).
        "RMSprop": (optim.RMSprop, {"lr": 0.001}),
        
        # Adagrad (Adaptive Gradient): Acumula gradientes cuadrados. Bueno para datos dispersos (NLP),
        # pero su LR tiende a decaer agresivamente a cero.
        "Adagrad": (optim.Adagrad, {"lr": 0.01}),
        
        # AdamW: Una corrección a Adam que desacopla la 'weight decay' (regularización L2)
        # de la actualización de gradientes adaptativa, mejorando la generalización.
        "AdamW": (optim.AdamW, {"lr": 0.001})
    }

    # Estructuras de datos para almacenar las métricas de cada 'run' (ejecución).
    results = {}
    accuracies = {}
    training_times = {}

    # Itera sobre cada configuración de optimizador definida en el experimento.
    for name, (opt_class, opt_params) in optimizers_to_test.items():
        
        print("\n" + "="*30)
        print(f"PROBANDO OPTIMIZADOR: {name}")
        print("="*30)

        # 1. Es CRUCIAL reinicializar los pesos del modelo por cada optimizador.
        #    De lo contrario, estaríamos continuando el entrenamiento sobre un modelo ya entrenado.
        model = SimpleNN()
        
        # 2. Mueve la arquitectura y los pesos del modelo al dispositivo de cómputo (GPU).
        model.to(DEVICE)
        
        # 3. Instancia el optimizador, pasándole los parámetros del modelo (que ya están en la GPU).
        #    Esto es vital: si el modelo está en CPU, optimizadores como Adagrad crearán su 'estado' en CPU.
        optimizer = opt_class(model.parameters(), **opt_params)
        
        # --- Medición de Eficiencia Computacional ---
        start_time = time.time()
        
        # Ejecuta el bucle de entrenamiento completo.
        train_losses = train_model(model, optimizer, train_loader, criterion, NUM_EPOCHS)
        
        end_time = time.time()
        total_train_time = end_time - start_time
        # --- Fin de Medición ---

        # Almacena la curva de convergencia (historial de loss)
        results[name] = train_losses
        # Almacena el tiempo total (wall-clock time)
        training_times[name] = total_train_time
        
        # Evalúa el rendimiento final del modelo en datos nunca vistos.
        accuracy = evaluate_model(model, test_loader)
        accuracies[name] = accuracy
        
        # Imprime el reporte final para esta ejecución.
        print(f"Accuracy (Test Set) con {name}: {accuracy:.2f}%")
        print(f"Tiempo total de entrenamiento: {total_train_time:.2f} segundos")
        
        # Serializa y guarda el 'state_dict' (solo los pesos aprendidos) del modelo.
        model_path = os.path.join("models", f"model_{name.lower()}.pth")
        torch.save(model.state_dict(), model_path)
        print(f"Artefacto del modelo guardado en: {model_path}")

    # --- Visualización y Comparativa de Métricas ---
    plt.figure(figsize=(12, 7))
    for name, losses in results.items():
        # Genera una leyenda descriptiva para cada curva, incluyendo las métricas clave de rendimiento.
        label_text = (
            f"{name} (Acc: {accuracies[name]:.2f}%, "
            f"Tiempo: {training_times[name]:.2f}s)"
        )
        plt.plot(losses, label=label_text)
    
    plt.title("Comparación de Convergencia de Optimizadores en MNIST")
    # El eje X representa el progreso del entrenamiento (no épocas, sino pasos/batches).
    plt.xlabel("Pasos de entrenamiento (x250 batches)")
    plt.ylabel("Cross-Entropy Loss (Pérdida)")
    plt.legend()
    plt.grid(True)
    
    plot_path = os.path.join("results", "loss_comparison.png")
    plt.savefig(plot_path)
    print(f"\nGráfico de comparación guardado en: {plot_path}")
    plt.show()