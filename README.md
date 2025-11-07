
# Comparación de Optimizadores en PyTorch con MNIST y FastAPI

[](https://pytorch.org/)
[](https://fastapi.tiangolo.com/)

Este proyecto tiene un doble objetivo:

1.  **Educativo:** Realizar un experimento para comparar la convergencia, velocidad y precisión de cinco algoritmos de optimización de PyTorch (SGD, Adam, RMSprop, Adagrad, AdamW) en la tarea de clasificación de MNIST.
2.  **Práctico:** Desplegar los modelos entrenados en una aplicación web interactiva usando FastAPI, donde el usuario puede dibujar un dígito y recibir una predicción en tiempo real.

-----

## 🔬 Resultados del Experimento

El experimento entrena un modelo de red neuronal simple (MLP) en el dataset MNIST, una vez por cada optimizador, usando los mismos hiperparámetros de base. La visualización principal compara la caída de la función de pérdida (Loss) a lo largo de los pasos de entrenamiento para cada optimizador.

**El gráfico generado por el script (`results/loss_comparison.png`) se mostrará aquí:**

*(Esta imagen se genera automáticamente en la carpeta `results/` al ejecutar `src/train.py`)*

-----

## 🧠 Fundamentos: Una Breve Mirada a los Optimizadores

Un **optimizador** es el algoritmo que ajusta los pesos (parámetros) de la red neuronal para minimizar la función de pérdida. Es el "motor" que impulsa el aprendizaje. Los gradientes (calculados por *backpropagation*) nos dicen la *dirección* del ascenso, y el optimizador decide *cómo y cuánto* movernos en la dirección opuesta.

### 1\. SGD (Stochastic Gradient Descent)

  * **Qué es:** El algoritmo fundamental. Actualiza los pesos basándose únicamente en el gradiente del lote actual.
  * **Concepto:** `peso = peso - (learning_rate * gradiente)`
  * **Pros:** Simple, fácil de entender, computacionalmente ligero.
  * **Contras:** Puede ser ruidoso y lento para converger. Puede atascarse en mínimos locales o puntos de silla.
  * **Variación (la que usamos):** **SGD con Momentum** añade una fracción del vector de actualización anterior, lo que ayuda a suavizar la trayectoria y acelerar la convergencia a través de "valles".

### 2\. Adagrad (Adaptive Gradient)

  * **Qué es:** Un optimizador adaptativo. Mantiene *learning rates* separados para cada parámetro y los adapta basándose en los gradientes pasados.
  * **Concepto:** Da "pasos" más pequeños para parámetros que han recibido gradientes grandes y frecuentes (se "cansa" rápido).
  * **Pros:** Excelente para datos dispersos (sparse data), ya que presta más atención a características raras.
  * **Contras:** Su *learning rate* global decae agresivamente y puede llegar a ser tan pequeño que el entrenamiento se detiene prematuramente.

### 3\. RMSprop (Root Mean Square Propagation)

  * **Qué es:** La solución al problema de Adagrad. En lugar de sumar *todos* los gradientes cuadrados pasados, utiliza un **promedio móvil exponencial**.
  * **Concepto:** Mantiene la adaptabilidad por parámetro, pero evita que el *learning rate* muera tan rápido.
  * **Pros:** Convergencia rápida y estable en muchos problemas.
  * **Contras:** Puede ser sensible a la elección del *learning rate* inicial.

### 4\. Adam (Adaptive Moment Estimation)

  * **Qué es:** El estándar de facto actual en *Deep Learning*. Combina lo mejor de dos mundos: **Momentum** (primer momento, la "velocidad") y **RMSprop** (segundo momento, la "adaptabilidad" del *learning rate*).
  * **Concepto:** Mantiene un promedio móvil tanto del gradiente como de su cuadrado.
  * **Pros:** Generalmente converge más rápido que otros métodos y es menos sensible a la elección de hiperparámetros.
  * **Contras:** Requiere más memoria para almacenar sus "momentos" por cada parámetro.

### 5\. AdamW (Adam with Weight Decay)

  * **Qué es:** Una corrección a Adam. La implementación original de Adam mezclaba la regularización L2 (*weight decay*) con la actualización del gradiente, lo cual no es óptimo para optimizadores adaptativos.
  * **Concepto:** Desacopla el *weight decay* de la actualización adaptativa, aplicándolo directamente al peso al final del paso.
  * **Pros:** A menudo conduce a una mejor generalización (mejor rendimiento en el conjunto de prueba) que el Adam estándar.

-----

## 📁 Estructura del Proyecto

```
/
├── app/                  # Código del backend (FastAPI) y frontend
│   ├── static/js/
│   │   └── drawing.js    # Lógica del canvas
│   ├── templates/
│   │   └── index.html    # Frontend HTML
│   └── main.py           # Servidor FastAPI y lógica de predicción
├── src/                  # Código de Machine Learning
│   ├── data_loader.py    # Funciones para cargar MNIST
│   ├── model.py          # Definición de la clase SimpleNN
│   └── train.py          # Script principal para entrenar y comparar
├── models/               # (Ignorado por Git) Aquí se guardan los .pth
├── results/              # (Ignorado por Git) Aquí se guardan los gráficos
├── .gitignore            # Ignora modelos, datos y caches
├── README.md             # ¡Este archivo!
└── requirements.txt      # Dependencias de Python
```

-----

## 🚀 Instalación y Uso

Sigue estos pasos para poner en marcha el proyecto.

### 1\. Preparar el Entorno

1.  **Clona el repositorio:**

    ```bash
    git clone https://github.com/TuUsuario/pytorch-mnist-optimizers.git
    cd pytorch-mnist-optimizers
    ```

2.  **(Recomendado) Crea un entorno virtual:**

    ```bash
    python -m venv venv
    # En Windows:
    .\venv\Scripts\activate
    # En macOS/Linux:
    source venv/bin/activate
    ```

3.  **Instala las dependencias:**

    ```bash
    pip install -r requirements.txt
    ```

### 2\. Entrenar los Modelos

Este es el paso más importante. El script entrenará los 5 modelos secuencialmente y guardará los artefactos (`.pth`) y el gráfico de comparación.

```bash
# Navega a la carpeta 'src'
cd src/

# Ejecuta el script de entrenamiento
python train.py
```

Al finalizar, deberías tener 5 archivos `.pth` en la carpeta `models/` y un `loss_comparison.png` en la carpeta `results/`.

### 3\. Ejecutar la Aplicación Web

Una vez entrenados los modelos, puedes iniciar el servidor FastAPI.

```bash
# Navega a la carpeta 'app' (desde la raíz del proyecto)
cd app/

# Inicia el servidor
uvicorn main:app --reload
```

### 4\. Probar el Proyecto

Abre tu navegador web y ve a **`http://127.0.0.1:8000`**.

¡Ahora puedes dibujar un dígito, seleccionar uno de los modelos entrenados (basados en el optimizador) y ver la predicción en tiempo real\!
