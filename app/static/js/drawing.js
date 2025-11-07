window.onload = () => {
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    const clearBtn = document.getElementById('clear-btn');
    const predictBtn = document.getElementById('predict-btn');
    const resultDiv = document.getElementById('result');
    const modelSelect = document.getElementById('model-select');

    let drawing = false;
    ctx.lineWidth = 25; // Grosor del trazo
    ctx.lineCap = 'round';
    ctx.strokeStyle = 'black';

    // Fondo blanco (importante para la imagen)
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    function startDraw(e) {
        drawing = true;
        draw(e);
    }

    function endDraw() {
        drawing = false;
        ctx.beginPath(); // Resetea el trazo
    }

    function draw(e) {
        if (!drawing) return;
        
        // Obtener posición correcta del mouse
        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        ctx.lineTo(x, y);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(x, y);
    }

    function clearCanvas() {
        ctx.fillStyle = "white";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        resultDiv.textContent = '...';
    }
    
    async function predict() {
        resultDiv.textContent = 'Prediciendo...';
        
        // 1. Crear una imagen temporal de 28x28
        // (Esto es un truco: dibujamos el canvas grande en uno pequeño)
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = 28;
        tempCanvas.height = 28;
        const tempCtx = tempCanvas.getContext('2d');

        // Escalar la imagen de 280x280 a 28x28
        tempCtx.drawImage(canvas, 0, 0, 280, 280, 0, 0, 28, 28);
        
        // 2. Obtener la imagen como Data URL (base64)
        const imageData = tempCanvas.toDataURL('image/png');
        
        // 3. Obtener el modelo seleccionado
        const modelName = modelSelect.value;

        // 4. Enviar a la API
        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    model_name: modelName,
                    image_data: imageData
                })
            });

            const data = await response.json();

            if (data.error) {
                resultDiv.textContent = `Error: ${data.error}`;
            } else {
                resultDiv.textContent = `Predicción: ${data.prediction} (${data.confidence})`;
            }
        } catch (err) {
            resultDiv.textContent = 'Error en la conexión.';
            console.error(err);
        }
    }

    // Event Listeners
    canvas.addEventListener('mousedown', startDraw);
    canvas.addEventListener('mouseup', endDraw);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseout', endDraw); // Parar si el mouse sale

    clearBtn.addEventListener('click', clearCanvas);
    predictBtn.addEventListener('click', predict);
};