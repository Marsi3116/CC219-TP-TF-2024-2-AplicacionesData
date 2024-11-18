document.addEventListener("DOMContentLoaded", () => {
    const modelSelector = document.getElementById("model-selector");
    const chatContent = document.getElementById("chat-content");
    const userInput = document.getElementById("user-input");
    const sendBtn = document.getElementById("send-btn");

    // Función para cargar los modelos disponibles desde el servidor
    fetch("/models")
        .then(response => response.json())
        .then(models => {
            models.forEach(model => {
                const option = document.createElement("option");
                option.value = model;
                option.textContent = model;
                modelSelector.appendChild(option);
            });

            // Seleccionar el primer modelo por defecto
            if (models.length > 0) {
                modelSelector.value = models[0];
            }
        })
        .catch(error => {
            console.error("Error al cargar los modelos:", error);
        });

    // Función para enviar el mensaje al backend
    const sendMessage = () => {
        const message = userInput.value.trim();
        const selectedModel = modelSelector.value;

        if (!message) {
            alert("Por favor, escribe un mensaje.");
            return;
        }

        fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ message, model: selectedModel }),
        })
        .then(response => response.json())
        .then(data => {
            // Mostrar el mensaje del usuario
            const userDiv = document.createElement("div");
            userDiv.className = "user-message";
            userDiv.textContent = message;
            chatContent.appendChild(userDiv);

            // Mostrar la respuesta del chatbot
            const botDiv = document.createElement("div");
            botDiv.className = "bot-message";
            botDiv.textContent = data.response;
            chatContent.appendChild(botDiv);

            // Desplazar el contenido hacia el final
            chatContent.scrollTop = chatContent.scrollHeight;

            // Limpiar el campo de entrada
            userInput.value = "";
        })
        .catch(error => {
            console.error("Error al enviar el mensaje:", error);
        });
    };

    // Evento para el botón "Enviar"
    sendBtn.addEventListener("click", sendMessage);

    // Permitir el envío de mensajes con la tecla "Enter"
    userInput.addEventListener("keypress", (event) => {
        if (event.key === "Enter") {
            sendMessage();
        }
    });
});
