// Generate one session ID per browser tab/session so this
// user's conversation stays separate from anyone else's.
const sessionId = crypto.randomUUID();


async function sendQuestion() {

    const input = document.getElementById("question");

    const question = input.value.trim();

    if (!question) {
        return;
    }


    // Display user question

    addMessage(
        "You",
        question,
        "user"
    );


    // Clear input

    input.value = "";


    try {

        const response = await fetch(
            "/chat",
            {
                method: "POST",

                headers: {
                    "Content-Type": "application/json"
                },

                body: JSON.stringify({
                    question: question,
                    session_id: sessionId
                })
            }
        );


        const data = await response.json();


        // Display AI answer

        addMessage(
            "AI",
            data.answer,
            "ai"
        );


    } catch (error) {

        console.error(error);

        addMessage(
            "AI",
            "Unable to connect to the server.",
            "ai"
        );

    }
}


function addMessage(sender, message, type) {

    const chatContainer =
        document.getElementById("chat-container");


    const messageDiv =
        document.createElement("div");


    messageDiv.classList.add(
        "message",
        type
    );


    messageDiv.innerHTML =
        `<strong>${sender}:</strong> ${message}`;


    chatContainer.appendChild(
        messageDiv
    );


    chatContainer.scrollTop =
        chatContainer.scrollHeight;
}