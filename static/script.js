document.addEventListener('DOMContentLoaded', () => {
  const chatMessages = document.getElementById('chatMessages');
  const userInput = document.getElementById('userInput');
  const sendBtn = document.getElementById('sendBtn');
  const typingIndicator = document.getElementById('typingIndicator');

  // Format time (e.g. 10:45 AM)
  const formatTime = () => {
    const now = new Date();
    return now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  // Add a message to the chat
  const appendMessage = (sender, text, metaData = null) => {
    const wrapper = document.createElement('div');
    wrapper.classList.add('message-wrapper', sender);

    const bubble = document.createElement('div');
    bubble.classList.add('message-bubble');
    bubble.textContent = text;
    wrapper.appendChild(bubble);

    // Add metadata row (timestamp + optional confidence)
    const meta = document.createElement('div');
    meta.classList.add('message-meta');
    
    let metaHTML = `<span>${formatTime()}</span>`;
    
    // If it's a bot message and we have confidence data, show it
    if (sender === 'bot' && metaData && metaData.intent !== 'greeting') {
      const confidencePercent = (metaData.confidence * 100).toFixed(1) + '%';
      metaHTML += `<span class="confidence-badge" title="Intent: ${metaData.intent}">${confidencePercent}</span>`;
    }
    
    meta.innerHTML = metaHTML;
    wrapper.appendChild(meta);

    chatMessages.appendChild(wrapper);
    scrollToBottom();
  };

  const scrollToBottom = () => {
    chatMessages.scrollTop = chatMessages.scrollHeight;
  };

  const setTyping = (isTyping) => {
    if (isTyping) {
      typingIndicator.classList.add('visible');
      scrollToBottom();
    } else {
      typingIndicator.classList.remove('visible');
    }
    
    // Disable/enable input while bot is typing
    userInput.disabled = isTyping;
    sendBtn.disabled = isTyping;
    if (!isTyping) {
      userInput.focus();
    }
  };

  const handleSend = async () => {
    const text = userInput.value.trim();
    if (!text) return;

    // 1. Add user message
    appendMessage('user', text);
    userInput.value = '';

    // 2. Show typing indicator
    setTyping(true);

    try {
      // 3. Send to Flask backend
      const response = await fetch('/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: text }),
      });

      if (!response.ok) {
        throw new Error('Network response was not ok');
      }

      const data = await response.json();

      // Simulate a small network delay so the typing animation is visible
      setTimeout(() => {
        setTyping(false);
        appendMessage('bot', data.response, {
          intent: data.intent,
          confidence: data.confidence
        });

        if (data.action === "close_session") {
          userInput.disabled = true;
          sendBtn.disabled = true;
          userInput.placeholder = "Chat session ended.";
        }
      }, 600 + Math.random() * 400); // 600-1000ms delay

    } catch (error) {
      console.error('Error:', error);
      setTyping(false);
      appendMessage('bot', 'Sorry, I am having trouble connecting to the server right now. Please try again later.');
    }
  };

  // Event Listeners
  sendBtn.addEventListener('click', handleSend);

  userInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
      handleSend();
    }
  });

  // Welcome message  // Initial greeting
  setTimeout(() => {
    setTyping(false);
    appendMessage('bot', 'Hello, I am Zeus, How may I assist you today?', {
      intent: 'greeting',
      confidence: 1.0
    });
  }, 1000);

});
