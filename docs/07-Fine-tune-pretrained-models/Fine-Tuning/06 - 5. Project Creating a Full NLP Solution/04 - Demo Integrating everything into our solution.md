
<!-- Aero-Navigation-Start -->
[🏠 Home](../../../../index.md) > [07 Fine tune pretrained models](../../../index.md) > [Fine Tuning](../../index.md) > [06   5. Project Creating a Full NLP Solution](../index.md)

---
### 🧭 Điều hướng nhanh

- [🏠 Cổng tài liệu](../../../../index.md)
- [📚 Module 01: LLM Course](../../../../01-LLM_Course/index.md)
- [🔢 Module 02: Tokenization](../../../../02-Words-to-tokens-to-numbers/index.md)
- [🏗️ Module 04: Build GPT](../../../../04-buildGPT/index.md)
- [🎯 Module 07: Fine-tuning](../../../../07-Fine-tune-pretrained-models/index.md)
- [🔍 Module 19: AI Safety](../../../../19-AI-safety/index.md)
- [🐍 Module 20: Python for AI](../../../../20-Python-Colab-notebooks/index.md)
---
<!-- Aero-Navigation-End -->
# Demo Tích Hợp Mọi Thứ vào Giải Pháp

## Tổng Quan

Trong bài hướng dẫn thực hành cuối cùng này, chúng ta sẽ xem xét cách tạo một chatbot hoàn chỉnh sử dụng ba mô hình đã được fine-tune: phân tích cảm xúc, tóm tắt và trả lời câu hỏi. Chatbot sẽ được triển khai sử dụng Flask.

## 1. Giới Thiệu Kiến Trúc

### 1.1 Các Thành Phần

Một chatbot hoàn chỉnh bao gồm:

| Thành phần | Chức năng | Mô hình |
|------------|-----------|---------|
| Sentiment Analysis | Phân tích cảm xúc người dùng | SST-2 fine-tuned |
| Summarization | Tóm tắt cuộc trò chuyện | CNN DailyMail fine-tuned |
| Question Answering | Trả lời câu hỏi | SQuAD fine-tuned |
| Text Generation | Tạo phản hồi | GPT-2 |

### 1.2 Luồng Hoạt Động

```
User Input → Sentiment Analysis → Response Generation
                  ↓
            Summarization → Conversation History
                  ↓
            Question Answering → (if question detected)
```

## 2. Triển Khai Chi Tiết

### 2.1 Cài Đặt Flask

```python
!pip install flask requests

from flask import Flask, request, jsonify
import requests
```

### 2.2 Tải Các Mô Hình

```python
from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer

# Tải mô hình phân tích cảm xúc
sentiment_model = TFAutoModelForSeq2SeqLM.from_pretrained("path/to/sentiment_model")
sentiment_tokenizer = AutoTokenizer.from_pretrained("path/to/sentiment_tokenizer")

# Tải mô hình tóm tắt
summarization_model = TFAutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
summarization_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")

# Tải mô hình QA
qa_model = TFAutoModelForSeq2SeqLM.from_pretrained("path/to/qa_model")
qa_tokenizer = AutoTokenizer.from_pretrained("path/to/qa_tokenizer")

# Tải mô hình text generation (GPT-2)
from transformers import TFGPT2LMHeadModel
generation_model = TFGPT2LMHeadModel.from_pretrained("gpt2")
generation_tokenizer = AutoTokenizer.from_pretrained("gpt2")
```

### 2.3 Định Nghĩa Các Hàm Chức Năng

#### 2.3.1 Tóm Tắt Cuộc Trò Chuyện

```python
def summarize_conversation(conversation_history):
    """Tóm tắt toàn bộ cuộc trò chuyện"""
    prompt = "summarize: " + " ".join(conversation_history)
    
    inputs = summarization_tokenizer(prompt, return_tensors="tf", max_length=512)
    outputs = summarization_model.generate(**inputs, max_length=150)
    
    return summarization_tokenizer.decode(outputs[0], skip_special_tokens=True)
```

#### 2.3.2 Trả Lời Câu Hỏi

```python
def answer_question(context, question):
    """Trả lời câu hỏi dựa trên context"""
    prompt = f"{context} Question: {question} Answer:"
    
    inputs = qa_tokenizer(prompt, return_tensors="tf", max_length=384)
    outputs = qa_model.generate(**inputs, max_length=128)
    
    return qa_tokenizer.decode(outputs[0], skip_special_tokens=True)
```

#### 2.3.3 Tạo Phản Hồi

```python
def generate_response(user_message, sentiment):
    """Tạo phản hồi dựa trên cảm xúc"""
    if sentiment == "negative":
        prompt = f"The user is angry. Their message: {user_message}"
    else:
        prompt = user_message
    
    inputs = generation_tokenizer(prompt, return_tensors="tf")
    outputs = generation_model.generate(**inputs, max_length=100)
    
    return generation_tokenizer.decode(outputs[0], skip_special_tokens=True)
```

#### 2.3.4 Phân Tích Cảm Xúc

```python
def analyze_sentiment(message):
    """Phân tích cảm xúc của tin nhắn"""
    prompt = f"sst2 sentence: {message}"
    
    inputs = sentiment_tokenizer(prompt, return_tensors="tf")
    outputs = sentiment_model.generate(**inputs)
    
    result = sentiment_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return "negative" if "negative" in result.lower() else "positive"
```

### 2.4 Tạo Flask App

```python
app = Flask(__name__)

# Lưu trữ lịch sử cuộc trò chuyện
conversation_history = []
conversation_summary = []

@app.route('/reset', methods=['POST'])
def reset():
    """Reset cuộc trò chuyện"""
    global conversation_history, conversation_summary
    
    # Tóm tắt cuộc trò chuyện trước khi reset
    if conversation_history:
        summary = summarize_conversation(conversation_history)
        conversation_summary.append(summary)
    
    conversation_history = []
    return jsonify({"summary": summary})

@app.route('/greet', methods=['GET'])
def greet():
    """Chào hỏi"""
    return jsonify({"message": "Hello! How can I help you today?"})

@app.route('/chat', methods=['POST'])
def chat():
    """Xử lý tin nhắn chat"""
    global conversation_history
    
    # Lấy tin nhắn từ request
    user_message = request.json.get('message')
    
    # Thêm vào lịch sử
    conversation_history.append(user_message)
    
    # Phân tích cảm xúc
    sentiment = analyze_sentiment(user_message)
    
    # Kiểm tra loại tin nhắn
    if "summarize" in user_message.lower():
        response = summarize_conversation(conversation_history)
    
    elif "?" in user_message:
        # Sử dụng QA model
        context = " ".join(conversation_history[-5:])
        response = answer_question(context, user_message)
    
    else:
        # Sử dụng text generation
        response = generate_response(user_message, sentiment)
    
    # Lưu phản hồi vào lịch sử
    conversation_history.append(response)
    
    return jsonify({
        "response": response,
        "sentiment": sentiment
    })

if __name__ == '__main__':
    app.run(port=5000)
```

## 3. Kiểm Tra Chatbot

### 3.1 Chạy Chatbot

```python
# Khởi động Flask app trong background
import subprocess
subprocess.Popen(["python", "chatbot.py"], stdout=open("nohup.out", "w"))
```

### 3.2 Các Lệnh Test

```python
import requests

BASE_URL = "http://localhost:5000"

# Test greeting
response = requests.get(f"{BASE_URL}/greet")
print(response.json())

# Test chat - câu hỏi
response = requests.post(f"{BASE_URL}/chat", json={
    "message": "What is the capital of France?"
})
print(response.json())

# Test chat - tóm tắt
response = requests.post(f"{BASE_URL}/chat", json={
    "message": "Summarize this conversation"
})
print(response.json())

# Test chat - tin nhắn thường
response = requests.post(f"{BASE_URL}/chat", json={
    "message": "I want to build an app"
})
print(response.json())

# Test reset
response = requests.post(f"{BASE_URL}/reset")
print(response.json())
```

### 3.3 Kết Quả Mẫu

| Loại test | Input | Output |
|-----------|-------|--------|
| Greeting | - | "Hello! How can I help you today?" |
| Question | "What is the capital of France?" | "Paris" |
| Summarization | "Summarize..." | "The Bastille was a fortress..." |
| Text gen | "I want to build an app" | "What do you want?" |
| Reset | - | Summary of conversation |

## 4. Phân Tích Kiến Trúc

### 4.1 Sơ Đồ Luồng Dữ Liệu

```
┌─────────────┐     ┌──────────────────┐
│  User      │────>│  Sentiment        │
│  Input     │     │  Analysis         │
└─────────────┘     └────────┬─────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
         ▼                   ▼                   ▼
┌─────────────┐     ┌──────────────────┐  ┌─────────────┐
│ Summarize   │     │  Question        │  │   Text      │
│ (if asked)  │     │  Answer          │  │  Generation │
└─────────────┘     │  (if "?")        │  └─────────────┘
                    └──────────────────┘
```

### 4.2 Ưu và Nhược Điểm

| Ưu điểm | Nhược điểm |
|----------|-------------|
| Module, dễ mở rộng | Cần nhiều model |
| Linh hoạt | Latency cao |
| Dễ debug | Tài nguyên lớn |

## 5. Kết Luận

Trong bài hướng dẫn này, chúng ta đã:

1. **Tích hợp** ba mô hình fine-tuned vào một chatbot
2. **Triển khai** sử dụng Flask
3. **Xử lý** các loại tin nhắn khác nhau
4. **Quản lý** lịch sử cuộc trò chuyện

Chatbot có thể được cải thiện bằng:
- Sử dụng mô hình lớn hơn
- Thêm memory cho long-term context
- Fine-tune cho domain cụ thể

## Tài Liệu Tham Khảo

1. Adiwardana, D., et al. (2020). "Towards a Human-like Open-Domain Chatbot." *Google Research*.

2. Roller, S., et al. (2020). "Recipes for building an open-domain chatbot." *EACL 2021*.

3. Huang, M., et al. (2020). "Sextant: A Conversational AI System for Crisis Support." *ACL 2020*.
<!-- Aero-Footer-Start -->
---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
