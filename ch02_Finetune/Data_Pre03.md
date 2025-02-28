## **1. Tạo bộ dữ liệu từ sách (PDF, EPUB, DOCX)**
### **Phương pháp 1: Trích xuất văn bản từ sách số hóa**
Nếu sách ở định dạng **PDF, EPUB, hoặc DOCX**, bạn có thể dùng thư viện để trích xuất văn bản.

#### **Trích xuất văn bản từ PDF** (sử dụng `pdfplumber`)
```python
import pdfplumber

def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

text = extract_text_from_pdf("sample.pdf")
print(text[:500])  # Xem trước 500 ký tự đầu tiên
```

#### **Trích xuất văn bản từ EPUB** (sử dụng `ebooklib`)
```python
from ebooklib import epub
from bs4 import BeautifulSoup

def extract_text_from_epub(epub_path):
    book = epub.read_epub(epub_path)
    text = ""
    for item in book.items:
        if item.get_type() == 9:  # Type 9 là XHTML content
            soup = BeautifulSoup(item.content, 'html.parser')
            text += soup.get_text() + "\n"
    return text

text = extract_text_from_epub("sample.epub")
print(text[:500])
```

### **Phương pháp 2: OCR nếu sách ở dạng ảnh (Scan)**
Nếu sách là ảnh hoặc scan dạng PDF, bạn có thể dùng **OCR (Nhận diện ký tự quang học)**.

```python
import pytesseract
from PIL import Image

# Đọc ảnh và trích xuất văn bản
image = Image.open("page1.png")
text = pytesseract.image_to_string(image, lang="eng")
print(text)
```
🚀 **Ứng dụng:** Chuyển sách giấy thành văn bản số để làm dataset!

---

## **2. Tạo bộ dữ liệu từ video**
Bạn có thể **trích xuất phụ đề hoặc tạo transcript từ video**.

### **Phương pháp 1: Trích xuất phụ đề từ YouTube (nếu có)**
```python
from youtube_transcript_api import YouTubeTranscriptApi

video_id = "dQw4w9WgXcQ"  # Thay bằng ID video của bạn
transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])

# Ghép nội dung thành văn bản
text = "\n".join([t['text'] for t in transcript])
print(text[:500])
```

### **Phương pháp 2: Tạo transcript từ video bằng Whisper**
Nếu video không có phụ đề, bạn có thể dùng mô hình **Whisper** của OpenAI để nhận diện giọng nói.

```python
import whisper

model = whisper.load_model("small")  # Chọn mô hình từ 'tiny' đến 'large'
result = model.transcribe("video.mp4")
print(result["text"])
```
🚀 **Ứng dụng:** Chuyển nội dung video thành dataset để huấn luyện mô hình!

---

## **3. Tạo bộ dữ liệu từ âm thanh**
Nếu bạn có **file ghi âm hoặc podcast**, bạn có thể sử dụng mô hình **ASR (Automatic Speech Recognition)** để chuyển thành văn bản.

### **Phương pháp 1: Dùng Whisper**
```python
import whisper

model = whisper.load_model("small")
result = model.transcribe("audio.mp3")
print(result["text"])
```

### **Phương pháp 2: Dùng Google Speech-to-Text API**
```python
import speech_recognition as sr

recognizer = sr.Recognizer()
audio_file = "audio.wav"

with sr.AudioFile(audio_file) as source:
    audio = recognizer.record(source)

text = recognizer.recognize_google(audio, language="vi-VN")  # Nhận diện tiếng Việt
print(text)
```
🚀 **Ứng dụng:** Chuyển podcast, bài giảng thành dữ liệu huấn luyện!

---

## **4. Tiền xử lý dữ liệu để tạo dataset**
Sau khi thu thập văn bản từ sách, video, hoặc âm thanh, bạn cần:
- **Loại bỏ dấu câu, ký tự không cần thiết**
- **Chia nhỏ thành các câu hỏi - câu trả lời**
- **Gán nhãn dữ liệu (category, difficulty, source, etc.)**

Ví dụ tạo dataset:
```python
import json

dataset = [
    {
        "question": "Vận tốc ánh sáng trong chân không là bao nhiêu?",
        "answer": "Vận tốc ánh sáng trong chân không là 299,792,458 m/s.",
        "source": "Sách vật lý lớp 12",
        "category": "physics",
        "difficulty": "easy"
    }
]

with open("dataset.json", "w", encoding="utf-8") as f:
    json.dump(dataset, f, ensure_ascii=False, indent=4)
```

---

## **5. Đưa bộ dữ liệu lên Hugging Face**
Sau khi có dataset, bạn có thể tải lên Hugging Face:
```python
from huggingface_hub import HfApi

api = HfApi()
api.upload_file(
    path_or_fileobj="dataset.json",
    path_in_repo="dataset.json",
    repo_id="your-username/physics-dataset",
    repo_type="dataset"
)
```
🚀 **Ứng dụng:** Chia sẻ bộ dữ liệu với cộng đồng AI!

---

### **Bạn muốn tôi giúp code phần nào không?** 🚀
