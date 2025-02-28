# Bật/tắt chế độ lưu và tải lên
SAVE_ENABLED = True   # Đặt thành True nếu muốn lưu model
UPLOAD_ENABLED = True # Đặt thành True nếu muốn tải lên Hugging Face

# Thay đổi username của Hugging Face tại đây
HF_USERNAME = "cgsharefive"
HF_TOKEN = "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"

# Lưu mô hình ở định dạng 8-bit GGUF
if SAVE_ENABLED:
    model.save_pretrained_gguf("model", tokenizer)

# Tải lên Hugging Face 8-bit GGUF
if UPLOAD_ENABLED:
    model.push_to_hub_gguf(f"{HF_USERNAME}/model", tokenizer, token=HF_TOKEN)

# Lưu mô hình ở định dạng 16-bit GGUF
if SAVE_ENABLED:
    model.save_pretrained_gguf("model", tokenizer, quantization_method="f16")

# Tải lên Hugging Face 16-bit GGUF
if UPLOAD_ENABLED:
    model.push_to_hub_gguf(f"{HF_USERNAME}/model", tokenizer, quantization_method="f16", token=HF_TOKEN)

# Lưu mô hình ở định dạng 4-bit GGUF (Q4_K_M)
if SAVE_ENABLED:
    model.save_pretrained_gguf("model", tokenizer, quantization_method="q4_k_m")

# Tải lên Hugging Face 4-bit GGUF
if UPLOAD_ENABLED:
    model.push_to_hub_gguf(f"{HF_USERNAME}/model", tokenizer, quantization_method="q4_k_m", token=HF_TOKEN)

# Tải lên nhiều phiên bản GGUF cùng lúc
if UPLOAD_ENABLED:
    model.push_to_hub_gguf(
        f"{HF_USERNAME}/model",
        tokenizer,
        quantization_method=["q4_k_m", "q8_0", "q5_k_m"],
        token=HF_TOKEN,
    )
