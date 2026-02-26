# Demo Prompt Engineering Với FLAN-T5

## Giới Thiệu

Chào mừng mọi người đến với demo đầu tiên của khóa học này. Tất cả các demo trong khóa học này sẽ sử dụng Google Colaboratory.

Google Colab là một nền tảng cho phép chúng ta lưu trữ các file notebook và kết nối miễn phí đến một instance trên Google Cloud Platform nơi chúng ta cũng có thể kết nối GPU. Điều này rất hữu ích, đặc biệt cho việc prototype các ý tưởng.

Truy cập: colab.research.google.com

## Thiết Lập Môi Trường

### Kết Nối Google Colab

1. Truy cập trang web Colab
2. Upload notebook từ Exercise Files
3. Click "Connect" để kết nối với GPU miễn phí

**Lưu ý:** Loại GPU phụ thuộc vào:
- Khả năng sẵn có theo múi giờ
- Tần suất sử dụng GPU gần đây
- Vì là miễn phí nên không đảm bảo được loại GPU cụ thể

## Cài Đặt Thư Viện

```python
# Cài đặt Transformers và TensorFlow
!pip install transformers tensorflow
```

## Tải Mô Hình FLAN-T5

```python
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM

# Tải tokenizer và model
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
model = TFAutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
```

**Lưu ý về warnings:**
- Warning về xác thực HuggingFace là bình thường
- Warning về việc model được train bằng PyTorch rồi convert sang TensorFlow - độ chính xác 99.9% tương đương

## Quy Trình Prompt Với FLAN-T5

Việc prompt một LLM luôn gồm 4 bước:
1. Định nghĩa prompt
2. Tokenize (chuyển đổi văn bản thành tokens)
3. Model.generate() (tạo output)
4. Tokenizer.decode() (chuyển đổi IDs về văn bản)

### 1. Tóm Tắt Văn Bản (Summarization)

```python
# Định nghĩa prompt
prompt = "summarize: Studies show that eating carrots help improve vision..."

# Tokenize
inputs = tokenizer(prompt, return_tensors="tf", max_length=512, 
                    truncation=True, padding=True)

# Generate
outputs = model.generate(inputs.input_ids, max_length=50)

# Decode
summary = tokenizer.decode(outputs[0])
print(summary)
```

**Kết quả:** "eat carrots" - một bản tóm tắt ngắn gọn

### 2. Dịch Thuật (Translation)

```python
# Prompt dịch tiếng Anh sang tiếng Tây Ban Nha
prompt = "translate English to Spanish: cheese is delicious"

# Tokenize
inputs = tokenizer(prompt, return_tensors="tf", max_length=512, 
                    truncation=True, padding=True)

# Generate
outputs = model.generate(inputs.input_ids, max_length=40)

# Decode
translation = tokenizer.decode(outputs[0])
print(translation)
```

### 3. Trả Lời Câu Hỏi (Question Answering)

```python
# Context và câu hỏi
context = "The Great Wall of China is over 13,000 miles long."
question = "question: How long is the Great Wall of China?"

prompt = context + " " + question

# Tokenize
inputs = tokenizer(prompt, return_tensors="tf", max_length=512, 
                    truncation=True, padding=True)

# Generate
outputs = model.generate(inputs.input_ids, max_length=50)

# Decode
answer = tokenizer.decode(outputs[0])
print(answer)
```

**Kết quả:** "It's over 13,000 miles long."

## Tổng Kết

Bạn đã thấy:
- Toàn bộ workflow hoạt động với chỉ 4 bước đơn giản
- FLAN-T5 có thể thực hiện nhiều tác vụ: tóm tắt, dịch thuật, trả lời câu hỏi
- Không cần training - chỉ cần prompt là có kết quả

Với kiến thức này, bạn có thể tích hợp LLMs vào bất kỳ chatbot nào mà không cần làm thêm nhiều công việc phức tạp.

## Tài liệu tham khảo

1. **Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017).** *Attention Is All You Need.* Advances in Neural Information Processing Systems, 30, 5998-6008. https://arxiv.org/abs/1706.03762

2. **Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018).** *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* arXiv preprint arXiv:1810.04805. https://arxiv.org/abs/1810.04805

3. **Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., Zhou, Y., Li, W., & Liu, P. J. (2019).** *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer.* arXiv preprint arXiv:1910.10683. https://arxiv.org/abs/1910.10683

4. **Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2021).** *LoRA: Low-Rank Adaptation of Large Language Models.* arXiv preprint arXiv:2106.09685. https://arxiv.org/abs/2106.09685

5. **Chung, H. W., Hou, L., Longpre, S., Zoph, B., Tay, Y., Fedus, W., ... & Wei, J. (2022).** *Scaling Instruction-Finetuned Language Models.* arXiv preprint arXiv:2210.11416. https://arxiv.org/abs/2210.11416

6. **Longpre, S., Hou, L., Vu, T., Webson, A., Chung, H. W., Tay, Y., Zhou, D., Le, Q. V., Zoph, B., Wei, J., & Roberts, A. (2023).** *The Flan Collection: Designing Data and Methods for Effective Instruction Tuning.* arXiv preprint arXiv:2301.13688. https://arxiv.org/abs/2301.13688

7. **Han, Z., Gao, C., Liu, J., Zhang, J., & Zhang, S. Q. (2024).** *Parameter-Efficient Fine-Tuning for Large Models: A Comprehensive Survey.* arXiv preprint arXiv:2403.14608. https://arxiv.org/abs/2403.14608

8. **Weidinger, L., Mellor, J., Rauh, M., Griffin, C., Uesato, J., Huang, P. S., Cheng, M., Glaese, M., Balle, B., Kasirzadeh, A., Kenton, Z., Brown, S., Hawkins, W., Stepleton, T., Biles, C., Birhane, A., Haas, J., Rimell, L., Hendricks, L. A., ... & Gabriel, I. (2021).** *Ethical and Social Risks of Harm from Language Models.* DeepMind. https://storage.googleapis.com/deepmind-media/research/language-research/Ethical%20and%20social%20risks.pdf

9. **Bengio, Y., Mindermann, S., Privitera, D., Besiroglu, T., Bommasani, R., Casper, S., Choi, Y., Goldfarb, D., Heidari, H., Khalatbari, L., Longpre, S., Mavroudis, V., Mazeika, M., Ng, K. Y., Okolo, C. T., Raji, D., Skeadas, T., Tramèr, F., Adekanmbi, B., ... & Zhou, D. (2024).** *International Scientific Report on the Safety of Advanced AI (Interim Report).* arXiv preprint arXiv:2412.05282. https://arxiv.org/abs/2412.05282

10. **Amodei, D., Ananthanarayanan, S., Bapna, R., Chen, Z., Du, E., Goodfellow, I., ... & Sutskever, I. (2016).** *Concrete Problems in AI Safety.* arXiv preprint arXiv:1606.06565. https://arxiv.org/abs/1606.06565

11. **Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2024).** *QLoRA: Efficient Finetuning of Quantized LLMs.* arXiv preprint arXiv:2305.14314. https://arxiv.org/abs/2305.14314

12. **Zhang, Y., Yang, X., Cai, Y., & Giannakis, G. B. (2025).** *ScaLoRA: Optimally Scaled Low-Rank Adaptation for Efficient High-Rank Fine-Tuning.* arXiv preprint arXiv:2510.23818. https://arxiv.org/abs/2510.23818

13. **Wang, R., Dvijotham, K. D., & Manchester, I. R. (2025).** *Norm-Bounded Low-Rank Adaptation.* arXiv preprint arXiv:2501.19050. https://arxiv.org/abs/2501.19050

14. **Liu, H., Tam, D., Muqeeth, M., Mohta, J., Huang, T. A., Bernhard, M., ... & Houlsby, N. (2022).** *LoRA+: Efficient Low Rank Adaptation of Large Models.* arXiv preprint arXiv:2402.12354. https://arxiv.org/abs/2402.12354

15. **Wang, L., Lyu, C., Ji, T., Chen, M., Yu, Z., Shi, A., ... & Yu, P. S. (2023).** *A Survey on Parameter-Efficient Fine-Tuning for Foundation Models.* arXiv preprint arXiv:2504.21099. https://arxiv.org/abs/2504.21099

16. **Laakso, A., Kemell, K. K., & Nurminen, J. K. (2024).** *Ethical Issues in Large Language Models: A Systematic Literature Review.* CEUR Workshop Proceedings, 3901. https://ceur-ws.org/Vol-3901/paper_4.pdf

17. **Bosma, M., & Wei, J. (2021).** *Introducing FLAN: More Generalizable Language Models with Instruction Fine-Tuning.* Google AI Blog. https://research.google/blog/introducing-flan-more-generalizable-language-models-with-instruction-fine-tuning/

18. **Roberts, A., & Raffel, C. (2020).** *Exploring Transfer Learning with T5: the Text-To-Text Transfer Transformer.* Google AI Blog. https://research.google/blog/exploring-transfer-learning-with-t5-the-text-to-text-transfer-transformer/

19. **Lester, B., Al-Rfou, R., & Wang, L. (2021).** *The Power of Scale for Parameter-Efficient Prompt Tuning.* Proceedings of EMNLP 2021. https://arxiv.org/abs/2104.08691

20. **Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P., ... & Amodei, D. (2020).** *Language Models are Few-Shot Learners.* Advances in Neural Information Processing Systems, 33, 1877-1901. https://arxiv.org/abs/2005.14165
