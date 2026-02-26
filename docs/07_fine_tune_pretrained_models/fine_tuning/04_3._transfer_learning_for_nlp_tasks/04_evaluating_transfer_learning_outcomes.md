
<!-- Aero-Navigation-Start -->
[🏠 Home](../../../index.md) > [07 fine tune pretrained models](../../index.md) > [fine tuning](../index.md) > [04 3. transfer learning for nlp tasks](index.md)

---
### 🧭 Điều hướng nhanh

- [🏠 Cổng tài liệu](../../../index.md)
- [📚 Module 01: LLM Course](../../../01_llm_course/index.md)
- [🔢 Module 02: Tokenization](../../../02_words_to_tokens_to_numbers/index.md)
- [🏗️ Module 04: Build GPT](../../../04_buildgpt/index.md)
- [🎯 Module 07: Fine-tuning](../../../07_fine_tune_pretrained_models/index.md)
- [🔍 Module 19: AI Safety](../../../19_ai_safety/index.md)
- [🐍 Module 20: Python for AI](../../../20_python_colab_notebooks/index.md)
---
<!-- Aero-Navigation-End -->
# Đánh Giá Kết Quả Transfer Learning

## Giới Thiệu

Bây giờ chúng ta đã thấy transfer learning hoạt động như thế nào, hãy tập trung vào việc đánh giá kết quả của transfer learning, cụ thể là sử dụng các metrics như ROUGE và BLEU. Các metrics này rất cần thiết để đánh giá chất lượng của các tác vụ như tóm tắt văn bản và dịch máy, tương tự như cách độ chính xác được sử dụng trong các tác vụ phân loại.

## ROUGE Là Gì?

ROUGE viết tắt của Recall-Oriented Understudy for Gisting Evaluation. Nó chủ yếu được sử dụng để đánh giá chất lượng của các bản tóm tắt bằng cách đo lường sự chồng chéo giữa bản tóm tắt được tạo và một tập hợp các bản tóm tắt tham chiếu.

Hãy nghĩ về ROUGE như một thước đo về sự chồng chéo giữa văn bản được tạo và văn bản tham chiếu, giống như so sánh hai công thức có bao nhiêu nguyên liệu giống nhau.

### Tính ROUGE-1

**Ví dụ:**
- Bản tóm tắt được tạo: "The cat sat on the mat."
- Bản tóm tắt tham chiếu: "The cat is sitting on the mat."

Để tính ROUGE-1 (unigram overlap), chúng ta đếm số từ chồng chéo và chia cho tổng số từ trong bản tóm tắt tham chiếu.

Các từ chồng chéo: "the", "cat", "on", "the", "mat" (5 từ)
Tổng số từ trong tham chiếu: 6 từ
**ROUGE-1 = 5/6 = 83%**

## BLEU Là Gì?

BLEU viết tắt của Bilingual Evaluation Understudy. Nó đo lường mức độ văn bản được tạo khớp với văn bản tham chiếu bằng cách so sánh các n-grams, có nghĩa là các chuỗi của n từ.

BLEU giống như một nhà phê bình thực phẩm tỉ mỉ, người không chỉ kiểm tra các nguyên liệu có đúng không, mà còn xem chúng có được kết hợp đúng thứ tự và tỷ lệ hay không.

### Tính BLEU-1

Sử dụng cùng ví dụ:
- Bản tạo: "The cat sat on the mat"
- Tham chiếu: "The cat is sitting on the mat"

BLEU-1 = 5/5 = 100%

## So Sánh ROUGE Và BLEU

- **ROUGE:** Tập trung vào recall - nắm bắt bao nhiêu của bản tóm tắt tham chiếu có mặt trong bản tóm tắt được tạo
- **BLEU:** Tập trung vào precision - đánh giá mức độ văn bản được tạo khớp với tham chiếu về các cụm từ chính xác và thứ tự của chúng

Bạn có thể nghĩ về BLEU như một phép đo precision và ROUGE như một phép đo recall.

ụng Tr## Sử Dong Code

```python
# Đánh giá với ROUGE
from datasets import load_metric
rouge = load_metric("rouge")
results = rouge.compute(predictions=predictions, references=references)

# Đánh giá với BLEU
bleu = load_metric("bleu")
results = bleu.compute(predictions=predictions, references=references)

## Kết Luận

Hiểu và sử dụng các metrics ROUGE và BLEU là điều cần thiết để đánh giá hiệu quả các tác vụ tạo văn bản. Bằng cách so sánh các metrics này với độ chính xác và phân loại, chúng ta có thể đánh giá cao hơn vai trò của chúng trong việc đánh giá chất lượng văn bản được tạo bởi AI.

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
<!-- Aero-Footer-Start -->

## 📄 Tài liệu cùng chuyên mục
| Bài học | Liên kết |
| :--- | :--- |
| [Transfer Learning Trong LLMs](01_transfer_learning_in_llms.md) | [Xem bài viết →](01_transfer_learning_in_llms.md) |
| [Chọn Mô Hình Cho Transfer Learning](02_choosing_models_for_transfer_learning.md) | [Xem bài viết →](02_choosing_models_for_transfer_learning.md) |
| [Demo Transfer Learning với FLAN-T5](03_demo_transfer_learning_with_flan_t5.md) | [Xem bài viết →](03_demo_transfer_learning_with_flan_t5.md) |
| 📌 **[Đánh Giá Kết Quả Transfer Learning](04_evaluating_transfer_learning_outcomes.md)** | [Xem bài viết →](04_evaluating_transfer_learning_outcomes.md) |
| [Demo Đánh Giá Bản Dịch](05_demo_evaluating_translations.md) | [Xem bài viết →](05_demo_evaluating_translations.md) |
| [Giải Pháp Nâng Cao Dịch Thuật với Transfer Learning](06_solution_enhancing_translation_with_transfer_learning.md) | [Xem bài viết →](06_solution_enhancing_translation_with_transfer_learning.md) |

---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
