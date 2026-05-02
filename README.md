# 🎯 CS-Blindspot: Adversarial Multi-hop QA Dataset for Vision-Language Models

**CS-Blindspot** là một hệ thống Red Teaming Pipeline tự động, được thiết kế để đánh giá điểm mù của các mô hình Vision-Language (VLM) hàng đầu hiện nay (như GPT-4o, Claude 3.5, Gemini 1.5 Pro). Hệ thống tập trung vào việc tạo ra các bẫy logic đa bước (multi-hop reasoning) trên các sơ đồ Khoa học Máy tính phức tạp.


---

## 🌟 Features

*   **End-to-End Pipeline:** Tự động hóa hoàn toàn từ khâu đọc PDF, bóc tách ảnh, sinh câu hỏi bẫy, cho AI làm bài, đến khâu chấm điểm tự động (LLM-as-a-Judge).
*   **Adversarial Generation:** Sử dụng AI để gài bẫy AI bằng các chiến thuật như: Xung đột Ngữ nghĩa - Thị giác, Ảo giác Không gian, và Cắt đứt luồng logic.
*   **Zero-Tolerance Curation:** Tự động phát hiện và loại bỏ các câu hỏi mập mờ, lỗi API, hoặc vi phạm logic để bảo vệ tính toàn vẹn của dataset.
*   **Incremental Auditing:** Cơ chế chấm điểm thông minh, chỉ chấm bù những câu còn thiếu, giúp tiết kiệm tối đa chi phí API.
*   **Analytical Ready:** Xuất dữ liệu cuối cùng ra định dạng CSV chuẩn mực, chia làm 4 cụm rõ ràng, đi kèm công cụ thống kê (Metrics) và vẽ biểu đồ tự động.

---

## ⚙️ Cấu trúc hệ thống (The 4-Phase Pipeline)

Hệ thống được chia thành 4 trạm độc lập, giao tiếp qua định dạng JSON để chống mất mát dữ liệu:

1.  **Trạm 1 (Curation):** Lọc nhiễu ảnh tự động, chỉ giữ lại các sơ đồ kiến trúc, lưu đồ, đồ thị từ các file PDF khoa học.
2.  **Trạm 2 (Generator):** Đóng vai Kẻ Tấn Công. Viết "Sách luật" phản thực tế và sinh câu hỏi suy luận đa bước. Tích hợp Thẩm phán (Gatekeeper) để loại bỏ câu hỏi chất lượng thấp.
3.  **Trạm 3 (Victim Inference):** Đưa đề thi cho các mô hình (GPT, Gemini, Claude). Ép chúng xuất luồng tư duy (CoT) trước khi chốt đáp án.
4.  **Trạm 4 (Auditor):** Đóng vai Giám Khảo. Đọc CoT của nạn nhân, so sánh với barem, chấm điểm và gán mã lỗi chi tiết (Visual Miss, Reasoning Error...).

---

## 🛠️ Yêu cầu môi trường & Cài đặt (Installation)

**1. Clone kho lưu trữ:**
```bash
git clone https://github.com/ZaKhimNe/Automate_dataset.git
cd Automate_dataset
```

**2. Cài đặt các thư viện cần thiết:**
Đảm bảo bạn đang sử dụng Python 3.9+. Nên sử dụng môi trường ảo (virtualenv/conda).
```bash
pip install -r requirements.txt
```

**3. Cấu hình API Keys:**
*   Hệ thống sử dụng LiteLLM để gọi đa dạng các mô hình.
*   Bạn cần tạo một file `.env` ở thư mục gốc (ngang hàng `README.md`) để cấu hình API Key. (Xem file `.env.example` nếu có).
*   Nếu sử dụng Google Vertex AI, đặt file JSON chứa thông tin xác thực vào đường dẫn: `credentials/cs-blindspot-agent.json`.
*   Cấu hình tên các mô hình bạn muốn sử dụng trong file `credentials/config.py`.

---

## 🚀 Hướng dẫn sử dụng (Workflow)

Để tạo ra một mẻ dataset mới, hãy chạy lần lượt các script sau từ thư mục gốc của dự án:

**Bước 1: Chuẩn bị dữ liệu thô**
*   Copy các bài báo định dạng PDF vào thư mục `data/pdfs/`. (Lưu ý: Tên file PDF nên là ID của bài báo để hệ thống tự động tạo source link).
*   Nếu có danh sách link nguồn, đặt vào file `src/source_links.json`.
*   Chạy lệnh trích xuất:
```bash
python src/crawldata.py
```

**Bước 2: Chạy Pipeline 4 Trạm**
Chạy lần lượt các lệnh sau:
```bash
# Trạm 1: Lọc ảnh rác
python src/01_data_curation.py

# Trạm 2: Sinh câu hỏi bẫy đối kháng
python src/02_generator_agent.py

# (Tùy chọn) Double check lại câu hỏi
python utils/verify_questions.py

# Trạm 3: Cho các mô hình làm bài kiểm tra 
python src/03_victim_inference.py

# Trạm 4: Giám khảo chấm điểm
python src/04_auditor_agent.py
```

**Bước 3: Dọn dẹp & Phân tích (Utils)**
Sau khi Trạm 4 chạy xong, hãy dùng các công cụ sau để đảm bảo chất lượng và trích xuất báo cáo:
```bash
# 1. Cứu hộ lỗi mạng của các Nạn nhân (Victim AI)
python utils/repair_api_errors.py
# (Nếu tool này báo có lỗi, hãy chạy LẠI Trạm 3 để các AI thi lại phần bị rớt, 
# sau đó chạy LẠI Trạm 4 để chấm điểm cho các phần thi bổ sung đó).

# 2. Cứu hộ lỗi mạng của Giám khảo (Auditor AI)
python utils/repair_audit_errors.py
# (Nếu tool này báo có lỗi được reset, hãy chạy LẠI Trạm 4 để Giám khảo chấm vét).

# 3. Thống kê kết quả & Độ khó (Reporting)
python utils/count_dataset.py

# 4. Vẽ biểu đồ phân tích lỗi (Radar/Heatmap) chuẩn báo cáo
pyhon utils/visualize_errors.py
```
---

## 📊 Phân loại lỗi (Taxonomy of Failures)

Khi một VLM trả lời sai, hệ thống không chỉ ghi nhận False mà còn phân tích bản chất cái sai:

*   **Visual_Miss:** Mù màu, nhìn sót mũi tên, sai vị trí không gian.
*   **Pretrained_Overreliance:** Bỏ qua hình ảnh, dùng kiến thức học vẹt để đoán mò.
*   **Rule_Ignored:** Cãi luật, không áp dụng các quy tắc phản thực tế trong Sách luật.
*   **Reasoning_Error:** Nhìn đúng ảnh nhưng suy luận logic bắc cầu (A -> B -> C) bị sai.
*   **Hallucination:** Bịa ra module hoặc đường nối không hề tồn tại trên ảnh.

---

## 📁 Cấu trúc thư mục (Directory Structure)

```text
AI_REDTEAMING_PIPELINE/
├── credentials/          # Chứa cấu hình model và API Keys (Bỏ qua trên Git)
│   ├── config.py
│   └── cs-blindspot-agent.json
├── data/                 # Chứa toàn bộ dữ liệu (Bỏ qua trên Git)
│   ├── images/           # Ảnh sạch đã lọc
│   ├── pdfs/             # File PDF gốc
│   ├── pipeline_results/ # Kết quả trung gian (JSON)
│   └── raw_data/         # Ảnh thô bóc từ PDF
├── src/                  # Mã nguồn chính của 4 Trạm
├── utils/                # Các công cụ hỗ trợ dọn dẹp, thống kê
├── Dataset.csv           # File kết quả cuối cùng (Sạch & Chuẩn hóa)
├── README.md
└── requirements.txt
```
