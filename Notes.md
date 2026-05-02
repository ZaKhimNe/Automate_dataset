# ADVERSARIAL RED TEAMING CONVENTIONS (CS DOMAIN - PIPELINE & FRAMEWORK FOCUS)

## 1. 10 CHIẾN THUẬT TẤN CÔNG (ADVERSARIAL TECHNIQUES)
*(Sử dụng 1-3 chiến thuật kết hợp để tạo câu hỏi đánh lừa các mô hình VLM. Mục tiêu tối thượng: Ép AI phải nhìn thật kỹ từng pixel của đường nối, thay vì đoán mò bằng kiến thức CS có sẵn).*

**Nhóm 1: Topological & Flow Reasoning (Đánh lận Cấu trúc Mạng lưới & Luồng)**
1. **Deep Transitive Routing (Truy vết luồng sâu):** Buộc AI phải lần theo một đường đi ngoằn ngoèo qua ít nhất 3-4 node trung gian. *Ví dụ: "Bắt đầu từ node Input, đi theo đường nét đứt, bỏ qua nhánh rẽ trái đầu tiên, node thứ 3 bạn gặp có tên là gì?"*
2. **Recursive/Feedback Loop Unrolling (Gỡ rối Vòng lặp đệ quy):** Tấn công vào điểm yếu không hiểu vòng lặp của AI. *Ví dụ: "Dữ liệu X đi qua Module A. Nếu hàm Loss chưa đạt, nó quay lại theo đường mũi tên ngược màu đỏ. Ở lần chạy thứ 2, dữ liệu sẽ đi qua những khối nào trước khi xuất ra?"*
3. **Counter-factual Severance (Giả lập cắt đứt/Phản thực tế):** Phá vỡ sơ đồ hiện tại bằng một giả định. *Ví dụ: "Nếu đường truyền nối giữa 'Encoder' và 'Decoder' bị đứt, module nào sẽ trở thành điểm nghẽn (bottleneck) không nhận được dữ liệu?"*
4. **Cross-Component Mapping (Ánh xạ chéo đa thể thức):** Bắt AI kết nối một phương trình/ký hiệu toán học (nằm ở góc ảnh hoặc Sách luật) với một khối chức năng không tên (hoặc tên viết tắt) trên sơ đồ.

**Nhóm 2: Semantic vs. Visual Conflict (Xung đột Ngữ nghĩa & Thị giác - Điểm Mù)**
5. **Pre-trained Bias Overriding (Ghi đè định kiến trọng số):** VLM thường mặc định "mũi tên hai chiều" là tương tác qua lại, "màu đỏ" là lỗi, "màu xanh" là thành công. Cố tình thiết lập Sách luật ngược lại. *Ví dụ: "Trong hệ thống này, mũi tên nét liền màu đỏ biểu thị dữ liệu hợp lệ. Khối nào nhận dữ liệu hợp lệ từ Database?"*
6. **Heuristic Hijacking (Gài bẫy lối tắt nhận thức / Ảnh lồng ảnh):** Đặt một bức ảnh con (ví dụ: ảnh mặt người, ảnh chó mèo) ở đuôi một pipeline. VLM sẽ vội kết luận đây là "Hệ thống nhận diện khuôn mặt" thay vì nhìn vào sơ đồ kiến trúc thực tế (ví dụ: đang biểu diễn nén dữ liệu).
7. **Spatial Anchor Chain (Định vị đệ quy không gian):** Ép VLM quét tọa độ thay vì đọc text. *Ví dụ: "Tìm khối nằm góc trên cùng bên phải. Đi thẳng xuống dưới 2 khối, sau đó nhìn sang module bên trái liền kề. Tên của nó là gì?"*
8. **Text-Visual Mismatch Trap (Bẫy mâu thuẫn Chữ - Hình):** Trong ảnh, khối A nối thẳng với khối C (bỏ qua B). Nhưng kiến thức chuẩn của ngành CS (hoặc text chú thích) lại nói A phải qua B rồi mới tới C. Câu hỏi ép AI phải tin vào hình ảnh (A -> C) thay vì tin vào lý thuyết.

**Nhóm 3: Integrity & Negative Space (Tính toàn vẹn & Không gian trống)**
9. **Conditional Phantom Node (Bẫy Thực thể ảo):** Hỏi về sự tương tác của một module/luồng dữ liệu KHÔNG HỀ TỒN TẠI trên sơ đồ. Một AI xuất sắc phải dám mạnh dạn từ chối và chỉ ra sự phi logic. *Ví dụ: "Dữ liệu từ module 'Global Pooling' truyền tới 'SVM Classifier' qua đường nét đứt màu gì?" (Thực tế không hề có SVM Classifier).*
10. **Negative Space Deduction (Suy luận từ khoảng trống):** Suy luận đặc tính của một node dựa trên việc nó KHÔNG có yếu tố nào đó. *Ví dụ: "Trong số 4 nhánh giải thuật, nhánh duy nhất KHÔNG sử dụng cơ chế Attention là nhánh nào?"*

---

## 2. QUY TẮC ĐẶT CÂU HỎI ĐA BƯỚC (MULTIHOP RULES) BẮT BUỘC
*Để phân biệt dự án này với các Benchmark hình ảnh thông thường, Generator Agent KHÔNG ĐƯỢC PHÉP vi phạm các quy tắc sau:*

* **CẤM HỎI NHẬN DIỆN (NO OBJECT DETECTION):** Không đếm số hộp, không đếm số mũi tên, không hỏi màu sắc quần áo của người trong ảnh minh họa.
* **CẤM HỎI SINGLE-HOP:** Không hỏi trực tiếp "Khối A nối với khối nào?".
* **CÔNG THỨC CHUẨN:** [Điểm Neo 1] + [Điều kiện Không gian/Luật] + [Truy vết] +[Điểm Neo 2] = [Đáp án].
* *Ví dụ đúng chuẩn CS-Blindspot:* "Tìm module có màu xanh lá (Neo 1). Đi theo luồng dữ liệu đầu ra duy nhất của nó cho đến khi gặp một khối có ký hiệu phép cộng (Truy vết). Module nằm ngay bên dưới khối phép cộng đó (Neo 2) đóng vai trò gì theo Sách luật?"

---

## 3. QUY TẮC PHÂN LOẠI LỖI (FAILURE REASONS) CHO AUDITOR
*(Auditor Agent dùng để gán nhãn lý do các mô hình VLM trả lời sai)*

1. **Visual_Miss (Lỗi trích xuất thị giác):** Nhìn sót mũi tên, không thấy đường nét đứt, sai lệch về vị trí không gian (trái/phải, trên/dưới).
2. **Pretrained_Overreliance (Lỗi phụ thuộc kiến thức cũ):** Bỏ qua hình ảnh thực tế, tự động lấy kiến thức chuẩn của CS trong não AI để trả lời (bị sập bẫy Text-Visual Mismatch).
3. **Rule_Ignored (Lỗi phớt lờ Sách luật):** Không áp dụng hoặc áp dụng sai các quy tắc phản thực tế được cung cấp trong Sách luật.
4. **Reasoning_Error (Lỗi đứt gãy logic):** Đã nhìn đúng, đã trích xuất đúng các node, nhưng suy luận logic bắc cầu (A->B->C) bị sai ở bước cuối.
5. **Hallucination (Ảo giác cục bộ):** Tự bịa ra tên một module hoặc một đường nối không hề xuất hiện trong ảnh do bị câu hỏi gài bẫy ám thị.
6. **Model_Refusal:** Từ chối trả lời do bộ lọc an toàn hoặc nhận diện sai định dạng.
7. **Survived:** Trả lời đúng hoàn toàn.

---

## 4. BẢO MẬT ĐỊNH DẠNG (JSON SAFETY)
* Mọi lập luận, lý thuyết trong trường `Logic_Steps` PHẢI viết bằng văn bản thường (Plain Text).
* **TUYỆT ĐỐI CẤM** dùng LaTeX (`$$...$$`, `\text{}`, `\frac{}`) hoặc Markdown code blocks.
* Dùng dấu nháy đơn (`'`) thay cho dấu nháy kép (`"`) bên trong chuỗi text để bảo vệ cấu trúc JSON gốc.