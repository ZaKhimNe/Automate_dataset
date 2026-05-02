import base64
import os
import time
import json
import sys
import litellm
from litellm import completion
from tenacity import retry, wait_random_exponential, stop_after_attempt
from pydantic import BaseModel, Field, ValidationError, field_validator

litellm.drop_params = True
litellm._turn_on_debug()

current_dir = os.path.dirname(os.path.abspath(__file__)) 
parent_dir = os.path.dirname(current_dir)
credentials_dir = os.path.join(parent_dir, "credentials") 
sys.path.append(credentials_dir)
import config

DATA_DIR = os.path.join(parent_dir, "data")
IMAGES_DIR = os.path.join(DATA_DIR, "images")
PIPELINE_RESULTS_DIR = os.path.join(DATA_DIR, "pipeline_results")

CLEAN_JSON_PATH = os.path.join(PIPELINE_RESULTS_DIR, "01_clean_data.json")
ADV_QUESTIONS_PATH = os.path.join(PIPELINE_RESULTS_DIR, "02_adversarial_questions.json")

NOTES_MD_PATH = os.path.join(parent_dir, "Notes.md")
BLINDSPOT_AGENT_PATH = os.path.join(credentials_dir, "cs-blindspot-agent.json")

class AdversarialData(BaseModel):
    Attack_Type_1: str = Field(..., min_length=1)
    Attack_Type_2: str = ""
    Attack_Type_3: str = ""
    Adversarial_Prompt: str = Field(..., min_length=10)
    Victim_Illusion: str = Field(..., min_length=20) 
    Logic_Steps: str = Field(..., min_length=50)
    Ground_Truth_Answer: str = Field(..., min_length=1) # ĐÃ SỬA TÊN

    @field_validator('Logic_Steps')
    def check_logic_length(cls, v):
        if len(v) < 100:
            raise ValueError("LỖI CẤU TRÚC: Logic_Steps quá ngắn (<100 ký tự). AI đang lười biếng!")
        return v

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def read_notes():
    notes_content = ""
    if os.path.exists(NOTES_MD_PATH):
        with open(NOTES_MD_PATH, "r", encoding="utf-8") as f:
            notes_content += f.read() + "\n\n"
    if os.path.exists(BLINDSPOT_AGENT_PATH):
        try:
            with open(BLINDSPOT_AGENT_PATH, "r", encoding="utf-8") as f:
                tactics = json.load(f)
                notes_content += "[CÁC CHIẾN THUẬT TẤN CÔNG (Từ cs-blindspot-agent.json)]:\n"
                notes_content += json.dumps(tactics, ensure_ascii=False, indent=2)
        except json.JSONDecodeError:
            pass
    return notes_content

@retry(wait=wait_random_exponential(min=config.RETRY_MIN_WAIT, max=config.RETRY_MAX_WAIT), stop=stop_after_attempt(3))
def assess_complexity(image_base64):
    print("   📏 [Assessor] Đang đo lường độ phức tạp của bức ảnh...")
    assessor_schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "complexity_assessment",
            "schema": {
                "type": "object",
                "properties": {
                    "Reasoning": {"type": "string"},
                    "Complexity_Score": {"type": "integer"},
                    "Target_Questions": {"type": "integer"}
                },
                "required":["Reasoning", "Complexity_Score", "Target_Questions"]
            }
        }
    }
    system_prompt = """
    Bạn là Chuyên gia Đánh giá Hình ảnh Khoa học. Hãy quan sát bức ảnh và đánh giá độ phức tạp của nó theo thang 1 đến 4.
    
    - Mức 1 (Đơn giản): Dưới 5 khối, luồng tuyến tính. -> Target_Questions: 2
    - Mức 2 (Trung bình): 5-10 khối, có nhánh rẽ, màu sắc/nét đứt. -> Target_Questions: 3
    - Mức 3 (Phức tạp): 10-20 khối, mũi tên đan chéo, lồng ghép (ảnh trong ảnh). -> Target_Questions: 4
    - Mức 4 (Rất phức tạp): Mạng nhện, chi tiết cực kỳ dày đặc. -> Target_Questions: 5
    
    Hãy viết vài dòng phân tích (Reasoning) rồi đưa ra Complexity_Score và Target_Questions tương ứng.
    """
    try:
        response = completion(
            model=config.VICTIM_AI_MODEL, 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content":[{"type": "text", "text": "Hãy đánh giá bức ảnh này."}, {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}]}
            ],
            response_format=assessor_schema
        )
        clean_json = response.choices[0].message.content.replace("```json", "").replace("```", "").strip()
        return json.loads(clean_json).get("Target_Questions", 2) 
    except Exception:
        return 2

@retry(wait=wait_random_exponential(min=config.RETRY_MIN_WAIT, max=config.RETRY_MAX_WAIT), stop=stop_after_attempt(3))
def generate_enhanced_context(image_base64, raw_caption):
    print("   🧠[Context Enhancer] Đang thiết lập Rulebook...")
    system_prompt = f"""
    Bạn là Kỹ sư Kiến tạo Ngữ cảnh (Context Enhancer) cho dự án Red Teaming đa tác nhân. 
    Nhiệm vụ của bạn là dựa vào [Bức ảnh] và [Chú thích gốc] dưới đây để tạo ra một "Sách luật" (Enhanced Context) cố định.

    [CHÚ THÍCH GỐC CỦA BÀI BÁO]:
    "{raw_caption}"

    [YÊU CẦU BẮT BUỘC ĐỂ TẠO BẪY ĐA BƯỚC - MULTIHOP]:
    1. Giữ nguyên tính chuyên ngành và ý nghĩa cốt lõi của sơ đồ.
    2. Thông tin nhiễu (Red Herrings): Bơm thêm mô tả chi tiết, dài dòng, trịnh trọng về những thành phần vật lý KHÔNG QUAN TRỌNG để đánh lạc hướng Thí sinh.
    3. Quy luật mâu thuẫn (Contradictory Rules): Đưa ra một "quy luật chung" trong Text, nhưng đảm bảo trên ảnh CÓ CHỨA MỘT NGOẠI LỆ đi ngược lại quy luật đó.
    4. Định hướng sai bằng ngữ nghĩa (Semantic Misdirection): Dùng từ ngữ mang tính ám thị, xúi giục mạnh để ép Thí sinh bị "ảo giác" tin vào đoạn text mà lười nhìn kỹ ảnh.

    [QUY TẮC ĐẦU RA - NGHIÊM NGẶT]:
    - Trả về ĐÚNG MỘT ĐOẠN VĂN BẢN (khoảng 4-6 câu).
    - TUYỆT ĐỐI KHÔNG mở bài hay kết bài.
    - KHÔNG trả về định dạng JSON hay Markdown.
    """
    try:
        response = completion(
            model=config.SMART_AI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content":[{"type": "text", "text": "Tạo Rulebook."}, {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}]}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return raw_caption 

@retry(wait=wait_random_exponential(min=config.RETRY_MIN_WAIT, max=config.RETRY_MAX_WAIT), stop=stop_after_attempt(config.MAX_RETRIES))
def generate_adversarial_data(image_base64, notes_content, static_enhanced_context, previous_context=""):
    response_schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "adversarial_prompt_schema",
            "schema": {
                "type": "object",
                "properties": {
                    "Attack_Type_1": {"type": "string"},
                    "Attack_Type_2": {"type": "string"},
                    "Attack_Type_3": {"type": "string"},
                    "Adversarial_Prompt": {"type": "string"},
                    "Victim_Illusion": {"type": "string"}, 
                    "Logic_Steps": {"type": "string"},
                    "Ground_Truth_Answer": {"type": "string"}
                },
                "required":["Attack_Type_1", "Adversarial_Prompt", "Victim_Illusion", "Logic_Steps", "Ground_Truth_Answer"]
            }
        }
    }

    history_prompt = f"\n[LỊCH SỬ THẤT BẠI - CẤM TRÙNG LẶP]:\n{previous_context}" if previous_context else ""

    system_prompt = f"""
    Bạn là chuyên gia AI Red Teaming cấp cao cho dự án CS-Blindspot. Nhiệm vụ: Nhìn vào một Sơ đồ Khoa học Máy tính (Kiến trúc/Lưu đồ) và tạo ra một CÂU HỎI BẪY ĐỐI KHÁNG (Adversarial Prompt) để hạ gục các mô hình VLM (như GPT-4o, Gemini).

    [SÁCH LUẬT HỆ THỐNG - BẮT BUỘC TUÂN THỦ]:
    "{static_enhanced_context}"

    [TÀI LIỆU CHIẾN THUẬT]:
    {notes_content}
    {history_prompt}

    [YÊU CẦU ĐỘ KHÓ]:
    - Câu hỏi BẮT BUỘC giải được 100% bằng cách quan sát vật lý (màu sắc, vị trí, nét đứt/liền) trên ảnh kết hợp Sách luật. KHÔNG dùng kiến thức ngành.
    - Victim_Illusion: Dự đoán Thí sinh sẽ bị ảo giác/nhìn nhầm chỗ nào.
    - Logic_Steps: Lập luận đập tan ảo giác đó bằng bằng chứng trên ảnh. Từng bước rõ ràng.[QUY TẮC SINH BẪY CỐT LÕI - KHÔNG ĐƯỢC VI PHẠM]:
    1. BẮT BUỘC ĐA BƯỚC (Multihop): Câu hỏi phải ép Victim lần theo luồng dữ liệu ít nhất 3 trạm (Ví dụ: Từ Module A -> qua Mũi tên X -> đến Module B -> đối chiếu Sách luật -> Kết luận C).
    2. TƯ DUY KIẾN TRÚC, KHÔNG PHẢI ĐẾM SỐ: Cấm các câu hỏi dạng đếm đối tượng (có bao nhiêu hộp đỏ?) hay OCR thuần túy (đọc text trong hộp). Phải kiểm tra sự thấu hiểu về "Luồng hoạt động" (Flow/Topology).
    3. VỎ BỌC CHUYÊN NGÀNH, LỜI GIẢI VẬT LÝ: Sử dụng thuật ngữ CS trong sơ đồ để làm câu hỏi có vẻ uyên bác, nhưng ĐÁP ÁN phải được giải quyết 100% bằng cách tinh mắt lần theo đường nét, vị trí, chiều mũi tên trên sơ đồ kết hợp với Sách luật. Điểm mù nằm ở chỗ AI thường dùng kiến thức CS có sẵn trong trọng số (weights) để đoán mò thay vì thực sự nhìn vào hình ảnh.

    [HƯỚNG DẪN TRƯỜNG DỮ LIỆU]:
    - Victim_Illusion: Dự đoán chính xác "Lối tắt nhận thức" (Shortcut/Heuristic) nào sẽ khiến Victim trả lời sai. (Ví dụ: "Victim sẽ thấy từ khóa 'Database' liền vội vàng kết luận luồng đi sang phải theo tư duy thông thường, bỏ qua mũi tên nét đứt hướng xuống").
    - Logic_Steps: Viết từng bước (1, 2, 3...) bẻ gãy ảo giác trên bằng bằng chứng vật lý rành rành trên ảnh.

    [QUY TẮC ĐỊNH DẠNG]: Trả về JSON đúng Schema cung cấp. Không dùng LaTeX, không dùng nháy kép bên trong chuỗi.
    """

    try:
        response = completion(
            model=config.SMART_AI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content":[{"type": "text", "text": "Tạo câu hỏi bẫy."}, {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}]}
            ],
            response_format=response_schema,
            safety_settings=[ 
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
            ]
        )
        clean_json = response.choices[0].message.content.replace("```json", "").replace("```", "").strip()
        clean_json = clean_json.replace("\\", "\\\\")
        return AdversarialData.model_validate_json(clean_json)
    except Exception as e:
        print(f"❌ Lỗi kết nối / Parse JSON: {e}")
        return None

@retry(wait=wait_random_exponential(min=config.RETRY_MIN_WAIT, max=config.RETRY_MAX_WAIT), stop=stop_after_attempt(3))
def semantic_gatekeeper(image_base64, adv_prompt, logic_steps, ground_truth, static_enhanced_context):
    print("   🛡️ [Gatekeeper] Đang rà soát...")
    
    gatekeeper_schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "gatekeeper_review",
            "schema": {
                "type": "object",
                "properties": {
                    "Reasoning": {"type": "string"},
                    "Decision": {"type": "string", "enum":["YES", "NO"]}
                },
                "required": ["Reasoning", "Decision"]
            }
        }
    }

    system_prompt = f"""
    Bạn là Thẩm phán Gác cổng (Gatekeeper) VÔ CÙNG KHẮT KHE cho dự án CS-Blindspot. Nhiệm vụ của bạn là TÌM MỌI CỚ ĐỂ LOẠI BỎ (REJECT) câu hỏi bẫy này.[SÁCH LUẬT HỆ THỐNG]:
    "{static_enhanced_context}"[DỮ LIỆU CẦN THẨM ĐỊNH]:
    - Câu hỏi: {adv_prompt}
    - Logic giải: {logic_steps}
    - Đáp án: {ground_truth}

    Hãy soi xét Dữ liệu thẩm định đối chiếu với Hình ảnh. Đánh giá "Decision": "NO" ngay lập tức nếu vi phạm MỘT TRONG 5 LỖI TỬ HUYỆT sau:

    1. LỖI ẢO GIÁC THỊ GIÁC (Visual Hallucination): Logic_Steps hoặc Câu hỏi viện dẫn một chi tiết (module, màu sắc, mũi tên nối) KHÔNG HỀ TỒN TẠI hoặc BỊ CẮT LẸM, CHE KHUẤT trên bức ảnh thực tế.
    2. LỖI ĐƠN BƯỚC / LỆCH PHA (Single-Hop / Misalignment): 
    - Thí sinh KHÔNG CẦN nhìn ảnh, chỉ đọc Sách luật là ra đáp án.
    - HOẶC Thí sinh chỉ cần nhìn ảnh mà không cần đối chiếu với Sách luật.
    - HOẶC Chỉ cần nhìn 1 module duy nhất là ra kết quả (Không có tính chất bắc cầu A -> B -> C).
    3. LỖI NHẬN DIỆN VỤN VẶT (Object Detection Trap): Câu hỏi mang tính chất đếm số lượng (đếm mũi tên, đếm hộp), hỏi màu sắc đơn thuần, hoặc chơi chữ/đếm ký tự thay vì kiểm tra tư duy "Luồng hệ thống" (System Flow) của Khoa học máy tính.
    4. LỖI CHỦ QUAN (Subjectivity): Câu hỏi dùng từ ngữ không thể xác định khách quan bằng mắt (ví dụ: "quan trọng nhất", "nổi bật nhất", "bước chính").
    5. THIẾU TÍNH ĐỐI KHÁNG (Lack of Adversarial Nature): Câu hỏi quá thẳng thắn, thiếu yếu tố "Gài bẫy" (Không đánh lừa hướng nhìn, không có luật phản thực tế, không có nhiễu).[QUY TẮC ĐẦU RA BẮT BUỘC]:
    Chỉ trả về MỘT chuỗi JSON hợp lệ với định dạng sau, tuyệt đối không có markdown ```json:
    {
    "Reasoning": "<Phân tích sâu sắc của bạn theo từng tiêu chí, tìm xem có lỗ hổng nào không>",
    "Violated_Rule": "<Số thứ tự của Lỗi vi phạm từ 1 đến 5, hoặc 'None' nếu hoàn hảo>",
    "Decision": "<Chỉ xuất 'YES' nếu Violated_Rule là None, ngược lại bắt buộc là 'NO'>"
    }

    """
    user_text = f"Câu hỏi: {adv_prompt}\nLogic: {logic_steps}\nĐáp án: {ground_truth}"
    
    try:
        response = completion(
            model=config.SMART_AI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content":[{"type": "text", "text": user_text}, {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}]}
            ],
            response_format=gatekeeper_schema,
            safety_settings=[
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
            ]
        )
        clean_json = response.choices[0].message.content.replace("```json", "").replace("```", "").strip()
        result = json.loads(clean_json)
        return result.get("Decision") == "YES"
    except Exception as e:
        print(f"Lỗi Gatekeeper: {e}")
        return False

def run_generator_pipeline():
    print("🚀 BƯỚC 2: KHỞI ĐỘNG XƯỞNG TẠO BẪY (GENERATOR AGENT)...")
    
    if not os.path.exists(CLEAN_JSON_PATH):
        print(f"⏭️ LỖI: Không tìm thấy {CLEAN_JSON_PATH}!")
        return

    with open(CLEAN_JSON_PATH, "r", encoding="utf-8") as f:
        clean_data = json.load(f)

    adv_data_results =[]
    if os.path.exists(ADV_QUESTIONS_PATH):
        try:
            with open(ADV_QUESTIONS_PATH, "r", encoding="utf-8") as f:
                adv_data_results = json.load(f)
        except json.JSONDecodeError:
            pass
            
    processed_images = {item.get("Image_Filename") for item in adv_data_results}
    notes = read_notes()

    for idx, record in enumerate(clean_data):
        img_name = record["Image_Filename"]
        if img_name in processed_images:
            continue
            
        print(f"\n[{'='*50}]\n[Ảnh {idx+1}/{len(clean_data)}] Đang xử lý: {img_name}")
        img_path = os.path.join(IMAGES_DIR, img_name)
        if not os.path.exists(img_path):
            continue

        img_base64 = encode_image(img_path)
        
        # Bắt đầu xây dựng Record chuẩn của Cụm 1 & 2
        img_record = {
            "Image_Filename": img_name,
            "Source_Link": record.get("Source_Link", "N/A"),
            "Image_Type": record.get("Image_Type", "Diagram"),
            "Visual_Context": "", 
            "Traps":[]
        }

        TARGET_PER_IMAGE = assess_complexity(img_base64)
        print(f"   🎯 Cần tạo: {TARGET_PER_IMAGE} câu hỏi.")
        MAX_ATTEMPTS = TARGET_PER_IMAGE * 2 + 2
        
        # Tạo Sách luật
        static_enhanced_context = generate_enhanced_context(img_base64, record.get("Caption", ""))
        img_record["Visual_Context"] = static_enhanced_context
        print(f"\n   📜 ĐÃ KHÓA SÁCH LUẬT:\n      '{static_enhanced_context}'\n")

        successful_traps =[]
        history_text = ""
        attempt = 1

        while len(successful_traps) < TARGET_PER_IMAGE and attempt <= MAX_ATTEMPTS:
            print(f"  👉 Lần thử: {attempt}/{MAX_ATTEMPTS} | Đã chốt: {len(successful_traps)}/{TARGET_PER_IMAGE} câu")
            
            gen_data = generate_adversarial_data(img_base64, notes, static_enhanced_context, history_text)
            if not gen_data:
                attempt += 1
                history_text += "=> Lần trước: AI lỗi định dạng.\n"
                continue 

            is_valid = semantic_gatekeeper(img_base64, gen_data.Adversarial_Prompt, gen_data.Logic_Steps, gen_data.Ground_Truth_Answer, static_enhanced_context)
            if not is_valid:
                print("   ⛔[Gatekeeper] TỪ CHỐI.")
                history_text += "=> Lần trước: Bị Gatekeeper loại do phi logic.\n"
                attempt += 1
                continue
            
            print("   ✅[Gatekeeper] DUYỆT!")
            
            file_name_without_ext = img_name.rsplit('.', 1)[0]
            trap_id = f"CSB_{file_name_without_ext}_{len(successful_traps) + 1:02d}"
            
            trap_record = {
                "ID": trap_id,
                "Attack_Type_1": gen_data.Attack_Type_1,
                "Attack_Type_2": gen_data.Attack_Type_2,
                "Attack_Type_3": gen_data.Attack_Type_3,
                "Adversarial_Prompt": gen_data.Adversarial_Prompt,
                "Victim_Illusion": gen_data.Victim_Illusion,
                "Logic_Steps": gen_data.Logic_Steps,
                "Ground_Truth_Answer": gen_data.Ground_Truth_Answer
            }
            successful_traps.append(trap_record)
            history_text += f"=> Lần {attempt}: Thành công! Hãy đổi điểm Neo sang vùng khác.\n"
            attempt += 1
            time.sleep(3)

        if successful_traps:
            img_record["Traps"] = successful_traps
            adv_data_results.append(img_record)
            
            with open(ADV_QUESTIONS_PATH, "w", encoding="utf-8") as f:
                json.dump(adv_data_results, f, ensure_ascii=False, indent=4)
            print(f"💾 Đã lưu {len(successful_traps)} bẫy.")

if __name__ == "__main__":
    run_generator_pipeline()