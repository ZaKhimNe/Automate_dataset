import os
import json
import base64
import sys
from litellm import completion
from tenacity import retry, wait_random_exponential, stop_after_attempt

# --- ĐỊNH TUYẾN ĐƯỜNG DẪN ---
current_dir = os.path.dirname(os.path.abspath(__file__)) 
parent_dir = os.path.dirname(current_dir)
credentials_dir = os.path.join(parent_dir, "credentials") 
sys.path.append(credentials_dir)

import config

DATA_DIR = os.path.join(parent_dir, "data")
IMAGES_DIR = os.path.join(DATA_DIR, "images")
PIPELINE_RESULTS_DIR = os.path.join(DATA_DIR, "pipeline_results")
ADV_QUESTIONS_PATH = os.path.join(PIPELINE_RESULTS_DIR, "02_adversarial_questions.json")

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

@retry(wait=wait_random_exponential(min=config.RETRY_MIN_WAIT, max=config.RETRY_MAX_WAIT), stop=stop_after_attempt(3))
def supreme_gatekeeper(image_base64, adv_prompt, logic_steps, ground_truth, static_enhanced_context):
    """
    Ban thanh tra tối cao: Duyệt lại các câu hỏi đã được sinh ra.
    Sử dụng SMART_AI_MODEL (Pro) để đảm bảo độ khắt khe tối đa.
    """
    gatekeeper_schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "supreme_gatekeeper_schema",
            "schema": {
                "type": "object",
                "properties": {
                    "Reasoning": {
                        "type": "string", 
                        "description": "Phân tích xem câu hỏi có dính lỗi ảo giác vật lý (chi tiết không có thật trên ảnh), câu hỏi chủ quan, hoặc quá dễ đoán không."
                    },
                    "Violated_Rule": {
                        "type": "string", 
                        "description": "Ghi 'None' nếu hoàn hảo. Nếu có lỗi, ghi tên lỗi (VD: Visual Hallucination, Subjectivity, Single-Hop...)"
                    },
                    "Decision": {
                        "type": "string", 
                        "enum": ["KEEP", "DELETE"],
                        "description": "KEEP nếu Violated_Rule là None. DELETE nếu dính bất kỳ lỗi nào."
                    }
                },
                "required": ["Reasoning", "Violated_Rule", "Decision"]
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
    {{
        "Reasoning": "<Phân tích sâu sắc của bạn theo từng tiêu chí, tìm xem có lỗ hổng nào không>",
        "Violated_Rule": "<Số thứ tự của Lỗi vi phạm từ 1 đến 5, hoặc 'None' nếu hoàn hảo>",
        "Decision": "<Chỉ xuất 'YES' nếu Violated_Rule là None, ngược lại bắt buộc là 'NO'>"
    }}
    """

    user_text = f"Câu hỏi: {adv_prompt}\nLogic: {logic_steps}\nĐáp án: {ground_truth}"
    
    try:
        response = completion(
            model=config.SMART_AI_MODEL, # Dùng Pro model để chấm chéo
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
                    ]
                }
            ],
            response_format=gatekeeper_schema
        )
        
        clean_json = response.choices[0].message.content.replace("`" * 3 + "json", "").replace("`" * 3, "").strip()
        result_data = json.loads(clean_json)
        return result_data

    except Exception as e:
        print(f"   ⚠️ Lỗi Supreme Gatekeeper: {e}")
        return {"Decision": "KEEP", "Reasoning": "Lỗi API, tạm giữ lại", "Violated_Rule": "Error"}

def run_verifier():
    print("🕵️‍♂️ BẮT ĐẦU THANH TRA TOÀN BỘ CÂU HỎI ĐỐI KHÁNG...")
    
    if not os.path.exists(ADV_QUESTIONS_PATH):
        print(f"❌ Không tìm thấy file {ADV_QUESTIONS_PATH}")
        return

    with open(ADV_QUESTIONS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    total_traps_before = sum(len(record.get("Traps", [])) for record in data)
    print(f"📊 Tổng số câu hỏi hiện có: {total_traps_before}")

    deleted_count = 0
    clean_data = []

    for idx, record in enumerate(data):
        img_name = record["Image_File"]
        img_path = os.path.join(IMAGES_DIR, img_name)
        
        if not os.path.exists(img_path):
            print(f"⚠️ Mất file ảnh vật lý: {img_name}. Bỏ qua.")
            clean_data.append(record)
            continue

        print(f"\n[{idx+1}/{len(data)}] Đang thanh tra ảnh: {img_name}")
        img_base64 = encode_image(img_path)
        static_context = record.get("Visual_Context", "")
        
        valid_traps = []
        for trap_idx, trap in enumerate(record.get("Traps", [])):
            print(f"   🔍 Soi câu hỏi {trap_idx + 1}: {trap['Adversarial_Prompt'][:50]}...")
            
            evaluation = supreme_gatekeeper(
                img_base64, 
                trap["Adversarial_Prompt"], 
                trap["Logic_Steps"], 
                trap["Ground_Truth"], 
                static_context
            )
            
            if evaluation.get("Decision") == "DELETE":
                print(f"      ⛔ ĐÃ XÓA! Lỗi: {evaluation.get('Violated_Rule')}")
                print(f"      📝 Lý do: {evaluation.get('Reasoning')}")
                deleted_count += 1
            else:
                print("      ✅ ĐẠT CHUẨN.")
                valid_traps.append(trap)
        
        # Chỉ giữ lại bản ghi ảnh nếu nó còn ít nhất 1 câu hỏi bẫy hợp lệ
        if valid_traps:
            record["Traps"] = valid_traps
            clean_data.append(record)
        else:
            print(f"   🗑️ Bức ảnh này đã bị xóa sạch câu hỏi!")

    # Lưu bản backup và ghi đè
    backup_path = ADV_QUESTIONS_PATH + ".bak"
    os.rename(ADV_QUESTIONS_PATH, backup_path)
    
    with open(ADV_QUESTIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(clean_data, f, ensure_ascii=False, indent=4)

    total_traps_after = sum(len(record.get("Traps", [])) for record in clean_data)
    
    print("\n" + "="*50)
    print("🎉 QUÁ TRÌNH THANH TRA HOÀN TẤT!")
    print(f"📉 Số câu hỏi bị tàn sát: {deleted_count}")
    print(f"📈 Số câu hỏi tinh hoa còn lại: {total_traps_after}")
    print(f"💾 File backup cũ: {backup_path}")

if __name__ == "__main__":
    run_verifier()