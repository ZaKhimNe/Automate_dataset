import base64
import os
import json
import sys
import time
import re
from datetime import datetime
import litellm
from litellm import completion
from tenacity import retry, wait_random_exponential, stop_after_attempt
import jsonlines

litellm.drop_params = True
#litellm._turn_on_debug()
current_dir = os.path.dirname(os.path.abspath(__file__)) 
parent_dir = os.path.dirname(current_dir)
credentials_dir = os.path.join(parent_dir, "credentials") 
sys.path.append(credentials_dir)

import config

DATA_DIR = os.path.join(parent_dir, "data")
IMAGES_DIR = os.path.join(DATA_DIR, "images")
PIPELINE_RESULTS_DIR = os.path.join(DATA_DIR, "pipeline_results")

ADV_QUESTIONS_PATH = os.path.join(PIPELINE_RESULTS_DIR, "02_adversarial_questions.json")
VICTIM_RESPONSES_PATH = os.path.join(PIPELINE_RESULTS_DIR, "03_victim_responses.json")
VICTIM_RESPONSES_JSONL = os.path.join(PIPELINE_RESULTS_DIR, "03_victim_responses.jsonl")

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def parse_victim_response(raw_text):
    """Hàm dùng Regex để bóc tách CoT và Đáp án từ định dạng XML"""
    cot = ""
    answer = raw_text 
    
    cot_match = re.search(r"<Phân_tích>(.*?)</Phân_tích>", raw_text, re.DOTALL | re.IGNORECASE)
    ans_match = re.search(r"<Đáp_án_Cuối_cùng>(.*?)</Đáp_án_Cuối_cùng>", raw_text, re.DOTALL | re.IGNORECASE)
    
    if cot_match:
        cot = cot_match.group(1).strip()
    if ans_match:
        answer = ans_match.group(1).strip()
        
    return cot, answer

@retry(wait=wait_random_exponential(min=config.RETRY_MIN_WAIT, max=config.RETRY_MAX_WAIT), stop=stop_after_attempt(3))
def ask_victim(model_name, image_base64, adversarial_prompt, static_enhanced_context):
    victim_prompt = f"""
    Bạn là một Chuyên gia Phân tích Hệ thống (Systems Analyst). Nhiệm vụ của bạn là giải quyết câu hỏi dựa trên sơ đồ hệ thống được cung cấp.[TÀI LIỆU KỸ THUẬT / NGỮ CẢNH HỆ THỐNG ĐÍNH KÈM]:
    {static_enhanced_context}[CÂU HỎI TỪ NGƯỜI DÙNG]:
    {adversarial_prompt}
    [QUY TẮC PHÂN TÍCH BẮT BUỘC]:
    1. TUYỆT ĐỐI TIN TƯỞNG VÀO HÌNH ẢNH: Sơ đồ này có thể chứa các thiết kế phi truyền thống. Không sử dụng kiến thức mặc định của Khoa học Máy tính để đoán mò.
    2. SỬ DỤNG SÁCH LUẬT: Tuân thủ nghiêm ngặt các quy định được nêu trong TÀI LIỆU KỸ THUẬT.[ĐỊNH DẠNG ĐẦU RA BẮT BUỘC]:
    <Phân_tích>
    (Trình bày chi tiết từng bước bạn dò đường trên bức ảnh. Mắt bạn nhìn thấy gì? Đi qua node nào? Mũi tên hướng đi đâu?)
    </Phân_tích>
    <Đáp_án_Cuối_cùng>
    (Chỉ ghi tên cụ thể của đáp án cuối cùng, ngắn gọn nhất có thể)
    </Đáp_án_Cuối_cùng>
    """

    try:
        completion_args = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": "Bạn là chuyên gia thị giác máy tính khách quan."},
                {
                    "role": "user",
                    "content":[
                        {"type": "text", "text": victim_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
                    ]
                }
            ],
        }

        if "gemini" in model_name.lower() or "vertex_ai" in model_name.lower():
            completion_args["safety_settings"] = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
            ]

        response = completion(**completion_args)
        
        if response is None or not response.choices[0].message.content:
            return "Error: Empty response"
        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"      ❌ Lỗi khi gọi {model_name}: {e}")
        return f"Error: API Exception - {e}"


def run_victim_inference():
    print("🚀 BƯỚC 3: MANG ĐỀ THI ĐI TEST CÁC MÔ HÌNH (VICTIM INFERENCE)...")
    
    if not os.path.exists(ADV_QUESTIONS_PATH):
        print(f"⏭️ LỖI: Không tìm thấy {ADV_QUESTIONS_PATH}!")
        return

    with open(ADV_QUESTIONS_PATH, "r", encoding="utf-8") as f:
        adv_data = json.load(f)

    victim_data =[]
    if os.path.exists(VICTIM_RESPONSES_PATH):
        try:
            with open(VICTIM_RESPONSES_PATH, "r", encoding="utf-8") as f:
                victim_data = json.load(f)
        except json.JSONDecodeError:
            pass

    processed_dict = {record["Image_Filename"]: record for record in victim_data}

    MODELS_TO_TEST = {
        "Gemini": config.VICTIM_AI_MODEL,  
        "GPT": config.GPT5_MODEL,        
        "Claude": config.CLAUDE_MODEL   
    }

    for img_idx, record in enumerate(adv_data):
        img_name = record["Image_Filename"]
        print(f"\n[{img_idx+1}/{len(adv_data)}] Đang xử lý ảnh: {img_name}")
        
        img_path = os.path.join(IMAGES_DIR, img_name)
        if not os.path.exists(img_path):
            continue
            
        img_base64 = None 
        static_context = record.get("Visual_Context", "")
        
        current_victim_record = processed_dict.get(img_name, {
            "Image_Filename": img_name,
            "Source_Link": record.get("Source_Link", ""),
            "Image_Type": record.get("Image_Type", "Diagram"),
            "Visual_Context": static_context,
            "Traps":[]
        })

        for trap_idx, trap in enumerate(record.get("Traps", [])):
            print(f"   🎯 Câu {trap_idx+1}: {trap['Adversarial_Prompt'][:60]}...")
            
            existing_traps = current_victim_record.get("Traps",[])
            existing_trap = next((t for t in existing_traps if t.get("ID") == trap.get("ID")), None)
            
            if existing_trap is None:
                existing_trap = trap.copy()
                
                existing_trap["Metadata_JSON"] = {
                    "Generation_Date": datetime.now().isoformat(),
                    "Generator_Model": config.SMART_AI_MODEL
                }
                existing_traps.append(existing_trap)
            
            for prefix, model_api_name in MODELS_TO_TEST.items():
                cot_key = f"{prefix}_CoT"
                ans_key = f"{prefix}_Answer"
                
                if f"{prefix}_Is_Correct" not in existing_trap:
                    existing_trap[f"{prefix}_Is_Correct"] = None
                    existing_trap[f"{prefix}_Failure_Reason"] = ""
                
                ans_value = str(existing_trap.get(ans_key, ""))
                cot_value = str(existing_trap.get(cot_key, ""))

                needs_inference = (
                    ans_value.strip() == "" or 
                    "Error" in ans_value or 
                    "Error" in cot_value
                )

                if needs_inference:
                    if img_base64 is None:
                        img_base64 = encode_image(img_path)
                        print(f"\n[{img_idx+1}/{len(adv_data)}] Đang MỞ ẢNH xử lý: {img_name}")

                    print(f"   🎯 Hỏi câu {trap_idx+1} cho mô hình: [{prefix}]")
                    raw_response = ask_victim(model_api_name, img_base64, trap["Adversarial_Prompt"], static_context)
                    
                    cot_text, ans_text = parse_victim_response(raw_response)
                    existing_trap[cot_key] = cot_text
                    existing_trap[ans_key] = ans_text
                    
                    time.sleep(3)
            
            current_victim_record["Traps"] = existing_traps
        
        processed_dict[img_name] = current_victim_record
        
        with open(VICTIM_RESPONSES_PATH, "w", encoding="utf-8") as f:
            json.dump(list(processed_dict.values()), f, ensure_ascii=False, indent=4)
            
        with jsonlines.open(VICTIM_RESPONSES_JSONL, mode='w') as writer:
            for rec in processed_dict.values():
                writer.write(rec)

    print("\n🎉 HOÀN TẤT TRẠM 3!")

if __name__ == "__main__":
    run_victim_inference()