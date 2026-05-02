import base64
import os
import json
import sys
import time
import csv
import litellm
from litellm import completion
from tenacity import retry, wait_random_exponential, stop_after_attempt

litellm.drop_params = True
# litellm._turn_on_debug()
current_dir = os.path.dirname(os.path.abspath(__file__)) 
parent_dir = os.path.dirname(current_dir)
credentials_dir = os.path.join(parent_dir, "credentials") 
sys.path.append(credentials_dir)

import config

DATA_DIR = os.path.join(parent_dir, "data")
IMAGES_DIR = os.path.join(DATA_DIR, "images")
PIPELINE_RESULTS_DIR = os.path.join(DATA_DIR, "pipeline_results")

VICTIM_RESPONSES_PATH = os.path.join(PIPELINE_RESULTS_DIR, "03_victim_responses.json")
AUDITED_RESULTS_PATH = os.path.join(PIPELINE_RESULTS_DIR, "04_audited_responses.json")
FINAL_CSV_PATH = os.path.join(parent_dir, "Dataset.csv")

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

@retry(wait=wait_random_exponential(min=config.RETRY_MIN_WAIT, max=config.RETRY_MAX_WAIT), stop=stop_after_attempt(3))
def auditor_evaluate(image_base64, adversarial_prompt, logic_steps, ground_truth, victim_cot, victim_answer, static_enhanced_context):
    print("   ⚖️ [Giám Khảo] Đang phân tích CoT và chấm điểm...")
    
    response_schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "auditor_evaluation_schema",
            "schema": {
                "type": "object",
                "properties": {
                    "Is_Correct": {"type": "boolean"},
                    "Failure_Reason": {"type": "string"},
                    "Feedback_Details": {"type": "string"}
                },
                "required":["Is_Correct", "Failure_Reason", "Feedback_Details"]
            }
        }
    }

    system_prompt = f"""
    Bạn là Giám khảo (Auditor) tối cao. Nhiệm vụ: Chấm điểm bài làm của Thí sinh dựa trên Đáp án đúng.[SÁCH LUẬT HỆ THỐNG]:
    "{static_enhanced_context}"
    
    QUY TẮC CHẤM:
    1. ĐÚNG (Is_Correct = true): Khi[Đáp án Thí sinh] khớp với [Đáp án đúng]. Lưu ý: Thí sinh có thể viết hơi khác về định dạng (Ví dụ: 'Module A' và 'A'), chỉ cần đúng ý nghĩa cốt lõi. Khi đó, Failure_Reason BẮT BUỘC là "Survived".
    2. SAI (Is_Correct = false): Đọc kỹ[Luồng suy nghĩ của Thí sinh] so với [Barem logic giải] và chọn ĐÚNG 1 MÃ LỖI DUY NHẤT trong danh sách sau:
       - "Visual_Miss": Nhìn sai màu sắc, sót mũi tên, sai vị trí không gian.
       - "Pretrained_Overreliance": Bỏ qua hình ảnh, dùng kiến thức Khoa học Máy tính mặc định để đoán mò.
       - "Rule_Ignored": Không áp dụng các quy tắc phản thực tế trong Sách luật.
       - "Reasoning_Error": Nhìn đúng ảnh nhưng suy luận logic bắc cầu (A->B->C) bị sai.
       - "Hallucination": Bịa ra module/đường nối không hề tồn tại.
       - "Model_Refusal": Thí sinh từ chối trả lời hoặc xin lỗi.

    [ĐỊNH DẠNG ĐẦU RA]: Tuân thủ chuẩn JSON. Cột Feedback_Details viết 1 câu thật ngắn gọn giải thích tại sao sai.
    """

    user_text = f"""[ĐỀ BÀI & BAREM]
    Câu hỏi: {adversarial_prompt}
    Barem Logic giải: {logic_steps}
    Đáp án đúng: {ground_truth}

    [BÀI LÀM CỦA THÍ SINH]
    Luồng suy nghĩ (CoT): {victim_cot}
    Đáp án Thí sinh chốt: {victim_answer}
    """

    try:
        response = completion(
            model=config.SMART_AI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content":[{"type": "text", "text": user_text}, {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}]}
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
        return json.loads(clean_json)
    except Exception as e:
        print(f"❌ Lỗi API ở bước Audit: {e}")
        return {"Is_Correct": False, "Failure_Reason": "Audit_Error", "Feedback_Details": str(e)}

def export_to_csv(data_list):
    """Bóp phẳng dữ liệu JSON thành Cấu trúc CSV 4 Cụm siêu đẹp"""
    print(f"\n📊 Đang xuất file dữ liệu chuẩn: {FINAL_CSV_PATH} ...")
    
    model_prefixes = set()
    for record in data_list:
        for trap in record.get("Traps",[]):
            for key in trap.keys():
                if key.endswith("_Answer"):
                    model_prefixes.add(key.replace("_Answer", ""))
    model_prefixes = sorted(list(model_prefixes)) #['Claude', 'GPT', 'Gemini']
    total_models = len(model_prefixes)

    headers =[
        "ID", "Image_Filename", "Source_Link", "Image_Type", 
        "Difficulty_Tier",  
        "Visual_Context", "Adversarial_Prompt", "Attack_Type_1", "Attack_Type_2", "Attack_Type_3", 
        "Victim_Illusion", "Logic_Steps", "Ground_Truth_Answer" 
    ]

    
    for prefix in model_prefixes:
        headers.extend([f"{prefix}_CoT", f"{prefix}_Answer", f"{prefix}_Is_Correct", f"{prefix}_Failure_Reason"])
        
    headers.append("Metadata_JSON") 


    with open(FINAL_CSV_PATH, "w", encoding="utf-8-sig", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()

        for record in data_list:
            img_filename = record.get("Image_Filename", "")
            source_link = record.get("Source_Link", "")
            img_type = record.get("Image_Type", "")
            static_context = record.get("Visual_Context", "")

            for trap in record.get("Traps",[]):
                fail_count = 0

                for prefix in model_prefixes:
                    if trap.get(f"{prefix}_Is_Correct") is False:
                        fail_count += 1
                
                if fail_count == 0:
                    tier_label = "Tier 1 (Foundation)"
                elif fail_count == total_models: 
                    tier_label = "Tier 3 (Blindspot)"
                else:
                    tier_label = "Tier 2 (Discriminative)"

                row = {
                    "ID": trap.get("ID", ""),
                    "Image_Filename": img_filename,
                    "Source_Link": source_link,
                    "Image_Type": img_type,
                    "Difficulty_Tier": tier_label,
                    "Visual_Context": static_context,
                    "Adversarial_Prompt": trap.get("Adversarial_Prompt", ""),
                    "Attack_Type_1": trap.get("Attack_Type_1", ""),
                    "Attack_Type_2": trap.get("Attack_Type_2", ""),
                    "Attack_Type_3": trap.get("Attack_Type_3", ""),
                    "Victim_Illusion": trap.get("Victim_Illusion", ""),
                    "Logic_Steps": trap.get("Logic_Steps", ""),
                    "Ground_Truth_Answer": trap.get("Ground_Truth_Answer", "")
                }
                

                for prefix in model_prefixes:
                    row[f"{prefix}_CoT"] = trap.get(f"{prefix}_CoT", "")
                    row[f"{prefix}_Answer"] = trap.get(f"{prefix}_Answer", "")
                    row[f"{prefix}_Is_Correct"] = trap.get(f"{prefix}_Is_Correct", "")
                    row[f"{prefix}_Failure_Reason"] = trap.get(f"{prefix}_Failure_Reason", "")

                # Điền dữ liệu Cụm 4
                row["Metadata_JSON"] = json.dumps(trap.get("Metadata_JSON", {}), ensure_ascii=False)

                writer.writerow(row)

    print("✅ Đã xuất CSV thành công! Bạn có thể mở bằng Excel/Google Sheets.")

def run_auditor_pipeline():
    print("🚀 BƯỚC 4: KHỞI ĐỘNG HỘI ĐỒNG GIÁM KHẢO (AUDITOR AGENT)...")
    
    if os.path.exists(AUDITED_RESULTS_PATH):
        print(f"📥 Tìm thấy bản lưu nháp. Đang đọc tiếp từ: 04_audited_responses.json")
        with open(AUDITED_RESULTS_PATH, "r", encoding="utf-8") as f:
            audited_data = json.load(f)
    elif os.path.exists(VICTIM_RESPONSES_PATH):
        print(f"📥 Bắt đầu chấm mới từ: 03_victim_responses.json")
        with open(VICTIM_RESPONSES_PATH, "r", encoding="utf-8") as f:
            audited_data = json.load(f)
    else:
        print(f"⏭️ LỖI: Không tìm thấy file dữ liệu nào. Hãy chạy Trạm 3 trước!")
        return



    for img_idx, record in enumerate(audited_data):
        img_name = record["Image_Filename"]
        img_path = os.path.join(IMAGES_DIR, img_name)
        
        img_base64 = None 
        static_context = record.get("Visual_Context", "")

        for trap_idx, trap in enumerate(record.get("Traps",[])):
            
            for key in list(trap.keys()):
                if key.endswith("_Answer") and key != "Ground_Truth_Answer": 
                    prefix = key.replace("_Answer", "")
                    is_correct_key = f"{prefix}_Is_Correct"
                    failure_reason_key = f"{prefix}_Failure_Reason"
                    needs_audit = (
                        trap.get(is_correct_key) is None or 
                        trap.get(failure_reason_key) in ["Audit_Error", "Unknown"]
                    )
                    
                    has_answer = trap.get(key) != "" and trap.get(key) is not None


                    if needs_audit and has_answer:
                        
                        if img_base64 is None and os.path.exists(img_path):
                            img_base64 = encode_image(img_path)
                            print(f"\n[{img_idx+1}/{len(audited_data)}] Đang MỞ ẢNH & chấm điểm: {img_name}")

                        print(f"   🎯 Chấm câu {trap_idx+1} cho mô hình: [{prefix}]")

                        
                        audit_result = auditor_evaluate(
                            image_base64=img_base64,
                            adversarial_prompt=trap["Adversarial_Prompt"],
                            logic_steps=trap["Logic_Steps"],
                            ground_truth=trap["Ground_Truth_Answer"],
                            victim_cot=trap.get(f"{prefix}_CoT", ""),
                            victim_answer=trap.get(key, ""),
                            static_enhanced_context=static_context
                        )

                        # Ghi nhận điểm số
                        trap[is_correct_key] = audit_result.get("Is_Correct", False)
                        trap[failure_reason_key] = audit_result.get("Failure_Reason", "Unknown")
                        
                        metadata = trap.get("Metadata_JSON", {})
                        metadata[f"{prefix}_Feedback"] = audit_result.get("Feedback_Details", "")
                        trap["Metadata_JSON"] = metadata

                        time.sleep(3)
                    elif not needs_audit and has_answer:
                        print(f"   ⏭️ Bỏ qua câu {trap_idx+1} [{prefix}] - Đã chấm xong.")


        with open(AUDITED_RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump(audited_data, f, ensure_ascii=False, indent=4)

    print("\n🎉 HOÀN TẤT CHẤM ĐIỂM!")
    
    # 3. Xuất kết quả ra file CSV cuối cùng
    export_to_csv(audited_data)

if __name__ == "__main__":
    run_auditor_pipeline()