import os
import json
import sys
import shutil

# --- ĐỊNH TUYẾN ĐƯỜNG DẪN ---
current_dir = os.path.dirname(os.path.abspath(__file__)) 
parent_dir = os.path.dirname(current_dir)
credentials_dir = os.path.join(parent_dir, "credentials") 
sys.path.append(credentials_dir)

import config

VICTIM_RESPONSES_PATH = os.path.join(config.DATA_DIR, "pipeline_results", "03_victim_responses.json")

def repair_api_errors():
    print("🔧 BẮT ĐẦU DỌN DẸP LỖI API TRONG FILE SỐ 03...")
    
    if not os.path.exists(VICTIM_RESPONSES_PATH):
        print("❌ Không tìm thấy file 03_victim_responses.json")
        return

    with open(VICTIM_RESPONSES_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    repaired_count = 0

    for record in data:
        img_name = record.get("Image_Filename", "Unknown")
        
        for trap in record.get("Traps", []):
            for prefix in ["Gemini", "GPT", "Claude"]:
                ans_key = f"{prefix}_Answer"
                cot_key = f"{prefix}_CoT"
                correct_key = f"{prefix}_Is_Correct"
                reason_key = f"{prefix}_Failure_Reason"

                ans_val = str(trap.get(ans_key, ""))

                # Phát hiện câu trả lời chứa thông báo lỗi API (chứ không phải câu trả lời thật)
                if "Error" in ans_val or "Exception" in ans_val:
                    print(f"   🧹 Đã tẩy trắng lỗi mạng của {prefix} tại ảnh: {img_name}")
                    
                    # Reset lại câu trả lời và luồng suy nghĩ về rỗng
                    trap[ans_key] = ""
                    trap[cot_key] = ""
                    
                    # Xóa luôn trạng thái chấm điểm (nếu có) để làm mới hoàn toàn
                    if correct_key in trap:
                        trap[correct_key] = None
                    if reason_key in trap:
                        trap[reason_key] = ""
                        
                    repaired_count += 1

    if repaired_count > 0:
        # Sao lưu an toàn
        backup_path = VICTIM_RESPONSES_PATH + ".bak_repair"
        shutil.copy2(VICTIM_RESPONSES_PATH, backup_path)
        
        # Ghi đè file
        with open(VICTIM_RESPONSES_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
            
        print("\n" + "="*60)
        print(f"✅ ĐÃ CỨU HỘ THÀNH CÔNG {repaired_count} CHỖ BỊ RỚT MẠNG/RATE LIMIT.")
        print("="*60)
        print("👉 BƯỚC TIẾP THEO:")
        print("1. Chạy lại lệnh: python -u src/03_victim_inference.py (Nó sẽ chỉ gọi API điền vào đúng những chỗ vừa bị tẩy trắng).")
        print("2. Chạy lại lệnh: python -u src/04_auditor_agent.py (Để Giám khảo đọc file 03 và tạo ra file 04 mới tinh, chuẩn 100%).")
    else:
        print("\n🎉 Không tìm thấy lỗi API nào trong file 03. Dữ liệu đã sạch!")

if __name__ == "__main__":
    repair_api_errors()