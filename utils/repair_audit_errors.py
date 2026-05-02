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

AUDITED_JSON_PATH = os.path.join(config.DATA_DIR, "pipeline_results", "04_audited_responses.json")

def rescue_audit_errors():
    print("🚑 BẮT ĐẦU CHIẾN DỊCH CỨU HỘ: ĐẶT LẠI CÁC LỖI AUDIT_ERROR...")
    
    if not os.path.exists(AUDITED_JSON_PATH):
        print(f"❌ Không tìm thấy file {AUDITED_JSON_PATH}")
        return

    with open(AUDITED_JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    rescued_count = 0
    total_traps = 0

    for record in data:
        img_name = record.get("Image_Filename", "Unknown")
        
        for trap in record.get("Traps", []):
            total_traps += 1
            needs_rescue = False
            
            # Quét qua các cột lỗi
            for prefix in ["Gemini", "GPT", "Claude", "Ground_Truth"]:
                reason_key = f"{prefix}_Failure_Reason"
                correct_key = f"{prefix}_Is_Correct"
                
                # Nếu phát hiện Audit_Error do rớt mạng hoặc Rate Limit
                if trap.get(reason_key) == "Audit_Error":
                    # Xóa trạng thái để Trạm 4 chấm lại
                    trap[reason_key] = ""
                    trap[correct_key] = None
                    needs_rescue = True
                    print(f"   🔧 Đã reset lỗi {prefix} tại ảnh: {img_name}")
            
            if needs_rescue:
                rescued_count += 1

    # Sao lưu an toàn
    backup_path = AUDITED_JSON_PATH + ".bak_rescue"
    shutil.copy2(AUDITED_JSON_PATH, backup_path)

    # Ghi đè file
    with open(AUDITED_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    # Báo cáo
    print("\n" + "="*60)
    print("🚑 BÁO CÁO CỨU HỘ HOÀN TẤT")
    print("="*60)
    print(f" 📂 Tổng số câu hỏi đã quét: {total_traps}")
    print(f" ♻️ Số câu hỏi được reset:   {rescued_count}")
    print("="*60)
    if rescued_count > 0:
        print("\n👉 BƯỚC TIẾP THEO QUAN TRỌNG:")
        print("Hãy chạy lại lệnh: python -u src/04_auditor_agent.py")
        print("Hệ thống sẽ tự động chỉ gọi API để chấm bù lại những chỗ vừa được reset!")
    else:
        print("\n🎉 Không tìm thấy Audit_Error nào cần cứu hộ.")

if __name__ == "__main__":
    rescue_audit_errors()