import os
import pandas as pd
import sys

current_dir = os.path.dirname(os.path.abspath(__file__)) 
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

DATASET_PATH = os.path.join(parent_dir, "Dataset.csv")

def count_dataset():
    if not os.path.exists(DATASET_PATH):
        print(f"❌ Không tìm thấy file {DATASET_PATH}")
        return

    # Đọc dữ liệu
    df = pd.read_csv(DATASET_PATH, encoding='utf-8-sig')
    
    total_traps = len(df)
    total_images = df['Image_Filename'].nunique()

    print("\n" + "="*70)
    print("📊 BÁO CÁO THỐNG KÊ DATASET (RED TEAMING)")
    print("="*70)

    # --- 1. TỔNG QUAN ---
    print("\n[1. TỔNG QUAN DỮ LIỆU]")
    print(f"  - Tổng số câu hỏi (Traps): {total_traps}")
    print(f"  - Tổng số hình ảnh độc lập: {total_images}")
    
    # --- 2. CHIẾN THUẬT TẤN CÔNG (Top 5) ---
    print("\n[2. CHIẾN THUẬT TẤN CÔNG (Attack Types)]")
    attack_counts = df['Attack_Type_1'].value_counts()
    for attack, count in attack_counts.head(5).items():
        print(f"  - {attack}: {count} câu ({count/total_traps*100:.1f}%)")

    # --- 3. HIỆU SUẤT MÔ HÌNH (PERFORMANCE) ---
    print("\n[3. HIỆU SUẤT MÔ HÌNH (Model Accuracy)]")
    models = ['Gemini', 'GPT', 'Claude']
    for model in models:
        correct_col = f"{model}_Is_Correct"
        if correct_col in df.columns:
            # Lọc bỏ những câu bị lỗi API (None/NaN)
            valid_answers = df[df[correct_col].notna()]
            correct_count = valid_answers[correct_col].sum()
            total_valid = len(valid_answers)
            accuracy = (correct_count / total_valid * 100) if total_valid > 0 else 0
            
            print(f"  - {model: <8} : Đúng {correct_count}/{total_valid} câu | Tỷ lệ Accuracy: {accuracy:.1f}%")

    # --- 4. ĐỘ KHÓ CỦA BẪY (TRAP DIFFICULTY TIERS) ---
    print("\n[4. ĐỘ KHÓ CỦA BẪY (Số lượng Model bị lừa)]")
    
    # Tính số model bị lừa cho mỗi câu
    df['Models_Fooled'] = 0
    for model in models:
        correct_col = f"{model}_Is_Correct"
        if correct_col in df.columns:
            # Nếu Is_Correct là False, tức là bị lừa (+1)
            df['Models_Fooled'] += (df[correct_col] == False).astype(int)

    tier_counts = df['Models_Fooled'].value_counts().sort_index()
    for tier, count in tier_counts.items():
        tier_name = "Quá dễ" if tier == 0 else "Sát thủ" if tier == 3 else f"Lừa {tier} model"
        print(f"  - {tier_name: <15}: {count} câu ({count/total_traps*100:.1f}%)")

    # --- 5. NGUYÊN NHÂN THẤT BẠI (FAILURE ANALYSIS) ---
    print("\n[5. NGUYÊN NHÂN THẤT BẠI PHỔ BIẾN (Failure Reasons)]")
    all_failures = []
    for model in models:
        reason_col = f"{model}_Failure_Reason"
        if reason_col in df.columns:
            # Lấy các lỗi, loại bỏ 'Survived' (làm đúng) và 'Unknown'
            failures = df[~df[reason_col].isin(['Survived', 'Unknown', 'Audit_Error'])][reason_col].dropna()
            all_failures.extend(failures.tolist())
    
    if all_failures:
        failure_series = pd.Series(all_failures)
        failure_counts = failure_series.value_counts()
        for reason, count in failure_counts.items():
            print(f"  - {reason: <25}: {count} lần")
    else:
        print("  - Chưa có dữ liệu phân tích lỗi.")

    print("="*70 + "\n")

if __name__ == "__main__":
    count_dataset()