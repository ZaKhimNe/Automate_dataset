import base64
import os
import json
import io
import sys
from PIL import Image
import litellm
from litellm import completion
from tenacity import retry, wait_random_exponential, stop_after_attempt

litellm.drop_params = True
litellm._turn_on_debug()

current_dir = os.path.dirname(os.path.abspath(__file__)) 
parent_dir = os.path.dirname(current_dir)
credentials_dir = os.path.join(parent_dir, "credentials") 
sys.path.append(credentials_dir)
import config

# --- ĐỊNH NGHĨA CÁC ĐƯỜNG DẪN RAW DATA ---
RAW_IMAGES_DIR = os.path.join(config.DATA_DIR, "raw_data", "raw_images")
RAW_JSON_PATH = os.path.join(config.DATA_DIR, "raw_data", "crawled_raw.json")
FINAL_IMG_DIR = os.path.join(config.DATA_DIR, "images")

PIPELINE_RESULTS_DIR = os.path.join(config.DATA_DIR, "pipeline_results")
CLEAN_JSON_PATH = os.path.join(PIPELINE_RESULTS_DIR, "01_clean_data.json")

os.makedirs(FINAL_IMG_DIR, exist_ok=True)
os.makedirs(PIPELINE_RESULTS_DIR, exist_ok=True)

def encode_image(image_path, max_size=1536):
    with Image.open(image_path) as img:
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        width, height = img.size
        if width > max_size or height > max_size:
            scaling_factor = max_size / float(max(width, height))
            new_size = (int(width * scaling_factor), int(height * scaling_factor))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        buffered = io.BytesIO()
        img.save(buffered, format="PNG", optimize=True)
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

def resize_and_save_physical(src_path, dest_path, max_size=1536):
    with Image.open(src_path) as img:
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        width, height = img.size
        if width > max_size or height > max_size:
            scaling_factor = max_size / float(max(width, height))
            new_size = (int(width * scaling_factor), int(height * scaling_factor))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        img.save(dest_path, format="PNG", optimize=True)

@retry(wait=wait_random_exponential(min=10, max=config.RETRY_MAX_WAIT), stop=stop_after_attempt(config.MAX_RETRIES))
def curation_agent(image_base64):
    """Phân loại ảnh và trích xuất Image_Type"""
    
    curation_schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "curation_assessment",
            "schema": {
                "type": "object",
                "properties": {
                    "Decision": {"type": "string", "enum":["ACCEPT", "REJECT"]},
                    "Image_Type": {"type": "string"}
                },
                "required":["Decision", "Image_Type"]
            }
        }
    }

    system_prompt = """
    Bạn là một chuyên gia phân loại dữ liệu ảnh học thuật khắt khe.[🔴 TIÊU CHÍ REJECT - LOẠI NGAY LẬP TỨC NẾU CÓ]:
    1. Dính chữ Caption hoặc Body Text ở viền ngoài.
    2. Biểu đồ định lượng thuần túy (Line, Bar, Scatter, Pie).
    3. Giao diện UI, hội thoại chat.
    4. Hình ảnh chụp thực tế đơn thuần.

    [🟢 TIÊU CHÍ ACCEPT - ĐỂ GIỮ LẠI]:
    1. Sơ đồ Pipeline/Kiến trúc hệ thống (System Architecture).
    2. Lưu đồ giải thuật (Flowchart).
    3. Sơ đồ cây phân loại (Taxonomy/Tree).
    4. Sơ đồ có ảnh lồng ảnh (Nested Visual Context).
    5. Cấu trúc mạng (Network Topology).

    Hãy trả về định dạng JSON gồm "Decision" (ACCEPT/REJECT) và "Image_Type" (Nếu REJECT thì để Image_Type là "N/A", nếu ACCEPT thì phân loại nó vào 1 trong 5 nhóm trên hoặc tự đặt tên ngắn gọn).
    """
    response = completion(
        model=config.VICTIM_AI_MODEL, 
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content":[
                    {"type": "text", "text": "Hãy phân loại bức ảnh này."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
                ]
            }
        ],
        response_format=curation_schema
    )
    clean_json = response.choices[0].message.content.replace("```json", "").replace("```", "").strip()
    return json.loads(clean_json)

def run_curation_pipeline():
    print("🔎 BẮT ĐẦU GIÁM ĐỊNH & LỌC ẢNH RÁC...")
    
    if not os.path.exists(RAW_JSON_PATH) or not os.path.exists(RAW_IMAGES_DIR):
        print("⏭️ LỖI: Không tìm thấy dữ liệu raw!")
        return

    with open(RAW_JSON_PATH, "r", encoding="utf-8") as f:
        crawled_data = json.load(f)

    # Đọc file clean data cũ (để chạy nối tiếp)
    clean_data_list =[]
    if os.path.exists(CLEAN_JSON_PATH):
        try:
            with open(CLEAN_JSON_PATH, "r", encoding="utf-8") as f:
                clean_data_list = json.load(f)
        except json.JSONDecodeError:
            clean_data_list = []

    processed_images = {item["Image_Filename"] for item in clean_data_list}
    processed_count = 0
    accepted_count = 0

    for img_name, meta in list(crawled_data.items()): # Ép kiểu list để pop/delete an toàn
        if img_name in processed_images:
            continue

        img_path = os.path.join(RAW_IMAGES_DIR, img_name)
        if not os.path.exists(img_path):
            continue

        processed_count += 1
        print(f" - Đang soi: {img_name}...", end=" ")
        img_base64 = encode_image(img_path)

        try:
            curation_result = curation_agent(img_base64)
            is_accepted = (curation_result.get("Decision") == "ACCEPT")
            
            if is_accepted:
                dest_path = os.path.join(FINAL_IMG_DIR, img_name)
                resize_and_save_physical(img_path, dest_path)
                os.remove(img_path) 
                
                # CẬP NHẬT THEO CẤU TRÚC "CỤM 1"
                record = {
                    "Image_Filename": img_name,
                    "Source_Link": meta.get("Source_Link", "N/A"),
                    "Image_Type": curation_result.get("Image_Type", "Diagram"),
                    "Caption": meta.get("Caption", "") # Tạm giữ để Trạm 2 đọc làm Sách luật
                }
                clean_data_list.append(record)
                accepted_count += 1
                print(f"✅ ACCEPT ({record['Image_Type']})")
            else:
                os.remove(img_path)
                print("⛔ REJECT (Đã xóa)")
            
            # Xóa khỏi crawled_data để đánh dấu đã xử lý
            del crawled_data[img_name]

            if processed_count % 10 == 0:
                with open(CLEAN_JSON_PATH, "w", encoding="utf-8") as f:
                    json.dump(clean_data_list, f, ensure_ascii=False, indent=4)
                with open(RAW_JSON_PATH, "w", encoding="utf-8") as f:
                    json.dump(crawled_data, f, ensure_ascii=False, indent=4)
                print(f"   💾 [Checkpoint] Đã lưu tiến độ.")

        except Exception as e:
            print(f"\n   ⚠️ LỖI NGHIÊM TRỌNG: {e}")
            break 
    
    # Lưu cuối cùng
    with open(CLEAN_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(clean_data_list, f, ensure_ascii=False, indent=4)
    with open(RAW_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(crawled_data, f, ensure_ascii=False, indent=4)

    print(f"\n🎉 HOÀN TẤT! Đã lưu file sạch tại: {CLEAN_JSON_PATH}")

if __name__ == "__main__":
    run_curation_pipeline()