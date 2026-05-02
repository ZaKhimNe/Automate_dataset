import base64
import os
import json
import io                  
from PIL import Image      
from litellm import completion
from tenacity import retry, wait_random_exponential, stop_after_attempt
import config

RAW_CAPTIONS_PATH = os.path.join(config.DATA_DIR, "raw_captions.json")

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

@retry(wait=wait_random_exponential(min=config.RETRY_MIN_WAIT, max=config.RETRY_MAX_WAIT), stop=stop_after_attempt(3))
def strict_curation_agent(image_base64):
    """Gemini Flash duyệt ảnh với bộ luật KHẮT KHE"""
    system_prompt = """
    Bạn là một chuyên gia phân loại dữ liệu ảnh học thuật khắt khe. Nhiệm vụ của bạn là trả về chữ "ACCEPT" hoặc "REJECT" dựa trên tiêu chí sau:
    
    [TIÊU CHÍ ACCEPT - Phải thỏa mãn TẤT CẢ]:
    1. Sơ đồ kiến trúc hệ thống (Architecture/Framework), pipeline thuật toán, hoặc lưu đồ (Flowchart) có tính phức tạp kỹ thuật (có các node, khối chức năng, mũi tên liên kết).
    2. Biểu đồ khoa học (Line/Bar/Scatter chart) CHỨA NHIỀU thành phần dữ liệu, có chú giải (legend) phức tạp đòi hỏi khả năng đọc hiểu và suy luận phân tích hiệu năng/kết quả.
    3. Hình ảnh tổng hợp (Multi-panel figure) mô tả phương pháp nghiên cứu, phân chia dữ liệu, hoặc bản đồ không gian địa lý có gắn nhãn phân tích rõ ràng.

    [TIÊU CHÍ REJECT - Chứa MỘT TRONG CÁC dấu hiệu sau là LOẠI NGAY]:
    1. Giao diện người dùng (UI), màn hình ứng dụng, hoặc các khung Chat/Hội thoại tin nhắn.
    2. Bức ảnh bị cắt lỗi chứa các đoạn văn bản dài (body text) của bài báo.
    3. Hình ảnh chụp phong cảnh, đời sống thực tế không mang tính chất biểu diễn dữ liệu khoa học.
    4. Biểu đồ hoặc hình học quá đơn giản (ví dụ: chỉ có 1 đường duy nhất, 1 hình tròn đơn giản không có chú giải chi tiết).
    
    CHỈ TRẢ VỀ ĐÚNG 1 TỪ DUY NHẤT: "ACCEPT" hoặc "REJECT".
    """
    try:
        response = completion(
            model=config.SMART_AI_MODEL, 
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Hãy phân loại bức ảnh này."},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
                    ]
                }
            ]
        )
        result = response.choices[0].message.content.strip().upper()
        return "ACCEPT" in result
    except Exception as e:
        print(f"⚠️ Lỗi API: {e}")
        return False

def run_re_scan():
    print("🧹 BẮT ĐẦU CHIẾN DỊCH TỔNG VỆ SINH THƯ MỤC CHÍNH THỨC...")
    
    # 1. Tải JSON hiện tại lên
    if not os.path.exists(RAW_CAPTIONS_PATH):
        print("❌ Không tìm thấy file raw_captions.json!")
        return
        
    with open(RAW_CAPTIONS_PATH, "r", encoding="utf-8") as f:
        try:
            raw_captions = json.load(f)
        except json.JSONDecodeError:
            print("❌ File JSON trống hoặc bị lỗi!")
            return

    image_files = [f for f in os.listdir(config.IMAGES_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print("🤷‍♂️ Không có ảnh nào trong data/images để dọn dẹp.")
        return

    deleted_count = 0
    survived_count = 0
    processed = 0
    for img_name in image_files:
        img_path = os.path.join(config.IMAGES_DIR, img_name)
        print(f" - Đang soi lại: {img_name}...", end=" ")
        
        img_base64 = encode_image(img_path)
        processed += 1
        
        try:
            is_accepted = strict_curation_agent(img_base64)
            
            if is_accepted:
                print("✅ VẪN ĐẠT CHUẨN (Giữ lại)")
                survived_count += 1
            else:
                os.remove(img_path)
                if img_name in raw_captions:
                    del raw_captions[img_name] 
                deleted_count += 1
                print("🗑️ PHÁT HIỆN RÁC (Đã xóa ảnh & JSON)")
                
            if processed % 20 == 0:
                with open(RAW_CAPTIONS_PATH, "w", encoding="utf-8") as f:
                    json.dump(raw_captions, f, ensure_ascii=False, indent=4)
                    
        except Exception as e:
            print(f"\n   ⚠️ LỖI NGHIÊM TRỌNG (Hết Quota/Mạng): {e}")
            print("   => Dừng chiến dịch để bảo toàn các ảnh còn lại.")
            break

    with open(RAW_CAPTIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(raw_captions, f, ensure_ascii=False, indent=4)
        
    print("\n=========================================")
    print(f"🎉 TỔNG KẾT VỆ SINH:")
    print(f"   - Đã xóa: {deleted_count} ảnh lỗi.")
    print(f"   - Sống sót: {survived_count} ảnh chất lượng cao.")
    print("👉 Bây giờ thư mục data/images của bạn đã sạch bóng, sẵn sàng chạy mainpipeline.py!")

if __name__ == "__main__":
    run_re_scan()