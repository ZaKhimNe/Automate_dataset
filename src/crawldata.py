import fitz  # PyMuPDF
import json
import os
import re

current_dir = os.path.dirname(os.path.abspath(__file__)) 
parent_dir = os.path.dirname(current_dir)

RAW_IMG_DIR = "data/raw_data/raw_images"
RAW_JSON_PATH = "data/raw_data/crawled_raw.json"
PDF_DIR = r"C:\Users\GIGA\VisualStudio2022Projects\job\AI_RedTeaming_Pipeline\data\pdfs" 
SOURCE_LINKS_FILE = os.path.join(parent_dir, "source_links.json")
EXTRACT_DPI = 400 

os.makedirs(RAW_IMG_DIR, exist_ok=True)
os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(os.path.dirname(RAW_JSON_PATH), exist_ok=True)

def load_source_links_dictionary(filepath):
    """Đọc file JSON chứa danh sách link và tạo thành một Dictionary (key=id, value=link) để tra cứu siêu tốc"""
    link_dict = {}
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                for item in data:
                    paper_id = item.get("id")
                    link = item.get("source_link", "")
                    
                    if paper_id and link:
                        link_dict[paper_id] = link
            print(f"📖 Đã nạp thành công 'Từ điển Link' với {len(link_dict)} mục.")
        except Exception as e:
            print(f"⚠️ Cảnh báo: Không thể đọc từ điển link {filepath}. Lỗi: {e}")
    else:
        print(f"⚠️ Không tìm thấy file {filepath}. Hệ thống sẽ dùng link tự tạo mặc định.")
    return link_dict


def is_valid_drawing(d, page_rect):
    """Lọc bỏ nhiễu: đường kẻ trang trí, khung trang, footer."""
    r = d['rect']
    if r.width < 10 or r.height < 10: return False
    if r.width > page_rect.width * 0.9 or r.height > page_rect.height * 0.9: return False
    return True

def extract_figures_high_res(pdf_path, source_url):
    doc = fitz.open(pdf_path)
    crawled_data = {}
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    extracted_images_info = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        page_dict = page.get_text("dict")
        
        captions = []
        for b in page_dict["blocks"]:
            if b['type'] == 0:
                text = " ".join([s["text"] for l in b["lines"] for s in l["spans"]]).strip()
                if re.match(r'^(Figure|Fig\.)\s*\d+', text, re.IGNORECASE):
                    captions.append({"bbox": fitz.Rect(b["bbox"]), "text": text})
        
        captions.sort(key=lambda x: x["bbox"].y0)

        page_images = page.get_image_info(hashes=True) 
        page_drawings = [d for d in page.get_drawings() if is_valid_drawing(d, page.rect)]

        for idx, cap in enumerate(captions):
            cap_bbox = cap["bbox"]
            upper_limit = captions[idx-1]["bbox"].y1 if idx > 0 else 0
            
            visual_elements = []

            for img in page_images:
                img_rect = fitz.Rect(img["bbox"])
                if upper_limit < img_rect.y1 <= cap_bbox.y0:
                    visual_elements.append(img_rect)
            
            for d in page_drawings:
                d_rect = d["rect"]
                if upper_limit < d_rect.y1 <= cap_bbox.y0:
                    visual_elements.append(d_rect)

            if visual_elements:
                union_rect = visual_elements[0]
                for r in visual_elements[1:]:
                    union_rect |= r 
                
                crop_rect = fitz.Rect(
                    max(0, union_rect.x0 - 5), 
                    max(upper_limit, union_rect.y0 - 5), 
                    min(page.rect.width, union_rect.x1 + 5), 
                    cap_bbox.y0 - 1
                )

                if crop_rect.height < 20: continue

                # Render ảnh chất lượng cao
                pix = page.get_pixmap(
                    matrix=fitz.Matrix(EXTRACT_DPI/72, EXTRACT_DPI/72), 
                    clip=crop_rect, 
                    alpha=False
                )

                img_filename = f"{base_name}_p{page_num+1}_fig{idx+1}.png"
                img_filepath = os.path.join(RAW_IMG_DIR, img_filename)
                pix.save(img_filepath)
                
                extracted_images_info.append({"name": img_filename, "size": f"{pix.width}x{pix.height}"})
                crawled_data[img_filename] = {
                    "caption": cap["text"],
                    "source_url": source_url,
                    "page": page_num + 1
                }

    if extracted_images_info:
        print(f"   ✅ Trích xuất thành công {len(extracted_images_info)} hình.")
    return crawled_data

def main():
    print(f"🚀 Khởi động Pipeline v4 (Scientific Paper Optimized)...")
    source_dictionary = load_source_links_dictionary(SOURCE_LINKS_FILE)
    existing_data = {}
    if os.path.exists(RAW_JSON_PATH):
        try:
            with open(RAW_JSON_PATH, "r", encoding="utf-8") as f:
                content = f.read()
                if content: existing_data = json.loads(content)
        except: pass

    pdf_files = [f for f in os.listdir(PDF_DIR) if f.lower().endswith('.pdf')]
    
    for idx, pdf_file in enumerate(pdf_files):
        paper_id = os.path.splitext(pdf_file)[0]
        print(f"\n[{idx+1}/{len(pdf_files)}] Đang quét: {pdf_file}")
        
        if paper_id in source_dictionary:
            source_url = source_dictionary[paper_id]
            print("   🔗 Dùng link từ Từ điển.")
        else:
            source_url = f"https://aclanthology.org/{paper_id}.pdf"
            print("   ⚠️ Không có trong Từ điển. Dùng link mặc định.")
        pdf_path = os.path.join(PDF_DIR, pdf_file)

        try:
            new_results = extract_figures_high_res(pdf_path, source_url)
            existing_data.update(new_results)
        except Exception as e:
            print(f"   ❌ Lỗi nghiêm trọng tại file này: {e}")
        
    with open(RAW_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=4)
        
    print(f"\n🎉 HOÀN TẤT! Đã lưu {len(existing_data)} mục vào Database.")

if __name__ == "__main__":
    main()