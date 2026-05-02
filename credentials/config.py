import os
from dotenv import load_dotenv

CREDENTIALS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CREDENTIALS_DIR)
load_dotenv()

env_path = os.path.join(ROOT_DIR, ".env")
load_dotenv(dotenv_path=env_path)

# --- CẤU HÌNH ĐƯỜNG DẪN ---
DATA_DIR = os.path.join(ROOT_DIR, "data")
IMAGES_DIR = os.path.join(DATA_DIR, "images")
DATASET_CSV_PATH = os.path.join(DATA_DIR, "Dataset.csv")
NOTES_MD_PATH = os.path.join(DATA_DIR, "Notes.md")

for folder in [DATA_DIR, IMAGES_DIR]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# --- KIỂM TRA BẢO MẬT GCP ---
GCP_KEY_PATH = os.getenv("GCP_KEY_PATH", os.path.join(CREDENTIALS_DIR, "cs-blindspot-agent.json"))
if not os.path.exists(GCP_KEY_PATH):
    GCP_KEY_PATH = os.path.join(CREDENTIALS_DIR, GCP_KEY_PATH)

if not os.path.exists(GCP_KEY_PATH):
    raise ValueError(f"LỖI: Không tìm thấy file Service Account JSON tại {GCP_KEY_PATH}")

# Thiết lập biến môi trường mặc định cho Google Cloud SDK
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GCP_KEY_PATH

# --- CẤU HÌNH GOOGLE CLOUD PROJECT & DOCUMENT AI ---
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID") 
GCP_LOCATION = os.getenv("GCP_LOCATION", "us") 
DOCAI_PROCESSOR_ID = os.getenv("DOCAI_PROCESSOR_ID") 

# --- CẤU HÌNH MODELS (Vertex AI Routing via LiteLLM) ---
SMART_AI_MODEL = "vertex_ai/gemini-2.5-pro" 
VICTIM_AI_MODEL = "vertex_ai/gemini-2.5-flash"
GPT5_MODEL = "openai/gpt-5"
CLAUDE_MODEL = "anthropic/claude-haiku-4-5-20251001" 

# --- CẤU HÌNH RETRY (Khớp với AI_NOTES.md) ---
MAX_RETRIES = 5
RETRY_MIN_WAIT = 1
RETRY_MAX_WAIT = 60 