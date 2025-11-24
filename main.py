import json
import os
import re
import tempfile
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

import fitz  # pymupdf
import pandas as pd
import requests
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pyngrok import ngrok
from together import Together

HOST = os.getenv("APP_HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8300"))
ENABLE_NGROK = os.getenv("ENABLE_NGROK", "").lower() in ("1", "true", "yes", "on")
TYPHOON_API_KEY = os.getenv("TYPHOON_API_KEY", "sk-isgEKmDdHNQGhjq0R7GVTKAUoOSRr0qOAwoJObIXs5w5CBNL")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY", "c0186a5f58d5dcf0f0528503bd34777f4f70fc36093d03def2738a94534cd775")
MODEL = os.getenv("TOGETHER_MODEL", "google/gemma-3n-E4B-it")
DEFAULT_ORIGINS = [
    "http://127.0.0.1:8000",
    "http://localhost:8000",
]
ENV_ORIGINS = os.getenv("ALLOWED_ORIGINS", "")
ORIGINS = [o.strip() for o in ENV_ORIGINS.split(",") if o.strip()] or DEFAULT_ORIGINS

app = FastAPI(title="Document Comparison API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = "google/gemma-3n-E4B-it"

@app.get("/")
async def root():
    """API Info"""
    return {
        "name": "Document Comparison API",
        "version": "1.0.0",
        "endpoints": {
            "/compare-documents": "POST - เปรียบเทียบเอกสาร 2 ฉบับ",
        }
    }

def pdf_to_images(pdf_path: str, output_dir: Path, zoom: float = 5.0) -> list:
    """แปลง PDF เป็นรูปภาพ PNG"""
    output_dir.mkdir(exist_ok=True)
    doc = fitz.open(pdf_path)
    mat = fitz.Matrix(zoom, zoom)
    
    image_paths = []
    for i, page in enumerate(doc):
        pix = page.get_pixmap(matrix=mat)
        out_path = output_dir / f"page_{i+1:03}.png"
        pix.save(out_path)
        image_paths.append(str(out_path))
        print(f"Saved: {out_path}")
    
    doc.close()
    return image_paths

def tokenize_text(text: str) -> list:
    """
    แยกข้อความเป็น tokens (คำ, ตัวเลข, เครื่องหมาย)
    รักษาทั้งคำไทยและอังกฤษ
    """
    tokens = []
    current_token = ""
    
    for char in text:
        # เว้นวรรค / ขึ้นบรรทัดใหม่ = จบคำ
        if char in ' \t\n\r':
            if current_token:
                tokens.append(current_token)
                current_token = ""
            # ไม่เก็บ whitespace
        # เครื่องหมายวรรคตอนต่าง ๆ แยกเป็น token เอง
        elif char in '.,;:!?()[]{}":\'/-=+*&%$#@':
            if current_token:
                tokens.append(current_token)
                current_token = ""
            tokens.append(char)
        else:
            current_token += char
    
    if current_token:
        tokens.append(current_token)
    
    return tokens


def ocr_image_typhoon(image_path: str, api_key: str, 
                      model: str = "typhoon-ocr",
                      task_type: str = "v1.5",
                      max_tokens: int = 16000) -> Optional[str]:
    """เรียกใช้ Typhoon OCR API"""
    url = "https://api.opentyphoon.ai/v1/ocr"
    
    try:
        with open(image_path, 'rb') as file:
            files = {'file': file}
            data = {
                'model': model,
                'task_type': task_type,
                'max_tokens': str(max_tokens),
                'temperature': '0.1',
                'top_p': '0.6',
                'repetition_penalty': '1.2',
            }
            headers = {'Authorization': f'Bearer {api_key}'}
            
            response = requests.post(url, files=files, data=data, headers=headers)
            
            if response.status_code != 200:
                print(f"[ERROR] {image_path} -> HTTP {response.status_code}")
                print(response.text)
                return None
            
            result = response.json()
            
            extracted_texts = []
            for page_result in result.get('results', []):
                if page_result.get('success') and page_result.get('message'):
                    content = page_result['message']['choices'][0]['message']['content']
                    try:
                        parsed = json.loads(content)
                        text = parsed.get('natural_text', content)
                    except json.JSONDecodeError:
                        text = content
                    extracted_texts.append(text)
                else:
                    print(f"[ERROR] {image_path} -> {page_result.get('error', 'Unknown')}")
            
            return "\n".join(extracted_texts)
    
    except Exception as e:
        print(f"[ERROR] {image_path}: {e}")
        return None


def normalize_text(text: str) -> str:
    """ทำความสะอาดและ normalize ข้อความ"""
    text = str(text) if text else ""
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n+", "\n", text)
    return text.strip()


def strip_markdown(md: str) -> str:
    """ลบ markdown syntax ออก"""
    md = re.sub(r"^#{1,6}\s*", "", md, flags=re.MULTILINE)
    md = re.sub(r"(\*{1,2}|_{1,2})(.+?)\1", r"\2", md)
    md = re.sub(r"^\s*[-*+]\s+", "", md, flags=re.MULTILINE)
    return md


def calculate_similarity(text1: str, text2: str) -> float:
    """คำนวณความคล้ายคลึงระหว่างข้อความ"""
    t1 = normalize_text(text1)
    t2 = normalize_text(text2)
    return SequenceMatcher(None, t1, t2).ratio()


def extract_key_values(text: str) -> dict:
    """
    แยก key-value pairs จากข้อความ
    รองรับรูปแบบ:
    - "ชื่อ: ค่า"
    - "ชื่อ ค่า" (ถ้ามี pattern ชัดเจน)
    - "ชื่อ=ค่า"
    """
    key_values = {}
    lines = text.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # รูปแบบ 1: key: value
        if ':' in line:
            parts = line.split(':', 1)
            key = parts[0].strip()
            value = parts[1].strip() if len(parts) > 1 else ""
            key_values[key] = value
        
        # รูปแบบ 2: key = value
        elif '=' in line:
            parts = line.split('=', 1)
            key = parts[0].strip()
            value = parts[1].strip() if len(parts) > 1 else ""
            key_values[key] = value
        
        # รูปแบบ 3: พยายามหา pattern (เช่น "รวมเบี้ย 2000 บาท")
        else:
            # หาคำที่เป็นตัวเลข + หน่วย
            import re
            # ลองหา pattern แบบ "ชื่อฟิลด์ ตัวเลข หน่วย"
            match = re.search(r'(.+?)\s+(\d[\d,\.]+)\s*(\S*)', line)
            if match:
                key = match.group(1).strip()
                value = f"{match.group(2)} {match.group(3)}".strip()
                key_values[key] = value
            else:
                # ถ้าไม่เจอ pattern ก็เก็บทั้งบรรทัด
                key_values[line] = line
    
    return key_values


def semantic_diff(text1: str, text2: str) -> str:
    """
    เปรียบเทียบแบบ Semantic (key-value based)
    """
    text1 = text1 or ""
    text2 = text2 or ""
    
    if text1 == text2:
        return "ไม่มีความแตกต่าง"
    
    kv1 = extract_key_values(text1)
    kv2 = extract_key_values(text2)
    
    all_keys = set(kv1.keys()) | set(kv2.keys())
    
    diffs = []
    diffs.append("=" * 80)
    diffs.append("สรุปความแตกต่าง (เทียบแบบ Key-Value)")
    diffs.append("=" * 80)
    
    added = []
    removed = []
    changed = []
    
    for key in sorted(all_keys):
        val1 = kv1.get(key)
        val2 = kv2.get(key)
        
        if val1 is None and val2 is not None:
            # เพิ่มใหม่
            added.append((key, val2))
        elif val1 is not None and val2 is None:
            # ลบออก
            removed.append((key, val1))
        elif val1 != val2:
            # เปลี่ยนแปลง
            changed.append((key, val1, val2))
    
    # แสดงผลที่เปลี่ยนแปลง
    if changed:
        diffs.append("\n[เปลี่ยนแปลง] ค่าที่แตกต่างกัน:")
        diffs.append("-" * 80)
        for key, old_val, new_val in changed:
            diffs.append(f"\n  📝 {key}")
            diffs.append(f"     เดิม: {old_val}")
            diffs.append(f"     ใหม่: {new_val}")
            
            # ถ้าเป็นตัวเลข ลองคำนวณส่วนต่าง
            import re
            old_num = re.search(r'[\d,]+\.?\d*', old_val)
            new_num = re.search(r'[\d,]+\.?\d*', new_val)
            if old_num and new_num:
                try:
                    old_n = float(old_num.group().replace(',', ''))
                    new_n = float(new_num.group().replace(',', ''))
                    diff = new_n - old_n
                    percent = ((new_n - old_n) / old_n * 100) if old_n != 0 else 0
                    diffs.append(f"     ส่วนต่าง: {diff:+,.2f} ({percent:+.2f}%)")
                except:
                    pass
    
    # แสดงผลที่เพิ่มใหม่
    if added:
        diffs.append("\n[เพิ่มใหม่] ข้อมูลที่มีในเอกสาร 2 แต่ไม่มีในเอกสาร 1:")
        diffs.append("-" * 80)
        for key, val in added:
            diffs.append(f"\n  ➕ {key}")
            diffs.append(f"     ค่า: {val}")
    
    # แสดงผลที่ถูกลบ
    if removed:
        diffs.append("\n[ลบออก] ข้อมูลที่มีในเอกสาร 1 แต่ไม่มีในเอกสาร 2:")
        diffs.append("-" * 80)
        for key, val in removed:
            diffs.append(f"\n  ➖ {key}")
            diffs.append(f"     ค่า: {val}")
    
    if not changed and not added and not removed:
        return "ไม่มีความแตกต่าง"
    
    diffs.append("")
    diffs.append("=" * 80)
    diffs.append(f"สรุป: เปลี่ยนแปลง {len(changed)} รายการ | เพิ่ม {len(added)} รายการ | ลบ {len(removed)} รายการ")
    diffs.append("=" * 80)
    
    return "\n".join(diffs)


def calculate_semantic_diff_stats(text1: str, text2: str) -> dict:
    """
    คำนวณสถิติความแตกต่างแบบ semantic
    """
    text1 = text1 or ""
    text2 = text2 or ""
    
    kv1 = extract_key_values(text1)
    kv2 = extract_key_values(text2)
    
    all_keys = set(kv1.keys()) | set(kv2.keys())
    
    stats = {
        'keys_added': 0,
        'keys_removed': 0,
        'keys_changed': 0,
        'total_keys_pdf1': len(kv1),
        'total_keys_pdf2': len(kv2),
        'total_diff_keys': 0
    }
    
    for key in all_keys:
        val1 = kv1.get(key)
        val2 = kv2.get(key)
        
        if val1 is None and val2 is not None:
            stats['keys_added'] += 1
        elif val1 is not None and val2 is None:
            stats['keys_removed'] += 1
        elif val1 != val2:
            stats['keys_changed'] += 1
    
    stats['total_diff_keys'] = stats['keys_added'] + stats['keys_removed'] + stats['keys_changed']
    
    return stats


def build_combined_pages(texts: list[str]) -> str:
    """รวมข้อความแต่ละหน้า พร้อม tag หน้าสำหรับการเปรียบเทียบรวม"""
    return "\n\n".join([f"=== Page {i + 1} ===\n{text}" for i, text in enumerate(texts)])


def build_comparison_row(text1_raw: str, text2_raw: str, page_label) -> dict:
    """คำนวณชุดสถิติและรายละเอียดความต่างสำหรับหนึ่งหน้า (หรือรวมทุกหน้า)"""
    t1_norm = normalize_text(strip_markdown(text1_raw))
    t2_norm = normalize_text(strip_markdown(text2_raw))
    
    sim = calculate_similarity(t1_norm, t2_norm)
    semantic_stats = calculate_semantic_diff_stats(t1_norm, t2_norm)
    semantic_diff_text = semantic_diff(t1_norm, t2_norm)
    word_stats = calculate_word_diff_stats(t1_norm, t2_norm)
    word_diff_text = word_based_diff(t1_norm, t2_norm)
    
    return {
        "page": page_label,
        "pdf1_text_raw": text1_raw,
        "pdf2_text_raw": text2_raw,
        "pdf1_text_normalized": t1_norm,
        "pdf2_text_normalized": t2_norm,
        "similarity": sim,
        "similarity_percent": round(sim * 100, 2),
        "is_equal": t1_norm == t2_norm,
        "keys_changed": semantic_stats['keys_changed'],
        "keys_added": semantic_stats['keys_added'],
        "keys_removed": semantic_stats['keys_removed'],
        "total_keys_pdf1": semantic_stats['total_keys_pdf1'],
        "total_keys_pdf2": semantic_stats['total_keys_pdf2'],
        "total_diff_keys": semantic_stats['total_diff_keys'],
        "semantic_diff_details": semantic_diff_text,
        "words_added": word_stats['words_added'],
        "words_removed": word_stats['words_removed'],
        "words_changed": word_stats['words_changed'],
        "word_diff_details": word_diff_text,
    }


def word_based_diff(text1: str, text2: str) -> str:
    """
    เปรียบเทียบแบบ word-by-word และแสดงความแตกต่างทั้งหมด
    """
    text1 = text1 or ""
    text2 = text2 or ""
    
    if text1 == text2:
        return "ไม่มีความแตกต่าง"
    
    tokens1 = tokenize_text(text1)
    tokens2 = tokenize_text(text2)
    
    sm = SequenceMatcher(None, tokens1, tokens2)
    diffs = []
    
    diffs.append("สรุปความแตกต่าง (เทียบเป็นคำ)")
    
    diff_count = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        
        diff_count += 1
        
        if tag == "insert":
            added_text = " ".join(tokens2[j1:j2])
            diffs.append(f"\n[จุดที่ {diff_count}] เพิ่ม {j2-j1} คำ:")
            diffs.append("-" * 80)
            diffs.append(f"+ {added_text}")
        
        elif tag == "delete":
            removed_text = " ".join(tokens1[i1:i2])
            diffs.append(f"\n[จุดที่ {diff_count}] ลบ {i2-i1} คำ:")
            diffs.append("-" * 80)
            diffs.append(f"- {removed_text}")
        
        elif tag == "replace":
            old_text = " ".join(tokens1[i1:i2])
            new_text = " ".join(tokens2[j1:j2])
            diffs.append(f"\n[จุดที่ {diff_count}] แก้ไข:")
            diffs.append("-" * 80)
            diffs.append(f"เดิม: {old_text}")
            diffs.append(f"ใหม่: {new_text}")
            
            # แสดง token ที่แตกต่างเฉพาะส่วนที่เปลี่ยน
            if i2 - i1 <= 20 and j2 - j1 <= 20:  # ถ้าไม่ยาวเกินไป
                diffs.append("")
                diffs.append("รายละเอียด:")
                
                # หาความแตกต่างภายใน tokens
                token_sm = SequenceMatcher(None, tokens1[i1:i2], tokens2[j1:j2])
                for t_tag, t_i1, t_i2, t_j1, t_j2 in token_sm.get_opcodes():
                    if t_tag == "equal":
                        continue
                    if t_tag == "insert":
                        diffs.append(f"  + เพิ่ม: {' '.join(tokens2[j1+t_j1:j1+t_j2])}")
                    elif t_tag == "delete":
                        diffs.append(f"  - ลบ: {' '.join(tokens1[i1+t_i1:i1+t_i2])}")
                    elif t_tag == "replace":
                        diffs.append(f"  ~ เปลี่ยน: '{' '.join(tokens1[i1+t_i1:i1+t_i2])}' → '{' '.join(tokens2[j1+t_j1:j1+t_j2])}'")
    
    if diff_count == 0:
        return "ไม่มีความแตกต่าง"
    
    diffs.append("")
    diffs.append("=" * 80)
    diffs.append(f"รวม {diff_count} จุดที่แตกต่างกัน")
    diffs.append("=" * 80)
    
    return "\n".join(diffs)


def calculate_word_diff_stats(text1: str, text2: str) -> dict:
    """
    คำนวณสถิติความแตกต่างแบบ word-based
    """
    text1 = text1 or ""
    text2 = text2 or ""
    
    tokens1 = tokenize_text(text1)
    tokens2 = tokenize_text(text2)
    
    sm = SequenceMatcher(None, tokens1, tokens2)
    
    stats = {
        'words_added': 0,
        'words_removed': 0,
        'words_changed': 0,
        'chars_added': 0,
        'chars_removed': 0,
        'total_diff_blocks': 0,
        'total_words_pdf1': len(tokens1),
        'total_words_pdf2': len(tokens2)
    }
    
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        
        stats['total_diff_blocks'] += 1
        
        if tag == "insert":
            stats['words_added'] += (j2 - j1)
            stats['chars_added'] += sum(len(t) for t in tokens2[j1:j2])
        elif tag == "delete":
            stats['words_removed'] += (i2 - i1)
            stats['chars_removed'] += sum(len(t) for t in tokens1[i1:i2])
        elif tag == "replace":
            stats['words_changed'] += max(i2 - i1, j2 - j1)
            stats['chars_removed'] += sum(len(t) for t in tokens1[i1:i2])
            stats['chars_added'] += sum(len(t) for t in tokens2[j1:j2])
    
    return stats


def compare_pdfs_with_ocr(pdf_path_1: str, pdf_path_2: str, 
                          api_key: str, output_csv: str = "comparison_result.csv",
                          temp_dir: str = "temp_images") -> pd.DataFrame:
    """
    เปรียบเทียบ PDF 2 ไฟล์โดยใช้ Typhoon OCR API
    
    Args:
        pdf_path_1: path ของ PDF ต้นฉบับ
        pdf_path_2: path ของ PDF คู่เทียบ (master)
        api_key: Typhoon API key
        output_csv: ชื่อไฟล์ CSV ผลลัพธ์
        temp_dir: โฟลเดอร์สำหรับเก็บรูปภาพชั่วคราว
    
    Returns:
        DataFrame ที่มีผลการเปรียบเทียบ
    """
    
    # สร้างโฟลเดอร์ชั่วคราว
    temp_path = Path(temp_dir)
    dir1 = temp_path / "pdf1"
    dir2 = temp_path / "pdf2"
    dir1.mkdir(parents=True, exist_ok=True)
    dir2.mkdir(parents=True, exist_ok=True)
    
    # แปลง PDF เป็นรูปภาพ
    print("\n[1/4] แปลง PDF 1 เป็นรูปภาพ...")
    images1 = pdf_to_images(pdf_path_1, dir1)
    
    print("\n[2/4] แปลง PDF 2 เป็นรูปภาพ...")
    images2 = pdf_to_images(pdf_path_2, dir2)
    
    # OCR ด้วย Typhoon API
    print("\n[3/4] ทำ OCR กับ PDF 1...")
    texts1 = []
    for img in images1:
        text = ocr_image_typhoon(img, api_key)
        texts1.append(text or "")
    
    print("\n[4/4] ทำ OCR กับ PDF 2...")
    texts2 = []
    for img in images2:
        text = ocr_image_typhoon(img, api_key)
        texts2.append(text or "")
    
    print("\n[5/5] เปรียบเทียบผลลัพธ์...")

    all_text1 = build_combined_pages(texts1)
    all_text2 = build_combined_pages(texts2)

    rows = [build_comparison_row(all_text1, all_text2, "ALL")]
    result_df = pd.DataFrame(rows)
    result_df.to_csv("all_pages " + output_csv, index=False, encoding="utf-8-sig")
    
    # เปรียบเทียบทีละหน้า
    print("\n[6/6] เปรียบเทียบผลลัพธ์...")
    n = max(len(texts1), len(texts2))
    rows = []
    
    for i in range(n):
        t1_raw = texts1[i] if i < len(texts1) else ""
        t2_raw = texts2[i] if i < len(texts2) else ""
        
        rows.append(build_comparison_row(t1_raw, t2_raw, i + 1))
    
    # สร้าง DataFrame และบันทึก
    result_df = pd.DataFrame(rows)
    result_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    
    return result_df

def create_comparison_prompt(text1: str, text2: str) -> str:
    """
    สร้าง prompt สำหรับเปรียบเทียบกรมธรรม์ 2 ฉบับ โดยเน้นหาความแตกต่างที่มีนัยสำคัญ
    ตรวจสอบความถูกต้องของข้อมูล (Data Integrity) และความหมายโดยรวม (Semantic Equivalence)
    """
    prompt = f"""
บทบาท: คุณคือผู้เชี่ยวชาญด้านการตรวจสอบเอกสารประกันภัย (Insurance Policy Auditor) ที่มีความละเอียดสูงสุด

งานของคุณ: เปรียบเทียบข้อมูลจาก "เอกสารฉบับที่ 1" และ "เอกสารฉบับที่ 2" เพื่อระบุความแตกต่าง **ทุกอย่างที่แตกต่างกัน** แม้จะเป็นเรื่องเล็กน้อย

**ข้อมูลนำเข้า:**

--- เอกสารฉบับที่ 1 (ต้นฉบับ) ---
{text1}

--- เอกสารฉบับที่ 2 (ฉบับเปรียบเทียบ) ---
{text2}

--------------------------------------------------

**หลักการสำคัญ - อ่านให้ละเอียด:**

1. **รายงานเฉพาะสิ่งที่แตกต่าง:** ห้ามรายงานฟิลด์หรือข้อมูลที่เหมือนกันทุกประการ ให้มุ่งเน้นเฉพาะสิ่งที่มีความแตกต่าง
2. **รายงานทุกความแตกต่าง:** ถึงแม้จะเป็นเรื่องเล็กน้อย เช่น การเว้นวรรค ตัวอักษรผิด 1 ตัว หรือตัวเลขต่างกัน 0.01 บาท ก็ต้องรายงาน
3. **ให้รายละเอียดชัดเจน:** บอกอย่างชัดเจนว่าอะไรต่างกันยังไง เช่น "เดิม: 1,950 บาท | ใหม่: 2,000 บาท" แทนที่จะบอกแค่ว่า "เบี้ยประกันต่างกัน"

**คำสั่งการวิเคราะห์:**

ให้ตอบกลับเป็น JSON Format เท่านั้น โดยใช้โครงสร้างและตรรกะดังนี้:

1. **โครงสร้าง JSON ผลลัพธ์:**
{{
  "summary": {{
    "is_identical": false,  // true ถ้าข้อมูลสำคัญเหมือนกันหมด (แม้รูปแบบต่าง)
    "total_changes": 0,
    "critical_changes": 0,
    "high_changes": 0,
    "medium_changes": 0,
    "low_changes": 0
  }},
  "changes": [
    {{
      "field_name": "ชื่อฟิลด์ที่พบปัญหา",
      "field_type": "sum_insured|premium|name|date|condition|coverage|policy_id|other",
      "old_value": "ค่าในเอกสาร 1",
      "new_value": "ค่าในเอกสาร 2",
      "change_type": "modified|added|removed",
      "severity": "CRITICAL|HIGH|MEDIUM|LOW",
      "description": "อธิบายความต่างสั้นๆ",
      "impact": "ผลกระทบที่อาจเกิดขึ้น (ถ้ามี)",
      "is_semantic_equivalent": false // true ถ้าความหมายเหมือนกันแต่เขียนต่าง
    }}
  ],
  "semantic_notes": [
    "ข้อสังเกตเพิ่มเติมเกี่ยวกับความไม่สอดคล้องภายในเอกสารเดียวกัน (ถ้ามี)"
  ]
}}

2. **เกณฑ์การตัดสินความรุนแรง (Severity Logic):**
   - **CRITICAL:** เลขที่กรมธรรม์ (Policy No.), ทุนประกันรวม, เบี้ยประกันรวม, ชื่อผู้เอาประกัน, เลขบัตรประชาชน, วันที่เริ่ม-สิ้นสุดความคุ้มครอง
   - **HIGH:** ชื่อผู้รับผลประโยชน์, รายการความคุ้มครองที่หายไป (Missing Coverage), ชื่อบริษัทประกันภัย (สะกดผิด), เงื่อนไขหลัก
   - **MEDIUM:** รายละเอียดที่อยู่, ข้อยกเว้นย่อย, วิธีการชำระเงิน, ชื่อตัวแทน/นายหน้า
   - **LOW:** การจัดรูปแบบ, การเว้นวรรค, เบอร์โทรศัพท์, คำนำหน้าชื่อ (นาย/นาง/คุณ)

3. **หลักการวิเคราะห์เชิงลึก (Deep Analysis Rules):**

   A. **การตรวจสอบตัวเลขและการคำนวณ (Math & Values Check):**
   - **สำคัญมาก (Calculation Rule):** หากเอกสารหนึ่งแสดง "ยอดรวมสุทธิ" (Total) แต่อีกเอกสารแสดง "ยอดแยกย่อย" (Breakdown: เบี้ยประกัน + อากร + ภาษี)
   - **Action:** ให้คุณทำการบวกตัวเลขย่อยเหล่านั้นก่อนเปรียบเทียบ
   - **Logic:** ถ้า (เบี้ยประกัน + อากรแสตมป์ + ภาษี) ในเอกสารหนึ่ง **เท่ากับ** ยอดรวมในอีกเอกสารหนึ่ง (เช่น 1,950 + 50 = 2,000)
   - **Result:** ให้ถือว่า `is_semantic_equivalent: true` (ความหมายเหมือนกัน) และอธิบายใน description ว่า "ยอดรวมเกิดจากการรวมเบี้ยและอากร"

   B. **การตรวจสอบตัวอักษรและการสะกด (Spelling & Typos):**
      - จับตาดูการสะกดชื่อเฉพาะ: เช่น "ประจำค่าย" vs "ประจำคาย" -> ถือเป็น `change` (Severity: HIGH)
      - จับตาดูชื่อบริษัท: เช่น "เอ็ม เอส ไอ จี" vs "เอ็น เอส ไอ จี" -> ถือเป็น `change` (Severity: HIGH/MEDIUM)

   C. **การตรวจสอบรหัสกรมธรรม์ (ID Verification):**
      - ให้ระวัง Prefix/Suffix: "24-xxxx" vs "25-xxxx" -> ถือเป็น `change` (Severity: CRITICAL) เพราะบ่งบอกว่าเป็นคนละปีสัญญา

   D. **การตรวจสอบข้อมูลที่สูญหาย (Missing Data):**
      - หากเอกสาร 1 มีรายการ "ความคุ้มครองการถูกฆาตกรรม" แต่เอกสาร 2 ไม่มี -> ต้องระบุเป็น `change_type: "removed"` และ `severity: "HIGH"`

   E. **การตัดสิ่งรบกวน (Noise Reduction):**
      - ไม่นับการเว้นวรรค (Space), การขึ้นบรรทัดใหม่, หรือสัญลักษณ์พิเศษ (- /) ที่ไม่มีผลต่อความหมาย เป็นความเปลี่ยนแปลง
      
   F. **การจัดการข้อมูลข้ามหน้า (Cross-Page Resolution):**
   - หากเอกสาร 1 ระบุ อาชีพ "พนักงานบริษัท" (หน้า 1)
   - แต่เอกสาร 2 ระบุ อาชีพ "Occupation" (หน้า 2 - ผิดปกติ) -> ให้ค้นหาต่อจนเจอ "พนักงานบริษัท" (หน้า 3)
   - **สรุปผล:** ต้องรายงานว่าเอกสาร 2 หน้า 2 ข้อมูลผิด/เป็น Placeholder แต่ไปเจอข้อมูลจริงที่หน้า 3

   

4. **ตัวอย่างการตอบ (Example):**
   [
     {{
       "field_name": "เลขที่กรมธรรม์",
       "field_type": "policy_id",
       "old_value": "24-51062678",
       "new_value": "25-51062678",
       "change_type": "modified",
       "severity": "CRITICAL",
       "description": "เลขปีหน้ากรมธรรม์เปลี่ยนจาก 24 เป็น 25",
       "is_semantic_equivalent": false
     }},
     {{
       "field_name": "ชื่อสกุลผู้เอาประกัน",
       "field_type": "name",
       "old_value": "ประจำค่าย",
       "new_value": "ประจำคาย",
       "change_type": "modified",
       "severity": "HIGH",
       "description": "ตัวสะกดนามสกุลไม่ตรงกัน (ย vs ย์)",
       "is_semantic_equivalent": false
     }},
  {{
    "field_name": "ความคุ้มครองกรณีถูกฆาตกรรม",
    "field_type": "coverage",
    "doc1_value": "900,000.00 บาท",
    "doc1_page": 2,
    "doc2_value": "900,000.00 บาท",
    "doc2_page": 3,
    "change_type": "relocated",
    "severity": "LOW",
    "description": "รายการนี้ย้ายจากหน้า 2 ในเอกสารเก่า ไปอยู่ที่หน้า 3 ในเอกสารใหม่",
    "is_semantic_equivalent": true
  }}
   ]

**ข้อควรระวัง:** จงตอบเป็น JSON Object ดิบๆ เท่านั้น ไม่ต้องใส่ Markdown code block (```json) ครอบ และไม่ต้องมีข้อความเกริ่นนำ
"""
    return prompt

def compare_with_together(text1: str, text2: str, api_key: str, 
                         model: str = MODEL) -> dict:
    """
    ส่งข้อความไปให้ Together AI เปรียบเทียบ
    
    Args:
        text1: ข้อความจากเอกสารฉบับที่ 1
        text2: ข้อความจากเอกสารฉบับที่ 2
        api_key: Together AI API key
        model: โมเดลที่จะใช้
        
    Returns:
        dict: ผลการเปรียบเทียบ
    """
    client = Together(api_key=api_key)
    
    prompt = create_comparison_prompt(text1, text2)
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": prompt
            }],
            temperature=0.1,
            max_tokens=8000,  # เพิ่มจาก 4000 เป็น 8000 เพื่อให้ AI ตอบได้ครบถ้วนมากขึ้น
        )
        
        result_text = response.choices[0].message.content
        
        # Parse JSON
        result_text = result_text.strip()
        if result_text.startswith("```json"):
            result_text = result_text[7:]
        if result_text.startswith("```"):
            result_text = result_text[3:]
        if result_text.endswith("```"):
            result_text = result_text[:-3]
        
        result_json = json.loads(result_text.strip())
        return result_json
        
    except json.JSONDecodeError as e:
        return {
            "error": "Failed to parse JSON",
            "raw_response": result_text
        }
    except Exception as e:
        return {
            "error": str(e)
        }


def compare_and_save(input_df: pd.DataFrame, output_csv: str, api_key: str):
    """
    เปรียบเทียบข้อความจาก DataFrame และบันทึกเป็น CSV
    
    Args:
        input_df: DataFrame ที่มี columns 'pdf1_text_raw' และ 'pdf2_text_raw'
        output_csv: ชื่อไฟล์ CSV output
        api_key: Together AI API key
    """
    results = []
    
    print(f"กำลังเปรียบเทียบ {len(input_df)} หน้า...")
    print("=" * 60)
    
    for idx in input_df.index:
        print(f"กำลังประมวลผล หน้า {idx + 1}/{len(input_df)}...", end=" ")
        
        text1 = str(input_df.loc[idx, 'pdf1_text_raw'])
        text2 = str(input_df.loc[idx, 'pdf2_text_raw'])
        
        # เรียก Together AI
        comparison = compare_with_together(text1, text2, api_key)
        
        # แยกข้อมูลออกมา
        if 'error' in comparison:
            row = {
                'page': idx + 1,
                'pdf1_text': text1,
                'pdf2_text': text2,
                'is_identical': None,
                'total_changes': -1,
                'critical_changes': -1,
                'high_changes': -1,
                'medium_changes': -1,
                'low_changes': -1,
                'changes_detail': str(comparison),
                'semantic_notes': '',
                'error': comparison.get('error', 'Unknown error')
            }
            print("❌ Error")
        else:
            summary = comparison.get('summary', {})
            changes = comparison.get('changes', [])
            notes = comparison.get('semantic_notes', [])
            
            row = {
                'page': idx + 1,
                'pdf1_text': text1,
                'pdf2_text': text2,
                'is_identical': summary.get('is_identical', False),
                'total_changes': summary.get('total_changes', 0),
                'critical_changes': summary.get('critical_changes', 0),
                'high_changes': summary.get('high_changes', 0),
                'medium_changes': summary.get('medium_changes', 0),
                'low_changes': summary.get('low_changes', 0),
                'changes_detail': json.dumps(changes, ensure_ascii=False, indent=2),
                'semantic_notes': '\n'.join(notes),
                'error': ''
            }
            
            status = "✅ เหมือนกัน" if summary.get('is_identical') else f"⚠️ แตกต่าง {summary.get('total_changes')} จุด"
            print(status)
        
        results.append(row)
    
    # สร้าง DataFrame และบันทึก
    result_df = pd.DataFrame(results)
    result_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    
    return result_df
def extract_and_parse_changes(detail_text: str) -> list:
    """แปลง changes_detail string เป็น list of dict"""
    if not detail_text or pd.isna(detail_text):
        return []
    
    # Convert to string
    text = str(detail_text)
    
    # ลบ escape characters
    text = text.replace('\\n', '\n').replace('\\"', '"')
    
    # หา JSON array pattern
    json_match = re.search(r'\[[\s\S]*\]', text)
    
    if json_match:
        json_str = json_match.group(0)
        try:
            # Parse JSON
            changes = json.loads(json_str)
            return changes
        except json.JSONDecodeError as e:
            print(f"JSON Parse Error: {e}")
            return []
    
    return []

def transform_changes_detail(df: pd.DataFrame, column: str = 'changes_detail') -> pd.DataFrame:
    """แปลง changes_detail column เป็น structured data"""
    df = df.copy()
    
    # Parse JSON
    df['changes_array'] = df[column].apply(extract_and_parse_changes)
    
    # Extract key metrics
    df['total_changes'] = df['changes_array'].apply(len)
    df['critical_count'] = df['changes_array'].apply(
        lambda x: sum(1 for item in x if item.get('severity') == 'CRITICAL')
    )
    df['high_count'] = df['changes_array'].apply(
        lambda x: sum(1 for item in x if item.get('severity') == 'HIGH')
    )
    df['medium_count'] = df['changes_array'].apply(
        lambda x: sum(1 for item in x if item.get('severity') == 'MEDIUM')
    )
    df['low_count'] = df['changes_array'].apply(
        lambda x: sum(1 for item in x if item.get('severity') == 'LOW')
    )
    
    # Extract semantic equivalent count
    df['semantic_equivalent_count'] = df['changes_array'].apply(
        lambda x: sum(1 for item in x if item.get('is_semantic_equivalent') == True)
    )
    
    return df

@app.post("/compare-documents")
async def compare_documents(
    document1: UploadFile = File(..., description="เอกสารฉบับที่ 1"),
    document2: UploadFile = File(..., description="เอกสารฉบับที่ 2")
):
    
    if not document1.filename.endswith('.pdf') or not document2.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="กรุณาอัพโหลดไฟล์ PDF เท่านั้น")
        
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        pdf1_path = temp_path / "document1.pdf"
        pdf2_path = temp_path / "document2.pdf"

        with open(pdf1_path, "wb") as f:
            f.write(await document1.read())
            
        with open(pdf2_path, "wb") as f:
            f.write(await document2.read())
            
        try:
            # สร้างโฟลเดอร์สำหรับรูปภาพ
            img_dir1 = temp_path / "images1"
            img_dir2 = temp_path / "images2"
            
            img_dir1.mkdir(parents=True, exist_ok=True)
            img_dir2.mkdir(parents=True, exist_ok=True)
             
            print("\n[1/4] แปลง PDF 1 เป็นรูปภาพ...")
            images1 = pdf_to_images(pdf1_path, img_dir1)
            
            print("\n[2/4] แปลง PDF 2 เป็นรูปภาพ...")
            images2 = pdf_to_images(pdf2_path, img_dir2)

            print("\n[3/4] ทำ OCR กับ PDF 1...")
            texts1 = []
            for img in images1:
                text = ocr_image_typhoon(img, TYPHOON_API_KEY)
                texts1.append(text or "")
            
            print("\n[4/4] ทำ OCR กับ PDF 2...")
            texts2 = []
            for img in images2:
                text = ocr_image_typhoon(img, TYPHOON_API_KEY)
                texts2.append(text or "")
            
            print("\n[5/5] เปรียบเทียบผลลัพธ์...")
        
            all_text1 = build_combined_pages(texts1)
            all_text2 = build_combined_pages(texts2)
            
            rows = [build_comparison_row(all_text1, all_text2, "ALL")]
            input_df = pd.DataFrame(rows)
            
            results = []
            print(f"กำลังเปรียบเทียบ {len(input_df)} หน้า...")
            for idx in input_df.index:
                print(f"กำลังประมวลผล หน้า {idx + 1}/{len(input_df)}...", end=" ")
                text1 = str(input_df.loc[idx, 'pdf1_text_raw'])
                text2 = str(input_df.loc[idx, 'pdf2_text_raw'])
                
                # เรียก Together AI
                comparison = compare_with_together(text1, text2, TOGETHER_API_KEY)
            
                # แยกข้อมูลออกมา
                if 'error' in comparison:
                    row = {
                        'page': idx + 1,
                        'pdf1_text': text1,
                        'pdf2_text': text2,
                        'is_identical': None,
                        'total_changes': -1,
                        'critical_changes': -1,
                        'high_changes': -1,
                        'medium_changes': -1,
                        'low_changes': -1,
                        'changes_detail': str(comparison),
                        'semantic_notes': '',
                        'error': comparison.get('error', 'Unknown error')
                    }
                    print("❌ Error")
                else:
                    summary = comparison.get('summary', {})
                    changes = comparison.get('changes', [])
                    notes = comparison.get('semantic_notes', [])
                    
                    row = {
                        'page': idx + 1,
                        'pdf1_text': text1,
                        'pdf2_text': text2,
                        'is_identical': summary.get('is_identical', False),
                        'total_changes': summary.get('total_changes', 0),
                        'critical_changes': summary.get('critical_changes', 0),
                        'high_changes': summary.get('high_changes', 0),
                        'medium_changes': summary.get('medium_changes', 0),
                        'low_changes': summary.get('low_changes', 0),
                        'changes_detail': json.dumps(changes, ensure_ascii=False, indent=2),
                        'semantic_notes': '\n'.join(notes),
                        'error': ''
                    }
                    
                    status = "✅ เหมือนกัน" if summary.get('is_identical') else f"⚠️ แตกต่าง {summary.get('total_changes')} จุด"
                    print(status)
                
                results.append(row)
                    
            result_df = pd.DataFrame(results)
            detail = result_df['changes_detail'].iloc[0]
            changes_list = extract_and_parse_changes(detail)
            return JSONResponse(content={
                "data": changes_list,
            })
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"เกิดข้อผิดพลาด: {str(e)}")
    
def start_ngrok_if_enabled(port: int) -> Optional[str]:
    """Start ngrok tunnel only when explicitly enabled via env."""
    if not ENABLE_NGROK:
        return None
    return ngrok.connect(port)


if __name__ == "__main__":
    public_url = start_ngrok_if_enabled(PORT)
    if public_url:
        print(f"Public URL: {public_url}")
    uvicorn.run(app, host=HOST, port=PORT)
