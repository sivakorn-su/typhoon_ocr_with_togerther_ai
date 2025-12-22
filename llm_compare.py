def clean_html_preserve_lines(text: str) -> list:
    """
    Clean HTML text but preserve lines/sentences.
    """
    if not text: return []
    text = re.sub(r'<br\s*/?>', '\n', text, flags=re.I)
    text = re.sub(r'</div>', '\n', text, flags=re.I)
    text = re.sub(r'</p>', '\n', text, flags=re.I)
    soup = BeautifulSoup(text, 'html.parser')
    text_content = soup.get_text(separator=' ')
    lines = [line.strip() for line in text_content.split('\n') if line.strip()]
    return lines

def llm_find_best_match(query_sentence: str, full_text_doc2: str, model: str = 'meta-llama/Llama-3.3-70B-Instruct-Turbo') -> dict:
    """
    ส่งประโยคจาก Doc 1 ไปให้ LLM ค้นหาใน Doc 2 ทั้งก้อน
    """
    client = Together(api_key=TOGETHER_API_KEY)
    
    prompt = f"""
    ภารกิจ: คุณคือนักสืบข้อมูล (Data Detective)
    ฉันมี "ประโยคเป้าหมาย" (Target Sentence) จากเอกสารฉบับเก่า
    ช่วยค้นหาว่าใน "เนื้อหาเอกสารฉบับใหม่" (Full Text Dataset) มีประโยคหรือข้อความส่วนไหนที่มีความหมาย **ตรงกัน** หรือ **สื่อถึงเรื่องเดียวกัน** หรือไม่
    
    --- ประโยคเป้าหมาย (จาก Doc 1) ---
    "{{query_sentence}}"
    
    --- เนื้อหาเอกสารฉบับใหม่ (Doc 2) ---
    {{full_text_doc2}}
    
    --- คำสั่ง ---
    1. จงค้นหาข้อความใน Doc 2 ที่ตรงกับ Doc 1 มากที่สุด (แม้จะเขียนต่างกัน, คนละภาษา, หรือสลับที่)
    2. ถ้าเจอ ให้คัดลอกข้อความนั้นมาใส่ใน 'found_text'
    3. ให้คะแนนความมั่นใจ (confidence) 0-100
    4. ถ้าไม่เจออะไรเลยที่เกี่ยวข้องกัน ให้ตอบ found_text: null
    
    ตอบเป็น JSON เท่านั้น:
    {{{{
        "found_text": "ข้อความที่เจอใน Doc 2 หรือ null",
        "confidence": 95,
        "reason": "เหตุผลสั้นๆ (เช่น เลขตรงกัน, แปลภาษามา)"
    }}}}
    """
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt.format(query_sentence=query_sentence, full_text_doc2=full_text_doc2)}],
            temperature=0.0, # ต้องนิ่งที่สุด
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
        if "```json" in content: content = content.replace("```json", "").replace("```", "")
        return json.loads(content)
    except Exception as e:
        return {"found_text": None, "confidence": 0, "reason": str(e)}

def compare_using_full_llm_search(raw_text1: str, raw_text2: str) -> list:
    # 1. แยก Doc 1 เป็นประโยค
    lines1 = clean_html_preserve_lines(raw_text1)
    
    # 2. Doc 2 ส่งไปทั้งก้อน (แต่ Clean HTML ก่อนนะ เดี๋ยว Token เกิน)
    lines2_all = clean_html_preserve_lines(raw_text2)
    doc2_full_text = "\n".join(lines2_all) # รวมเป็น Text ก้อนเดียว
    
    results = []
    
    print(f"🚀 กำลังใช้ AI สแกนหาคู่สำหรับ {len(lines1)} ประโยค (อาจใช้เวลาสักครู่)...")
    
    for i, line1 in enumerate(lines1):
        if len(line1) < 4: continue # ข้ามคำสั้นๆ
        
        # เรียก LLM (ทีละบรรทัด)
        # ⚠️ ระวัง: ถ้าเอกสารยาวมาก อาจจะเปลือง Token/Quota ได้
        print(f"[{i+1}/{len(lines1)}] กำลังหา: {line1[:30]}...") 
        match_result = llm_find_best_match(line1, doc2_full_text)
        
        results.append({
            "doc1": line1,
            "doc2": match_result.get('found_text'),
            "score": match_result.get('confidence', 0),
            "reason": match_result.get('reason', '-')
        })
        
    return results
