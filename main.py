import os
import re
import json
import base64
import requests
import google.generativeai as genai
from slack_bolt import App
from slack_bolt.adapter.flask import SlackRequestHandler
from flask import Flask, request

# ===== 환경변수 =====
SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")
SLACK_SIGNING_SECRET = os.environ.get("SLACK_SIGNING_SECRET")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
GITHUB_REPO = os.environ.get("GITHUB_REPO")
GITHUB_BRANCH = os.environ.get("GITHUB_BRANCH", "main")

BOT_USER_ID = None
USER_NAME_CACHE = {}  # {user_id: "서원"} 형태로 캐싱

# ===== 지식 베이스 로드 =====
KNOWLEDGE_BASE_PATH = "knowledge_base.txt"

def load_kb():
    with open(KNOWLEDGE_BASE_PATH, "r", encoding="utf-8") as f:
        return f.read()

KNOWLEDGE_BASE = load_kb()

# ===== Slack 앱 초기화 =====
app = App(
    token=SLACK_BOT_TOKEN,
    signing_secret=SLACK_SIGNING_SECRET
)

# ===== Gemini 초기화 =====
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")

# ===== 사용자 이름 조회 (성 제외, 이름만) =====
def get_user_first_name(client, user_id):
    """슬랙 user_id로 이름 조회. '최도아' → '도아', '양서원' → '서원'"""
    if not user_id:
        return None
    
    if user_id in USER_NAME_CACHE:
        return USER_NAME_CACHE[user_id]
    
    try:
        result = client.users_info(user=user_id)
        user = result.get("user", {})
        profile = user.get("profile", {})
        
        # 우선순위: display_name > real_name > name
        full_name = (
            profile.get("display_name") 
            or profile.get("real_name") 
            or user.get("real_name")
            or user.get("name")
            or ""
        ).strip()
        
        if not full_name:
            return None
        
        # 한국 이름 처리: 2~4글자면 첫 글자가 성 → 빼고 반환
        # 예: "최도아" → "도아", "양서원" → "서원", "남궁민수" → "민수" (단순 규칙으로는 한계)
        if 2 <= len(full_name) <= 4 and full_name[0] != ' ':
            # 한국 성씨 단순 패턴: 첫 글자만 성으로 가정
            first_name = full_name[1:]
        else:
            first_name = full_name
        
        USER_NAME_CACHE[user_id] = first_name
        return first_name
    except Exception:
        return None

# ===== 의도 분류 (질문 vs 학습) =====
def classify_intent(user_text, conversation_context=""):
    classify_prompt = f"""다음 슬랙 메시지를 분류해주세요.

{f"[직전 대화 맥락]{chr(10)}{conversation_context}{chr(10)}" if conversation_context else ""}
[현재 메시지]
"{user_text}"

분류 옵션:
- "learn": 사용자가 명지에게 새로운 정보/사실을 알려주려는 의도가 명확한 경우
  예시: "학습해줘 X는 Y야", "기억해 X는 Y", "X는 Y라고 알아둬", "X는 Y인 거 추가해줘"
  맥락에서: 명지가 "모른다"고 답한 후에 사용자가 답을 알려주는 경우
  
- "ask": 정보를 묻거나 질문하는 경우, 일반 대화, 후속 질문
  예시: "X가 뭐야?", "X 어떻게 해?", "X 알려줘", "그건?"

확실하지 않으면 "ask"로 분류하세요.

JSON 형식으로만 응답 (마크다운 없이):
{{"intent": "learn", "extracted_info": "학습할 핵심 사실만 깔끔하게 정리한 문장"}}
또는
{{"intent": "ask"}}
"""
    
    try:
        response = model.generate_content(classify_prompt)
        raw = response.text.strip()
        raw = re.sub(r'^```(?:json)?\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw)
        result = json.loads(raw)
        return result.get("intent", "ask"), result.get("extracted_info", user_text)
    except Exception:
        return "ask", user_text

# ===== 답변 생성 (간결 톤 + 스레드 맥락 + 호칭) =====
def ask_gemini(user_text, user_first_name=None, conversation_history=None):
    # 스레드 맥락 정리
    history_text = ""
    if conversation_history:
        for msg in conversation_history:
            if msg["role"] == "user":
                history_text += f"동료: {msg['content']}\n"
            else:
                history_text += f"명지: {msg['content']}\n"
    
    # 호칭 안내
    name_instruction = ""
    if user_first_name:
        name_instruction = f"\n질문한 사람의 이름은 '{user_first_name}'입니다. 자연스럽게 '{user_first_name}님' 호칭을 써도 됩니다 (매번 부를 필요는 없음, 적절히)."
    
    prompt = f"""당신은 명지각 숙소의 운영 도우미 AI '명지'입니다.
명지각 운영매니저들이 슬랙으로 운영 관련 질문을 합니다.

[답변 원칙]
- 반드시 존댓말 사용 (운영매니저들끼리 서로 존대함)
- 간결하고 사실 위주로 답변 (불필요한 수식어, 감탄사 금지)
- 이모지는 거의 사용하지 않음 (꼭 필요한 경우만 1개 정도)
- 매뉴얼에 답이 있으면 → 명확하게 답변
- 매뉴얼에 답이 없으면 → "매뉴얼에 없는 내용이에요. 알려주시면 학습할게요." 형태로 짧게 답변
- 매뉴얼에 없는 내용을 추측하거나 일반 상식으로 답하지 마세요 (특히 금액/계좌/시간/연락처/담당자명/구체 절차)
- "OO님" 같이 빈 자리 만들지 마세요. 이름을 모르면 호칭 자체를 생략하세요.{name_instruction}

[운영 매뉴얼]
{KNOWLEDGE_BASE}

{f"[지금까지의 대화]{chr(10)}{history_text}" if history_text else ""}
[현재 질문]
{user_text}

답변:"""
    
    response = model.generate_content(prompt)
    return response.text

# ===== 학습 처리 =====
def learn_from_text(context, new_info):
    global KNOWLEDGE_BASE
    
    learn_prompt = f"""당신은 명지각 운영 매뉴얼을 정리하는 편집자입니다.

[기존 매뉴얼]
{KNOWLEDGE_BASE}

[새로 학습할 정보]
대화 맥락: {context}
직원이 알려준 핵심 사실: {new_info}

이 정보를 매뉴얼에 어떻게 추가할지 결정해주세요.

규칙:
- 기존 항목과 주제가 같거나 보완 관계면 "merge"
- 완전히 새로운 주제면 "new"
- merge일 때 updated_section_content에는 해당 섹션의 "=== N. 제목 ===" 줄부터 다음 "===" 직전까지 전체 내용을 넣어주세요 (기존 + 새 정보 합친 버전)
- new일 때 updated_section_content에는 새 섹션 전체를 "=== N. 제목 ===" 형식으로 시작
- 새 섹션 번호는 기존 마지막 번호 + 1

JSON 형식으로만 응답 (마크다운 없이):
{{
  "action": "merge",
  "target_section_number": 18,
  "new_section_title": null,
  "updated_section_content": "=== 18. 동관 공간별 주의사항 ===\\n...전체 내용...",
  "reasoning": "기존 동관 시설 항목에 와이파이 정보 보완"
}}
또는:
{{
  "action": "new",
  "target_section_number": null,
  "new_section_title": "32. 와이파이 정보",
  "updated_section_content": "=== 32. 와이파이 정보 ===\\n- 명지각 와이파이는 비밀번호 없이 자유 사용 가능",
  "reasoning": "기존에 없던 와이파이 주제"
}}
"""
    
    response = model.generate_content(learn_prompt)
    raw = response.text.strip()
    raw = re.sub(r'^```(?:json)?\s*', '', raw)
    raw = re.sub(r'\s*```$', '', raw)
    
    return json.loads(raw)

def update_kb_file(decision):
    global KNOWLEDGE_BASE
    
    new_kb = KNOWLEDGE_BASE
    
    if decision["action"] == "merge":
        section_num = decision["target_section_number"]
        pattern = rf'=== {section_num}\..*?(?====\s*\d+\.|\Z)'
        new_section = decision["updated_section_content"].rstrip() + "\n\n"
        new_kb = re.sub(pattern, new_section, KNOWLEDGE_BASE, count=1, flags=re.DOTALL)
    else:
        new_section = "\n" + decision["updated_section_content"].rstrip() + "\n"
        new_kb = KNOWLEDGE_BASE.rstrip() + "\n" + new_section
    
    KNOWLEDGE_BASE = new_kb
    
    with open(KNOWLEDGE_BASE_PATH, "w", encoding="utf-8") as f:
        f.write(new_kb)
    
    return new_kb

def commit_to_github(new_content, commit_message):
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{KNOWLEDGE_BASE_PATH}"
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json"
    }
    
    r = requests.get(url, headers=headers, params={"ref": GITHUB_BRANCH})
    if r.status_code != 200:
        raise Exception(f"GitHub 파일 조회 실패: {r.status_code}")
    sha = r.json()["sha"]
    
    payload = {
        "message": commit_message,
        "content": base64.b64encode(new_content.encode("utf-8")).decode("utf-8"),
        "sha": sha,
        "branch": GITHUB_BRANCH
    }
    r = requests.put(url, headers=headers, json=payload)
    if r.status_code not in (200, 201):
        raise Exception(f"GitHub 커밋 실패: {r.status_code}")
    
    return r.json()

# ===== 스레드 대화 히스토리 가져오기 =====
def get_thread_history(client, channel, thread_ts, exclude_ts=None, max_msgs=20):
    global BOT_USER_ID
    
    try:
        result = client.conversations_replies(
            channel=channel,
            ts=thread_ts,
            limit=max_msgs
        )
        messages = result.get("messages", [])
    except Exception:
        return []
    
    history = []
    for msg in messages:
        if exclude_ts and msg.get("ts") == exclude_ts:
            continue
        
        text = msg.get("text", "")
        text = re.sub(r'<@[A-Z0-9a-z]+>', '', text).strip()
        if not text:
            continue
        
        if msg.get("user") == BOT_USER_ID or msg.get("bot_id"):
            history.append({"role": "assistant", "content": text})
        else:
            history.append({"role": "user", "content": text})
    
    return history

# ===== 슬랙 멘션 핸들러 =====
@app.event("app_mention")
def handle_mention(body, say, client, logger):
    global BOT_USER_ID
    
    event = body["event"]
    text = event["text"]
    channel = event["channel"]
    current_ts = event["ts"]
    thread_ts = event.get("thread_ts") or current_ts
    user_id = event.get("user")
    
    # 봇 ID 캐싱
    if BOT_USER_ID is None:
        try:
            auth = client.auth_test()
            BOT_USER_ID = auth["user_id"]
        except Exception as e:
            logger.error(f"[봇 ID 조회 실패] {e}")
    
    # 사용자 이름 조회
    user_first_name = get_user_first_name(client, user_id)
    logger.info(f"[사용자] {user_id} → {user_first_name}")
    
    # 멘션 부분 제거
    user_text = re.sub(r'<@[A-Z0-9a-z]+>', '', text).strip()
    logger.info(f"[멘션] {user_text}")
    
    if not user_text:
        say(text="안녕하세요. 명지각 운영 관련 질문 주세요.", thread_ts=thread_ts)
        return
    
    # 스레드 히스토리 (현재 메시지 제외)
    is_in_thread = event.get("thread_ts") and event["thread_ts"] != current_ts
    history = []
    if is_in_thread:
        history = get_thread_history(client, channel, thread_ts, exclude_ts=current_ts)
        logger.info(f"[스레드] {len(history)}개 메시지")
    
    # 의도 분류
    context_str = ""
    if history:
        last_few = history[-4:]
        context_str = "\n".join([f"{m['role']}: {m['content']}" for m in last_few])
    
    intent, extracted = classify_intent(user_text, context_str)
    logger.info(f"[의도] {intent}")
    
    # ===== 학습 처리 =====
    if intent == "learn":
        say(text="학습 중입니다...", thread_ts=thread_ts)
        
        try:
            context = ""
            if history:
                context = "\n".join([f"{m['role']}: {m['content']}" for m in history[-4:]])
            
            decision = learn_from_text(context, extracted)
            logger.info(f"[학습 결정] {decision.get('action')} / {decision.get('reasoning')}")
            
            new_kb = update_kb_file(decision)
            
            if decision["action"] == "merge":
                target = f"{decision['target_section_number']}번 항목"
                commit_msg = f"학습: {target} 업데이트"
            else:
                target = decision["new_section_title"]
                commit_msg = f"학습: {target} 신규 추가"
            
            commit_to_github(new_kb, commit_msg)
            
            result_msg = f"""학습 완료. {target}에 반영했습니다.
({decision.get('reasoning', '')})"""
            say(text=result_msg, thread_ts=thread_ts)
        
        except json.JSONDecodeError:
            say(text="학습 실패: 응답을 해석하지 못했어요. 다시 알려주세요.",
                thread_ts=thread_ts)
        except Exception as e:
            logger.error(f"[학습 실패] {e}")
            say(text=f"학습 실패: {str(e)[:200]}", thread_ts=thread_ts)
        
        return
    
    # ===== 일반 답변 =====
    try:
        answer = ask_gemini(user_text, user_first_name=user_first_name, conversation_history=history)
        say(text=answer, thread_ts=thread_ts)
    except Exception as e:
        logger.error(f"[답변 실패] {e}")
        say(text=f"답변 중 오류가 났어요: {str(e)[:200]}", thread_ts=thread_ts)

# ===== Flask 서버 =====
flask_app = Flask(__name__)
handler = SlackRequestHandler(app)

@flask_app.route("/slack/events", methods=["POST"])
def slack_events():
    return handler.handle(request)

@flask_app.route("/", methods=["GET"])
def health():
    return {"status": "ok"}, 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3000))
    flask_app.run(host="0.0.0.0", port=port)
