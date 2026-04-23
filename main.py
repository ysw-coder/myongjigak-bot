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

KNOWLEDGE_BASE_PATH = "knowledge_base.txt"

def load_kb():
    with open(KNOWLEDGE_BASE_PATH, "r", encoding="utf-8") as f:
        return f.read()

KNOWLEDGE_BASE = load_kb()

app = App(
    token=SLACK_BOT_TOKEN,
    signing_secret=SLACK_SIGNING_SECRET
)

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

SYSTEM_PERSONA = """당신은 '명지'입니다. 명지각 숙소에서 일하는 베테랑 운영매니저예요.
신입 직원이나 동료들이 운영 관련해서 슬랙으로 물어보면 친근하게 도와주는 역할이에요.

[당신의 성격과 말투]
- 따뜻하고 친근한 동료처럼 대화 (선배 동료 느낌)
- 딱딱한 매뉴얼 읽어주듯 말하지 말고, 자연스럽게 풀어서 설명
- 상황에 따라 답변 길이 조절: 단순 질문은 짧게, 복잡한 절차는 차근차근
- 필요하면 되물어도 됨 ("어느 객실 말씀이세요?" "본관이에요 동관이에요?")
- 이모지는 자연스럽게 가끔 사용 (남용 금지)
- 한국어 존댓말 사용

[가장 중요한 원칙: 사실은 매뉴얼만 따른다]
당신의 머릿속 지식은 아래 [운영 매뉴얼]이 전부입니다.
매뉴얼에 없는 내용은 절대 추측하거나 일반 상식으로 답하지 마세요.
특히 다음 정보는 매뉴얼에 명시되어 있지 않으면 절대 만들어내지 마세요:
- 금액, 가격, 할인율, 수수료
- 계좌번호, 전화번호, 주소
- 시간, 일정, 마감 시각
- 구체적인 절차나 단계
- 비밀번호, 코드, 연동 정보
- 담당자 이름, 연락처

매뉴얼에 답이 없으면 솔직하게 모른다고 말하세요. 예시:
"음, 그건 제 매뉴얼에는 없는 내용이에요 😅"
"아 그 부분은 저도 잘 모르겠어요. 혹시 아시는 분이 알려주시면 학습할 수 있어요!"
"매뉴얼에서 못 찾겠네요. 누가 알려주시면 다음번엔 답할 수 있을 거예요"

이렇게 모른다고 답할 때는 자연스럽게 학습 안내도 곁들이세요:
"이 스레드에 알려주시면 매뉴얼에 추가할게요" 같은 식으로요.

[대화 흐름]
- 같은 스레드에서는 이전 대화를 기억하고 자연스럽게 이어갑니다
- "그건 어떻게 해?" 같은 후속 질문은 직전 대화 맥락 보고 이해
- 동료가 정정하거나 새 정보 알려주면 그 자리에서 받아들이고, 학습 절차로 넘어가세요

"""

def classify_intent(user_text, conversation_context=""):
    classify_prompt = f"""다음 슬랙 메시지를 분류해주세요.

{f"[직전 대화 맥락]{chr(10)}{conversation_context}{chr(10)}" if conversation_context else ""}

[현재 메시지]
"{user_text}"

분류 옵션:
- "learn": 사용자가 명지에게 새로운 정보/사실을 알려주려는 의도가 명확한 경우
  예시: "학습해줘 X는 Y야", "아니 사실은 X야", "기억해, X는 Y", "X는 Y라고 알아둬"
  맥락에서: 명지가 "모른다"고 답한 후에 사용자가 답을 알려주는 경우
  
- "chat": 질문, 잡담, 후속 질문, 감사 인사 등 일반 대화
  예시: "X가 뭐야?", "고마워", "그건 어떻게 해?", "왜 그래?"

확실하지 않으면 "chat"으로 분류하세요.

JSON 형식으로만 응답 (마크다운 없이):
{{"intent": "learn", "extracted_info": "학습할 핵심 사실만 깔끔하게 정리한 문장"}}

또는

{{"intent": "chat"}}

"""
    
    try:
        response = model.generate_content(classify_prompt)
        raw = response.text.strip()
        raw = re.sub(r'^```(?:json)?\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw)
        result = json.loads(raw)
        return result.get("intent", "chat"), result.get("extracted_info", user_text)
    except Exception:
        return "chat", user_text

def chat_with_gemini(conversation_history, current_message):
    history_text = ""
    for msg in conversation_history:
        if msg["role"] == "user":
            history_text += f"동료: {msg['content']}\n"
        else:
            history_text += f"명지: {msg['content']}\n"
    
    prompt = f"""{SYSTEM_PERSONA}

[운영 매뉴얼]
{KNOWLEDGE_BASE}

{f"[지금까지의 대화]{chr(10)}{history_text}" if history_text else ""}

[지금 받은 메시지]
동료: {current_message}

위 대화 흐름과 매뉴얼을 참고해서, 명지답게 자연스럽게 답변해주세요.
다시 강조하지만, 매뉴얼에 없는 사실은 절대 만들어내지 마세요.

명지의 답변:"""
    
    response = model.generate_content(prompt)
    return response.text

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
- merge일 때 updated_section_content에는 해당 섹션의 "=== N. 제목 ===" 줄부터 다음 "===" 직전까지 전체 내용을 넣어주세요 (기존 + 새 정보 잘 합친 버전)
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
        pattern = rf'=== {section_num}\..*?(?====\s*\d+\.|Z)'
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

@app.event("app_mention")
def handle_mention(body, say, client, logger):
    global BOT_USER_ID
    
    event = body["event"]
    text = event["text"]
    channel = event["channel"]
    current_ts = event["ts"]
    thread_ts = event.get("thread_ts") or current_ts
    
    if BOT_USER_ID is None:
        try:
            auth = client.auth_test()
            BOT_USER_ID = auth["user_id"]
            logger.info(f"[봇 ID] {BOT_USER_ID}")
        except Exception as e:
            logger.error(f"[봇 ID 조회 실패] {e}")
    
    user_text = re.sub(r'<@[A-Z0-9a-z]+>', '', text).strip()
    
    logger.info(f"[멘션] 정제: {user_text}")
    
    if not user_text:
        say(text="안녕하세요! 무엇을 도와드릴까요? 🙂", thread_ts=thread_ts)
        return
    
    is_in_thread = event.get("thread_ts") and event["thread_ts"] != current_ts
    history = []
    if is_in_thread:
        history = get_thread_history(client, channel, thread_ts, exclude_ts=current_ts)
        logger.info(f"[스레드 히스토리] {len(history)}개 메시지")
    
    context_str = ""
    if history:
        last_few = history[-4:]
        context_str = "\n".join([f"{m['role']}: {m['content']}" for m in last_few])
    
    intent, extracted = classify_intent(user_text, context_str)
    logger.info(f"[의도] {intent} | 추출: {extracted}")
    
    if intent == "learn":
        say(text="🤔 잠시만요, 학습할게요...", thread_ts=thread_ts)
        
        try:
            context = ""
            if history:
                context = "\n".join([f"{m['role']}: {m['content']}" for m in history[-4:]])
            
            logger.info(f"[학습] 맥락: {context[:200]}")
            logger.info(f"[학습] 정보: {extracted}")
            
            decision = learn_from_text(context, extracted)
            logger.info(f"[학습] 결정: {decision.get('action')} / {decision.get('reasoning')}")
            
            new_kb = update_kb_file(decision)
            
            if decision["action"] == "merge":
                target = f"{decision['target_section_number']}번 항목"
                commit_msg = f"📚 학습: {target} 업데이트"
            else:
                target = decision["new_section_title"]
                commit_msg = f"📚 학습: {target} 신규 추가"
            
            commit_to_github(new_kb, commit_msg)
            
            result_msg = f"""✅ 학습했어요!

📚 {target}에 반영했습니다.

💡 {decision.get('reasoning', '')}

이제 다음에 같은 질문이 오면 바로 답변할 수 있어요 🙂"""
            say(text=result_msg, thread_ts=thread_ts)
        
        except json.JSONDecodeError:
            say(text="❌ 학습 중에 문제가 생겼어요. 한 번만 더 알려주실 수 있을까요?",
                thread_ts=thread_ts)
        except Exception as e:
            logger.error(f"[학습 실패] {e}")
            say(text=f"❌ 학습 실패: {str(e)[:200]}", thread_ts=thread_ts)
        
        return
    
    try:
        answer = chat_with_gemini(history, user_text)
        say(text=answer, thread_ts=thread_ts)
    except Exception as e:
        logger.error(f"[답변 실패] {e}")
        say(text=f"❌ 답변 중 오류가 났어요: {str(e)[:200]}", thread_ts=thread_ts)

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
