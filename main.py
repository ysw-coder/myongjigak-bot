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
GITHUB_REPO = os.environ.get("GITHUB_REPO")  # 예: ysw-coder/myongjigak-bot
GITHUB_BRANCH = os.environ.get("GITHUB_BRANCH", "main")

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
model = genai.GenerativeModel("gemini-1.5-flash")

# ===== 일반 답변 =====
def ask_gemini(user_text):
    prompt = f"""당신은 명지각 숙소의 운영 도우미 AI '명지'입니다.
아래 운영 매뉴얼을 참고하여 질문에 친절하고 간결하게 답변해주세요.
매뉴얼에 없는 내용은 "매뉴얼에 없는 내용이에요 😅 알려주시면 학습할게요! 스레드에 '@명지 학습 [내용]' 으로 알려주세요."라고 답해주세요.

운영 매뉴얼:
{KNOWLEDGE_BASE}

질문: {user_text}
답변:"""
    response = model.generate_content(prompt)
    return response.text

# ===== 학습 기능 =====
def learn_from_text(original_question, new_info):
    """
    Gemini에게 기존 KB와 새 정보를 주고, 어떻게 병합할지 JSON으로 받음
    """
    global KNOWLEDGE_BASE
    
    learn_prompt = f"""당신은 명지각 운영 매뉴얼을 정리하는 편집자입니다.
아래는 기존 매뉴얼이고, 새로 학습할 정보가 들어왔습니다.
이 정보를 매뉴얼에 어떻게 추가할지 결정해주세요.

[기존 매뉴얼]
{KNOWLEDGE_BASE}

[새로 학습할 정보]
원본 질문: {original_question}
직원이 알려준 정보: {new_info}

다음 규칙에 따라 JSON으로만 응답하세요 (마크다운 코드블록 ``` 없이, 순수 JSON만):

규칙:
- 기존 항목과 주제가 같거나 보완 관계면 "merge"
- 완전히 새로운 주제면 "new"
- merge일 때 updated_section_content에는 해당 섹션의 "=== N. 제목 ===" 줄부터 다음 "===" 직전까지 전체 내용을 넣어주세요 (기존 + 새 정보 합친 버전)
- new일 때 updated_section_content에는 새 섹션 전체를 "=== N. 제목 ===" 형식으로 시작하게 작성해주세요
- 새 섹션 번호는 기존 마지막 번호 + 1로 매기세요

응답 형식:
{{
  "action": "merge",
  "target_section_number": 18,
  "new_section_title": null,
  "updated_section_content": "=== 18. 동관 공간별 주의사항 ===\\n...전체 내용...",
  "reasoning": "동관 시설 관련 정보라 기존 18번 항목에 병합"
}}

또는:
{{
  "action": "new",
  "target_section_number": null,
  "new_section_title": "32. 와이파이 정보",
  "updated_section_content": "=== 32. 와이파이 정보 ===\\n- 동관 와이파이: myongji2024",
  "reasoning": "기존에 없던 새로운 주제"
}}
"""
    
    response = model.generate_content(learn_prompt)
    raw = response.text.strip()
    
    # 혹시 코드블록 들어있으면 제거
    raw = re.sub(r'^```(?:json)?\s*', '', raw)
    raw = re.sub(r'\s*```$', '', raw)
    
    return json.loads(raw)

def update_kb_file(decision):
    """
    Gemini가 결정한 내용대로 knowledge_base.txt 수정
    """
    global KNOWLEDGE_BASE
    
    new_kb = KNOWLEDGE_BASE
    
    if decision["action"] == "merge":
        section_num = decision["target_section_number"]
        # === N. ... === 부터 다음 === 직전까지 매칭
        pattern = rf'=== {section_num}\..*?(?====\s*\d+\.|\Z)'
        new_section = decision["updated_section_content"].rstrip() + "\n\n"
        new_kb = re.sub(pattern, new_section, KNOWLEDGE_BASE, count=1, flags=re.DOTALL)
    else:  # new
        new_section = "\n" + decision["updated_section_content"].rstrip() + "\n"
        new_kb = KNOWLEDGE_BASE.rstrip() + "\n" + new_section
    
    # 메모리 갱신
    KNOWLEDGE_BASE = new_kb
    
    # 로컬 파일도 갱신
    with open(KNOWLEDGE_BASE_PATH, "w", encoding="utf-8") as f:
        f.write(new_kb)
    
    return new_kb

def commit_to_github(new_content, commit_message):
    """
    GitHub API로 knowledge_base.txt를 커밋
    """
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{KNOWLEDGE_BASE_PATH}"
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json"
    }
    
    # 1. 현재 파일의 SHA 가져오기
    r = requests.get(url, headers=headers, params={"ref": GITHUB_BRANCH})
    if r.status_code != 200:
        raise Exception(f"GitHub 파일 조회 실패: {r.status_code} {r.text}")
    sha = r.json()["sha"]
    
    # 2. 새 내용으로 업데이트
    payload = {
        "message": commit_message,
        "content": base64.b64encode(new_content.encode("utf-8")).decode("utf-8"),
        "sha": sha,
        "branch": GITHUB_BRANCH
    }
    r = requests.put(url, headers=headers, json=payload)
    if r.status_code not in (200, 201):
        raise Exception(f"GitHub 커밋 실패: {r.status_code} {r.text}")
    
    return r.json()

# ===== 슬랙 멘션 핸들러 =====
@app.event("app_mention")
def handle_mention(body, say, client):
    event = body["event"]
    text = event["text"]
    channel = event["channel"]
    thread_ts = event.get("thread_ts") or event["ts"]
    
    # 멘션 부분 제거 (<@U12345> 형태)
    user_text = re.sub(r'<@[A-Z0-9]+>', '', text).strip()
    
    # ===== 학습 명령 분기 =====
    if user_text.startswith("학습"):
        # "학습" 키워드 뒤 내용 추출
        new_info = user_text[2:].strip()
        if not new_info:
            say(text="❓ 학습할 내용을 함께 적어주세요. 예: `@명지 학습 동관 와이파이는 myongji2024야`",
                thread_ts=thread_ts)
            return
        
        # 임시 메시지
        say(text="🤔 학습 중...", thread_ts=thread_ts)
        
        try:
            # 스레드 부모 메시지 가져오기 (원본 질문)
            original_question = ""
            if event.get("thread_ts") and event["thread_ts"] != event["ts"]:
                # 스레드 댓글로 학습 명령이 온 경우
                history = client.conversations_replies(
                    channel=channel,
                    ts=event["thread_ts"],
                    limit=1
                )
                if history["messages"]:
                    parent_text = history["messages"][0].get("text", "")
                    # 부모 메시지에서도 멘션 제거
                    original_question = re.sub(r'<@[A-Z0-9]+>', '', parent_text).strip()
            
            # Gemini에게 어떻게 병합할지 결정 요청
            decision = learn_from_text(original_question, new_info)
            
            # 파일 수정
            new_kb = update_kb_file(decision)
            
            # 커밋 메시지
            if decision["action"] == "merge":
                target = f"{decision['target_section_number']}번 항목"
                commit_msg = f"📚 학습: {target} 업데이트"
            else:
                target = decision["new_section_title"]
                commit_msg = f"📚 학습: {target} 신규 추가"
            
            # GitHub 커밋
            commit_to_github(new_kb, commit_msg)
            
            # 슬랙에 결과 알림
            result_msg = f"""✅ 학습 완료!
📚 *{target}*에 반영했습니다.
💡 {decision.get('reasoning', '')}

곧 자동 재배포되면 다음 질문부터 반영돼요."""
            say(text=result_msg, thread_ts=thread_ts)
        
        except json.JSONDecodeError as e:
            say(text=f"❌ 학습 실패: AI 응답을 해석하지 못했어요. 다시 시도해주세요.\n```{str(e)[:200]}```",
                thread_ts=thread_ts)
        except Exception as e:
            say(text=f"❌ 학습 실패: {str(e)[:300]}", thread_ts=thread_ts)
        
        return
    
    # ===== 일반 질문 답변 =====
    if not user_text:
        say(text="안녕하세요! 명지각 운영 관련 질문을 해주세요 🙂", thread_ts=thread_ts)
        return
    
    try:
        answer = ask_gemini(user_text)
        say(text=answer, thread_ts=thread_ts)
    except Exception as e:
        say(text=f"❌ 답변 생성 중 오류: {str(e)[:200]}", thread_ts=thread_ts)

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
