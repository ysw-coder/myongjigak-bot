import os
import google.generativeai as genai
from slack_bolt import App
from slack_bolt.adapter.flask import SlackRequestHandler
from flask import Flask, request

# 지식 베이스 로드
with open("knowledge_base.txt", "r", encoding="utf-8") as f:
    KNOWLEDGE_BASE = f.read()

# Slack 앱 초기화
app = App(
    token=os.environ.get("SLACK_BOT_TOKEN"),
    signing_secret=os.environ.get("SLACK_SIGNING_SECRET")
)

# Gemini 초기화
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")

def ask_gemini(user_text):
    prompt = f"""당신은 명지각 숙소의 운영 도우미 AI '명지'입니다.
아래 운영 매뉴얼을 참고하여 질문에 친절하고 간결하게 답변해주세요.
매뉴얼에 없는 내용은 "매뉴얼에 없는 내용이에요 😅"라고 답해주세요.

운영 매뉴얼:
{KNOWLEDGE_BASE}

질문: {user_text}
답변:"""
    response = model.generate_content(prompt)
    return response.text

# 멘션 이벤트 처리
@app.event("app_mention")
def handle_mention(body, say):
    user_text = body["event"]["text"]
    user_text = " ".join(user_text.split()[1:])  # 멘션 부분 제거
    say(ask_gemini(user_text))

# 채널 메시지 처리
@app.event("message")
def handle_message(body, say):
    event = body.get("event", {})
    if event.get("bot_id") or event.get("subtype"):
        return
    user_text = event.get("text", "")
    if user_text:
        say(ask_gemini(user_text))

# Flask 서버
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
