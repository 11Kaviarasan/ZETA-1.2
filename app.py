"""
app.py — Zeta AI Flask Backend
Domain : https://www.zetaai.com
Server : 150.230.133.117
DB     : MongoDB (via db.py)
Vectors: Pinecone (via proper_ai.py / mars_ai.py)
"""

import os, hmac, hashlib, logging
from functools import wraps
from datetime import datetime

import razorpay
from flask import Flask, request, jsonify, g, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
load_dotenv(env_path)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("zeta.api")

app = Flask(__name__, static_folder=".")
app.secret_key = os.getenv("SECRET_KEY", "dev_change_me")

CORS(app,
     origins=[
         "https://www.zetaai.com",
         "https://zetaai.com",
         "http://150.230.133.117",
         "http://localhost:3000",
         "http://localhost:5000",
     ],
     supports_credentials=True)

import db, mars_ai

# ─── Razorpay client ──────────────────────────────────────────────────────────

_rp = razorpay.Client(
    auth=(os.getenv("RAZORPAY_KEY_ID"), os.getenv("RAZORPAY_KEY_SECRET"))
)

# ─── Response helpers ─────────────────────────────────────────────────────────

def ok(data=None, **kw):
    p = {"status": "ok", "timestamp": datetime.utcnow().isoformat()}
    if data:
        p.update(data)
    p.update(kw)
    return jsonify(p)

def err(msg, code="ERROR", status=400):
    return jsonify({"status": "error", "error": msg, "code": code}), status

# ─── Auth decorators ──────────────────────────────────────────────────────────

def require_auth(f):
    @wraps(f)
    def dec(*a, **kw):
        sid = (request.headers.get("X-Session-Token")
               or request.cookies.get("zeta_session"))
        if not sid:
            return err("Unauthorized", "NO_TOKEN", 401)
        uid = db.validate_session(sid)
        if not uid:
            return err("Session expired", "INVALID_SESSION", 401)
        g.user_id = uid
        sub = db.get_subscription(uid)
        g.plan = sub.get("plan", "basic")
        return f(*a, **kw)
    return dec

def require_pro(f):
    @wraps(f)
    def dec(*a, **kw):
        if getattr(g, "plan", "basic") != "pro":
            return err("Pro subscription required.", "PRO_REQUIRED", 403)
        return f(*a, **kw)
    return dec

def optional_auth(f):
    @wraps(f)
    def dec(*a, **kw):
        sid = (request.headers.get("X-Session-Token")
               or request.cookies.get("zeta_session"))
        g.user_id = db.validate_session(sid) if sid else None
        if g.user_id:
            sub = db.get_subscription(g.user_id)
            g.plan = sub.get("plan", "basic")
        else:
            g.plan = "basic"
        return f(*a, **kw)
    return dec

def require_api_key(f):
    @wraps(f)
    def dec(*a, **kw):
        key = (request.headers.get("X-Zeta-API-Key")
               or request.headers.get("Authorization", "").replace("Bearer ", ""))
        if not key:
            return err("API key required", "NO_API_KEY", 401)
        user_info = db.validate_api_key(key)
        if not user_info:
            return err("Invalid or revoked API key", "INVALID_KEY", 401)
        g.user_id = user_info["user_id"]
        g.plan    = "pro"
        return f(*a, **kw)
    return dec

# ─── Static / Health ──────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/api")
def api_root():
    return jsonify({"message": "Zeta AI engine running", "domain": "https://www.zetaai.com"})

@app.route("/api/health")
def health():
    return ok({"engine": mars_ai.get_engine_status(), "version": "4.2.0",
               "db": "mongodb", "vectors": "pinecone"})

# ─── Auth ─────────────────────────────────────────────────────────────────────

@app.route("/api/auth/register", methods=["POST"])
def register():
    d = request.get_json(silent=True) or {}
    email    = d.get("email", "").strip()
    username = d.get("username", "").strip()
    password = d.get("password", "")
    if not all([email, username, password]):
        return err("All fields required.")
    if len(password) < 8:
        return err("Password must be at least 8 characters.")
    if "@" not in email:
        return err("Invalid email address.")
    try:
        user = db.create_user(email, username, password)
        sid  = db.create_session(user["user_id"])
        return ok({"user": user, "session_token": sid}), 201
    except ValueError as e:
        return err(str(e), "EMAIL_TAKEN")
    except Exception as e:
        logger.error(e)
        return err("Registration failed.", status=500)

@app.route("/api/auth/login", methods=["POST"])
def login():
    d    = request.get_json(silent=True) or {}
    user = db.authenticate_user(d.get("email", ""), d.get("password", ""))
    if not user:
        return err("Invalid credentials.", "INVALID_CREDENTIALS", 401)
    sid = db.create_session(user["user_id"])
    sub = db.get_subscription(user["user_id"])
    return ok({"user": user, "session_token": sid, "subscription": sub})

@app.route("/api/auth/me")
@require_auth
def me():
    user = db.get_user_by_id(g.user_id)
    sub  = db.get_subscription(g.user_id)
    if not user:
        return err("User not found.", status=404)
    return ok({"user": user, "subscription": sub})

@app.route("/api/auth/logout", methods=["POST"])
@require_auth
def logout():
    sid = (request.headers.get("X-Session-Token")
           or request.cookies.get("zeta_session"))
    if sid:
        db.delete_session(sid)
    return ok({"message": "Logged out."})

# ─── Conversations ────────────────────────────────────────────────────────────

@app.route("/api/conversations")
@require_auth
def list_conversations():
    return ok({"conversations": db.get_user_conversations(g.user_id)})

@app.route("/api/conversations", methods=["POST"])
@require_auth
def create_conversation():
    d   = request.get_json(silent=True) or {}
    cid = db.create_conversation(g.user_id, d.get("title", "New Conversation"),
                                 d.get("model", "zeta-4-turbo"))
    return ok({"conv_id": cid}), 201

@app.route("/api/conversations/<cid>/messages")
@require_auth
def get_messages(cid):
    return ok({"messages": db.get_conversation_messages(cid), "conv_id": cid})

# ─── Chat ─────────────────────────────────────────────────────────────────────

@app.route("/api/chat", methods=["POST"])
@optional_auth
def chat():
    d        = request.get_json(silent=True) or {}
    question = (d.get("question") or d.get("message", "")).strip()
    if not question:
        return err("Question required.")
    conv_id = d.get("conv_id")
    history = []
    if conv_id:
        try:
            history = db.get_conversation_messages(conv_id)
        except Exception:
            pass

    result = mars_ai.query(question, history, g.user_id, conv_id, d.get("model", "auto"))
    kid    = None
    try:
        kid = db.save_knowledge(question, result["answer"], conv_id, g.user_id,
                                result["source"], "chat", result["tokens"])
        if conv_id and not history:
            db.update_conversation_title(conv_id, question[:60])
    except Exception as e:
        logger.error(f"DB save: {e}")

    return ok({
        "answer":      result["answer"],
        "reply":       result["answer"],
        "knowledge_id": kid,
        "source":      result["source"],
        "intent":      result["intent"],
        "language":    result["language"],
        "tokens_used": result["tokens"],
        "cache_hit":   result["cache_hit"],
    })

# ─── External API (via zeta_live_ key) ───────────────────────────────────────

@app.route("/api/v1/chat", methods=["POST"])
@require_api_key
def external_chat():
    d        = request.get_json(silent=True) or {}
    question = (d.get("question") or d.get("message", "")).strip()
    if not question:
        return err("question field required.")
    result = mars_ai.query(question, [], g.user_id, None, d.get("model", "auto"))
    try:
        db.save_knowledge(question, result["answer"], None, g.user_id,
                          result["source"], "api", result["tokens"])
    except Exception:
        pass
    return ok({
        "answer":    result["answer"],
        "source":    result["source"],
        "intent":    result["intent"],
        "language":  result["language"],
        "tokens":    result["tokens"],
        "cache_hit": result["cache_hit"],
    })

# ─── API Keys ─────────────────────────────────────────────────────────────────

@app.route("/api/keys", methods=["GET"])
@require_auth
def get_key():
    return ok({"key": db.get_api_key_info(g.user_id), "requires_pro": False})

@app.route("/api/keys/generate", methods=["POST"])
@require_auth
@require_pro
def generate_key():
    d = request.get_json(silent=True) or {}
    try:
        key_data = db.generate_api_key(g.user_id, d.get("label", "Default Key"))
        return ok({"key": key_data,
                   "warning": "Store this key securely — it will not be shown again."}), 201
    except Exception as e:
        logger.error(e)
        return err("Key generation failed.", status=500)

@app.route("/api/keys/revoke", methods=["POST"])
@require_auth
@require_pro
def revoke_key():
    db.revoke_api_key(g.user_id)
    return ok({"message": "API key revoked."})

# ─── Razorpay Payments ────────────────────────────────────────────────────────

PLAN_AMOUNTS = {
    ("pro", "monthly"): int(os.getenv("PLAN_PRO_MONTHLY_PAISE", 18900)),
    ("pro", "yearly"):  int(os.getenv("PLAN_PRO_YEARLY_PAISE",  189900)),
}

@app.route("/api/payments/create-order", methods=["POST"])
@require_auth
def create_order():
    d     = request.get_json(silent=True) or {}
    plan  = d.get("plan", "pro")
    cycle = d.get("billing_cycle", "monthly")
    key   = (plan, cycle)
    if key not in PLAN_AMOUNTS:
        return err("Invalid plan/cycle combination.")
    amount = PLAN_AMOUNTS[key]
    try:
        order = _rp.order.create({
            "amount":   amount,
            "currency": "INR",
            "receipt":  f"zeta_{g.user_id[:8]}_{cycle[:1]}",
            "notes":    {"user_id": g.user_id, "plan": plan, "cycle": cycle},
        })
        return ok({
            "order_id":        order["id"],
            "amount":          amount,
            "currency":        "INR",
            "plan":            plan,
            "billing_cycle":   cycle,
            "razorpay_key_id": os.getenv("RAZORPAY_KEY_ID"),
        })
    except Exception as e:
        logger.error(f"Razorpay order: {e}")
        return err("Payment order creation failed.", status=500)

@app.route("/api/payments/verify", methods=["POST"])
@require_auth
def verify_payment():
    d             = request.get_json(silent=True) or {}
    rp_payment_id = d.get("razorpay_payment_id", "")
    rp_order_id   = d.get("razorpay_order_id", "")
    rp_signature  = d.get("razorpay_signature", "")
    plan          = d.get("plan", "pro")
    cycle         = d.get("billing_cycle", "monthly")
    if not all([rp_payment_id, rp_order_id, rp_signature]):
        return err("Payment details incomplete.")
    body     = f"{rp_order_id}|{rp_payment_id}"
    secret   = os.getenv("RAZORPAY_KEY_SECRET", "").encode()
    expected = hmac.new(secret, body.encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, rp_signature):
        return err("Signature mismatch.", "SIG_MISMATCH", 400)
    amount = PLAN_AMOUNTS.get((plan, cycle), 18900)
    try:
        db.save_payment(g.user_id, rp_payment_id, rp_order_id, rp_signature, amount, plan, cycle)
        db.upgrade_subscription(g.user_id, plan, cycle, rp_order_id)
        return ok({"message": f"Welcome to Zeta {plan.capitalize()}!",
                   "user": db.get_user_by_id(g.user_id), "plan": plan, "cycle": cycle})
    except Exception as e:
        logger.error(f"Upgrade error: {e}")
        return err("Payment recorded but upgrade failed. Contact support.", status=500)

@app.route("/api/payments/cancel", methods=["POST"])
@require_auth
def cancel_plan():
    db.cancel_subscription(g.user_id)
    return ok({"message": "Subscription cancelled."})

@app.route("/api/payments/subscription")
@require_auth
def subscription_info():
    return ok({"subscription": db.get_subscription(g.user_id)})

@app.route("/api/webhooks/razorpay", methods=["POST"])
def razorpay_webhook():
    sig    = request.headers.get("X-Razorpay-Signature", "")
    body   = request.get_data()
    secret = os.getenv("RAZORPAY_WEBHOOK_SECRET", "").encode()
    digest = hmac.new(secret, body, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(digest, sig):
        return jsonify({"error": "Invalid signature"}), 400
    payload = request.get_json(silent=True) or {}
    logger.info(f"Razorpay webhook: {payload.get('event','')}")
    return jsonify({"status": "ok"})

# ─── Feedback ─────────────────────────────────────────────────────────────────

@app.route("/api/feedback", methods=["POST"])
@optional_auth
def feedback():
    d      = request.get_json(silent=True) or {}
    kid    = d.get("knowledge_id")
    rating = d.get("rating")
    if not kid or rating not in (1, -1):
        return err("knowledge_id and rating (1 or -1) required.")
    db.save_feedback(kid, rating, g.user_id, d.get("comment", ""))
    return ok({"message": "Feedback saved."})

# ─── Admin ────────────────────────────────────────────────────────────────────

@app.route("/api/admin/stats")
@require_auth
def stats():
    try:
        # Check if user is actually admin or allowed (simple check for now)
        if g.user_id: # Add restricted check if needed
            return ok({"db": db.get_stats(), "engine": mars_ai.get_engine_status()})
        return err("Unauthorized", status=403)
    except Exception as e:
        return err(str(e), status=500)

# ─── Error handlers ───────────────────────────────────────────────────────────

@app.errorhandler(404)
def not_found(_):
    return err("Not found.", "NOT_FOUND", 404)

@app.errorhandler(500)
def server_error(e):
    logger.error(e)
    return err("Internal server error.", "SERVER_ERROR", 500)

# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        db.bootstrap_schema()
    except Exception as e:
        logger.warning(f"DB bootstrap: {e}")
    port  = int(os.getenv("FLASK_PORT", 5000))
    debug = os.getenv("DEBUG_MODE", "false").lower() == "true"
    logger.info(f"Zeta AI API starting on :{port} — https://www.zetaai.com")
    app.run(host="0.0.0.0", port=port, debug=debug)
