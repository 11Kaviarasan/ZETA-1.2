"""
db.py — Zeta AI Database Layer (MongoDB)
Replaces Oracle DB. All collections live in the 'zetaai' database.
"""

import os, secrets, hashlib, logging
from datetime import datetime, timedelta, timezone
from functools import wraps
from typing import Optional

import bcrypt
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import DuplicateKeyError

logger = logging.getLogger("zeta.db")

# ─── Connection ───────────────────────────────────────────────────────────────

_MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
_DB_NAME   = os.getenv("MONGO_DB",  "zetaai")

_client: Optional[MongoClient] = None
_db = None

def _get_db():
    global _client, _db
    if _db is None:
        _client = MongoClient(_MONGO_URI, serverSelectionTimeoutMS=5000)
        _db = _client[_DB_NAME]
    return _db

def bootstrap_schema():
    """Create indexes on first run."""
    db = _get_db()
    db.users.create_index("email",    unique=True)
    db.users.create_index("username", unique=True)
    db.sessions.create_index("token",      unique=True)
    db.sessions.create_index("expires_at", expireAfterSeconds=0)   # TTL index
    db.knowledge.create_index([("conv_id", ASCENDING), ("created_at", DESCENDING)])
    db.knowledge.create_index("user_id")
    db.conversations.create_index("user_id")
    db.api_keys.create_index("key_hash", unique=True, sparse=True)
    db.api_keys.create_index("user_id",  unique=True)
    db.payments.create_index("razorpay_payment_id", unique=True)
    logger.info("MongoDB indexes ensured.")

# ─── Users ────────────────────────────────────────────────────────────────────

def _hash_password(pw: str) -> str:
    return bcrypt.hashpw(pw.encode(), bcrypt.gensalt(int(os.getenv("BCRYPT_ROUNDS", 12)))).decode()

def _check_password(pw: str, hashed: str) -> bool:
    return bcrypt.checkpw(pw.encode(), hashed.encode())

def _fmt_user(doc: dict) -> dict:
    if not doc:
        return None
    return {
        "user_id":    str(doc["_id"]),
        "email":      doc["email"],
        "username":   doc["username"],
        "plan":       doc.get("plan", "basic"),
        "created_at": doc.get("created_at", datetime.utcnow()).isoformat(),
    }

def create_user(email: str, username: str, password: str) -> dict:
    db = _get_db()
    now = datetime.now(timezone.utc)
    doc = {
        "email":       email.lower(),
        "username":    username,
        "password_hash": _hash_password(password),
        "plan":        "basic",
        "created_at":  now,
        "updated_at":  now,
    }
    try:
        result = db.users.insert_one(doc)
        doc["_id"] = result.inserted_id
        return _fmt_user(doc)
    except DuplicateKeyError:
        raise ValueError("Email or username already taken.")

def authenticate_user(email: str, password: str) -> Optional[dict]:
    db = _get_db()
    doc = db.users.find_one({"email": email.lower()})
    if not doc or not _check_password(password, doc["password_hash"]):
        return None
    return _fmt_user(doc)

def get_user_by_id(user_id: str) -> Optional[dict]:
    from bson import ObjectId
    db = _get_db()
    try:
        doc = db.users.find_one({"_id": ObjectId(user_id)})
    except Exception:
        return None
    return _fmt_user(doc)

# ─── Sessions ─────────────────────────────────────────────────────────────────

def create_session(user_id: str) -> str:
    db  = _get_db()
    tok = secrets.token_urlsafe(48)
    exp = datetime.now(timezone.utc) + timedelta(hours=int(os.getenv("JWT_EXPIRY_HOURS", 24)))
    db.sessions.insert_one({"token": tok, "user_id": user_id, "expires_at": exp})
    return tok

def validate_session(token: str) -> Optional[str]:
    db  = _get_db()
    doc = db.sessions.find_one({"token": token})
    if not doc:
        return None
    if doc["expires_at"].replace(tzinfo=timezone.utc) < datetime.now(timezone.utc):
        db.sessions.delete_one({"token": token})
        return None
    return doc["user_id"]

def delete_session(token: str):
    _get_db().sessions.delete_one({"token": token})

# ─── Conversations ────────────────────────────────────────────────────────────

def create_conversation(user_id: str, title: str = "New Conversation", model: str = "zeta-4-turbo") -> str:
    db  = _get_db()
    doc = {
        "user_id":    user_id,
        "title":      title,
        "model":      model,
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
    }
    return str(db.conversations.insert_one(doc).inserted_id)

def get_user_conversations(user_id: str) -> list:
    db   = _get_db()
    docs = db.conversations.find({"user_id": user_id}).sort("updated_at", DESCENDING).limit(100)
    return [{"conv_id": str(d["_id"]), "title": d.get("title",""), "model": d.get("model",""),
             "created_at": d["created_at"].isoformat()} for d in docs]

def update_conversation_title(conv_id: str, title: str):
    from bson import ObjectId
    _get_db().conversations.update_one(
        {"_id": ObjectId(conv_id)},
        {"$set": {"title": title, "updated_at": datetime.now(timezone.utc)}}
    )

# ─── Knowledge / Messages ─────────────────────────────────────────────────────

def save_knowledge(question: str, answer: str, conv_id, user_id: str,
                   source: str = "ai", channel: str = "chat", tokens: int = 0) -> str:
    db  = _get_db()
    doc = {
        "question":   question,
        "answer":     answer,
        "conv_id":    conv_id,
        "user_id":    user_id,
        "source":     source,
        "channel":    channel,
        "tokens":     tokens,
        "rating":     0,
        "created_at": datetime.now(timezone.utc),
    }
    return str(db.knowledge.insert_one(doc).inserted_id)

def get_conversation_messages(conv_id: str) -> list:
    db   = _get_db()
    docs = db.knowledge.find({"conv_id": conv_id}).sort("created_at", ASCENDING)
    msgs = []
    for d in docs:
        msgs.append({"role": "user",      "content": d["question"]})
        msgs.append({"role": "assistant", "content": d["answer"]})
    return msgs

# ─── Feedback ─────────────────────────────────────────────────────────────────

def save_feedback(knowledge_id: str, rating: int, user_id: str, comment: str = ""):
    from bson import ObjectId
    db = _get_db()
    db.knowledge.update_one({"_id": ObjectId(knowledge_id)}, {"$inc": {"rating": rating}})
    db.feedback.insert_one({
        "knowledge_id": knowledge_id,
        "rating":       rating,
        "user_id":      user_id,
        "comment":      comment,
        "created_at":   datetime.now(timezone.utc),
    })

# ─── Subscriptions ────────────────────────────────────────────────────────────

def get_subscription(user_id: str) -> dict:
    db  = _get_db()
    doc = db.subscriptions.find_one({"user_id": user_id})
    if not doc:
        return {"plan": "basic", "billing_cycle": None, "status": "inactive",
                "started_at": None, "expires_at": None}
    return {
        "plan":          doc.get("plan", "basic"),
        "billing_cycle": doc.get("billing_cycle"),
        "status":        doc.get("status", "inactive"),
        "started_at":    doc.get("started_at", datetime.utcnow()).isoformat() if doc.get("started_at") else None,
        "expires_at":    doc.get("expires_at").isoformat() if doc.get("expires_at") else None,
    }

def upgrade_subscription(user_id: str, plan: str, cycle: str, order_id: str):
    from bson import ObjectId
    db  = _get_db()
    now = datetime.now(timezone.utc)
    exp = now + (timedelta(days=365) if cycle == "yearly" else timedelta(days=30))
    db.subscriptions.update_one(
        {"user_id": user_id},
        {"$set": {"plan": plan, "billing_cycle": cycle, "status": "active",
                  "order_id": order_id, "started_at": now, "expires_at": exp}},
        upsert=True
    )
    db.users.update_one({"_id": ObjectId(user_id)}, {"$set": {"plan": plan}})

def cancel_subscription(user_id: str):
    from bson import ObjectId
    db = _get_db()
    db.subscriptions.update_one({"user_id": user_id},
                                 {"$set": {"plan": "basic", "status": "cancelled"}})
    db.users.update_one({"_id": ObjectId(user_id)}, {"$set": {"plan": "basic"}})

# ─── Payments ─────────────────────────────────────────────────────────────────

def save_payment(user_id: str, payment_id: str, order_id: str,
                 signature: str, amount: int, plan: str, cycle: str):
    _get_db().payments.insert_one({
        "user_id":              user_id,
        "razorpay_payment_id":  payment_id,
        "razorpay_order_id":    order_id,
        "razorpay_signature":   signature,
        "amount":               amount,
        "plan":                 plan,
        "billing_cycle":        cycle,
        "status":               "captured",
        "created_at":           datetime.now(timezone.utc),
    })

# ─── API Keys ─────────────────────────────────────────────────────────────────

def generate_api_key(user_id: str, label: str = "Default Key") -> dict:
    db       = _get_db()
    raw_key  = "zeta_live_" + secrets.token_urlsafe(32)
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
    now      = datetime.now(timezone.utc)
    db.api_keys.update_one(
        {"user_id": user_id},
        {"$set": {"key_hash": key_hash, "label": label,
                  "revoked": False, "created_at": now, "last_used": None}},
        upsert=True
    )
    return {"key": raw_key, "label": label, "created_at": now.isoformat()}

def revoke_api_key(user_id: str):
    _get_db().api_keys.update_one({"user_id": user_id}, {"$set": {"revoked": True}})

def validate_api_key(raw_key: str) -> Optional[dict]:
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
    db  = _get_db()
    doc = db.api_keys.find_one({"key_hash": key_hash, "revoked": False})
    if not doc:
        return None
    db.api_keys.update_one({"_id": doc["_id"]}, {"$set": {"last_used": datetime.now(timezone.utc)}})
    return {"user_id": doc["user_id"]}

def get_api_key_info(user_id: str) -> Optional[dict]:
    doc = _get_db().api_keys.find_one({"user_id": user_id, "revoked": False})
    if not doc:
        return None
    return {"label": doc.get("label"), "created_at": doc.get("created_at","").isoformat()
            if doc.get("created_at") else None, "last_used": doc.get("last_used")}

# ─── Stats ────────────────────────────────────────────────────────────────────

def get_stats() -> dict:
    db = _get_db()
    return {
        "users":         db.users.count_documents({}),
        "conversations": db.conversations.count_documents({}),
        "knowledge":     db.knowledge.count_documents({}),
        "pro_users":     db.users.count_documents({"plan": "pro"}),
        "feedback":      db.feedback.count_documents({}),
    }
