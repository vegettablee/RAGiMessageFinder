import sqlite3
import os
from typing import Optional

DB_ABS_PATH = "/Users/prestonrank/RAGMessages/data/raw/chat.db"
conn = sqlite3.connect(DB_ABS_PATH)

def _normalize_digits(s: str) -> str:
    return "".join(ch for ch in s if ch.isdigit())

def _subject_e164ish(subject: str) -> str:
    d = _normalize_digits(subject)
    if len(d) == 10:
        return "+1" + d
    if len(d) == 11 and d.startswith("1"):
        return "+" + d
    if subject.strip().startswith("+"):
        return subject.strip()
    return "+" + d if d else subject.strip()

def _decode_attributed_body(blob: Optional[bytes]) -> str:
    """
    Extract human text from iMessage attributedBody blobs.

    Pattern seen in macOS archives:
      ... b'+' <len_byte> <utf8_text_bytes> b'\x02' ... (sometimes \x0c)
    We:
      1) Look for '+' and treat the next byte as a length hint.
      2) Slice until a control terminator (\x02 or \x0c).
      3) UTF-8 decode (ignore errors).
      4) Fallback to a generic cleanup if the pattern isn't found.
    """
    if not blob:
        return ""

    b = bytes(blob)

    # Primary parse: '+' <len> <text> until \x02 or \x0c
    plus = b.find(b'+')
    if plus != -1 and plus + 2 < len(b):
        # The byte after '+' is typically a length; skip it and capture until a control byte.
        start = plus + 2
        end = start
        while end < len(b) and b[end] not in (0x02, 0x0c):
            end += 1
        candidate = b[start:end]
        txt = candidate.decode("utf-8", errors="ignore").strip()
        if txt:
            return txt

    # Secondary parse: scan for any "+...<ctrl>" spans, in case multiple segments exist.
    i = 0
    best = ""
    while i < len(b):
        if b[i] == 0x2B and i + 2 < len(b):  # '+'
            start = i + 2
            j = start
            while j < len(b) and b[j] not in (0x02, 0x0c):
                j += 1
            cand = b[start:j].decode("utf-8", errors="ignore").strip()
            if len(cand) > len(best):
                best = cand
            i = j + 1
        else:
            i += 1
    if best:
        return best

    # Fallback: very rough cleanup when no '+' pattern is found.
    s = blob.decode("utf-8", errors="ignore")
    # Strip common archive keys to surface the human text.
    for kw in ("NSString", "NSDictionary", "NSNumber", "__kIMMessagePartAttributeName"):
        s = s.replace(kw, "")
    s = s.replace("\x00", " ")
    s = " ".join(s.split())
    return s.strip()


def getMessagesBySubject(subject: str, numberOfMessages: int):
    """
    Returns a list of tuples:
      (sent_at, 'me' or '', '', text, <normalized subject phone>)
    - Includes BOTH incoming and outgoing.
    - Excludes attachments and reactions.
    - Grabs latest N, outputs in chronological order.
    - Outgoing bodies are recovered from attributedBody when text is NULL.
    """
    c = conn.cursor()

    like_text = f"%{subject}%"
    digits = _normalize_digits(subject)
    ten = digits[-10:] if len(digits) >= 10 else digits
    like_10 = f"%{ten}%" if ten else ""
    like_11 = f"%1{ten}%" if ten else ""

    subject_out = _subject_e164ish(subject)

    query = f"""
    WITH target_chats AS (
      SELECT DISTINCT c.ROWID AS chat_id
      FROM chat c
      LEFT JOIN chat_handle_join chj ON chj.chat_id = c.ROWID
      LEFT JOIN handle h ON h.ROWID = chj.handle_id
      WHERE
        IFNULL(h.id,'') LIKE ? COLLATE NOCASE
        OR IFNULL(c.display_name,'') LIKE ? COLLATE NOCASE
        OR IFNULL(c.chat_identifier,'') LIKE ? COLLATE NOCASE
        OR REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(IFNULL(h.id,''), ' ', ''), '-', ''), '(', ''), ')', ''), '+', '') LIKE ?
        OR REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(IFNULL(c.chat_identifier,''), ' ', ''), '-', ''), '(', ''), ')', ''), '+', '') LIKE ?
        {"OR REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(IFNULL(h.id,''), ' ', ''), '-', ''), '(', ''), ')', ''), '+', '') LIKE ? OR REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(IFNULL(c.chat_identifier,''), ' ', ''), '-', ''), '(', ''), ')', ''), '+', '') LIKE ?" if ten else ""}
    ),
    lastN AS (
      SELECT
        m.ROWID AS msg_id,
        m.date AS raw_date,
        datetime(m.date/1000000000 + 978307200, 'unixepoch','localtime') AS sent_at,
        m.is_from_me,
        m.text,
        m.attributedBody
      FROM message m
      JOIN chat_message_join cmj ON cmj.message_id = m.ROWID
      JOIN target_chats tc ON tc.chat_id = cmj.chat_id
      WHERE
        -- Exclude any message that actually has attachments
        NOT EXISTS (SELECT 1 FROM message_attachment_join maj WHERE maj.message_id = m.ROWID)
        -- Exclude reactions/tapbacks
        AND IFNULL(m.associated_message_type, 0) = 0
      ORDER BY m.date DESC
      LIMIT ?
    )
    SELECT msg_id, raw_date, sent_at, is_from_me, text, attributedBody
    FROM lastN
    ORDER BY raw_date ASC;
    """

    params = [like_text, like_text, like_text, like_10, like_10]
    if ten:
        params.extend([like_11, like_11])
    params.append(int(numberOfMessages))

    rows = list(c.execute(query, tuple(params)))

    out = []
    for (_msg_id, _raw, sent_at, is_from_me, text, attrib) in rows:
        body = (text or "").strip()
        if not body:
            body = _decode_attributed_body(attrib)
        # Skip if still empty (rare system rows)
        if not body:
            continue
        sender_label = "me" if is_from_me == 1 else ""
        tup = (sent_at, sender_label, "", body, subject_out)
        print(tup)
        out.append(tup)

    return out

# Example
# getMessagesBySubject("4699106057", 50)
