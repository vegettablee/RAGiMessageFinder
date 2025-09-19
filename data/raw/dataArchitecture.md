# macOS Messages History Architecture

This document explains how macOS stores and structures your Messages (iMessage + SMS) history.

---

## 1. Where the data lives (local)

- **Main DB (SQLite):** `~/Library/Messages/chat.db`  
  - Protected by **Full Disk Access** (TCC).
  - SQLite in WAL mode (`chat.db`, `chat.db-wal`, `chat.db-shm`).

- **Attachments (files on disk):** `~/Library/Messages/Attachments/`  
  - Nested hex-like directories with images, videos, audio, vCards, etc.
  - Database references file paths/filenames; bytes live here.

---

## 2. How data gets there (ingest/transport)

- **iMessage:** Delivered via Apple’s IDS/APNs stack. End-to-end encrypted in transit. The Mac stores plaintext in `chat.db` and saves attachments locally.
- **SMS/MMS:** If **Text Message Forwarding** is enabled, the iPhone relays SMS/MMS to the Mac; they land in the same DB with `service_name='SMS'`.
- **Messages in iCloud (optional):** Syncs via CloudKit (end-to-end encrypted). Local DB still holds a full copy.

---

## 3. Data model (core tables & relations)

### Entities
- **`message`** — one row per message.  
  Key fields:
  - `ROWID`, `guid`
  - `date` (nanoseconds since 2001-01-01 UTC)
  - `is_from_me` (0/1)
  - `text` (nullable)
  - `handle_id` (FK to sender)
  - `cache_has_attachments`
  - `associated_message_guid`, `associated_message_type` (tapbacks, inline replies)
  - `thread_originator_guid` (for threaded replies)
  - `date_delivered`, `date_read`

- **`handle`** — one row per contact address (phone/email).  
  - `ROWID`, `id`

- **`chat`** — one row per conversation (1:1 or group).  
  - `ROWID`, `guid`, `chat_identifier`, `service_name`, `display_name`

### Join Tables
- **`chat_message_join`** — links messages ↔ chats  
- **`chat_handle_join`** — links chats ↔ participants  
- **`message_attachment_join`** — links messages ↔ attachments  

### Attachments
- **`attachment`** — metadata for files.  
  - `ROWID`, `filename`, `transfer_name`, `mime_type`, `total_bytes`

---

## 4. Content formats and quirks

- **Timestamps:**  
  Stored as nanoseconds since 2001-01-01.  
  Convert in SQL:  
  ```sql
  datetime(date/1000000000 + 978307200, 'unixepoch','localtime')
