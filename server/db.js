// db.js
const Database = require("better-sqlite3");
const path = require("path");

const db = new Database(path.join(__dirname, "opsai.db"));
db.pragma("journal_mode = WAL");
db.pragma("foreign_keys = ON");

db.exec(`
  CREATE TABLE IF NOT EXISTS meetings (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    original_file_url TEXT NOT NULL,
    original_file_path TEXT DEFAULT '',
    file_name TEXT NOT NULL,
    file_size INTEGER NOT NULL DEFAULT 0,
    processing_method TEXT NOT NULL,
    validate_summary INTEGER NOT NULL DEFAULT 1,
    status TEXT NOT NULL DEFAULT 'uploaded',
    progress INTEGER NOT NULL DEFAULT 0,
    created_date TEXT NOT NULL,
    created_by TEXT NOT NULL DEFAULT 'You',
    markdown_output TEXT NOT NULL DEFAULT ''
  );
`);

// Migration: add validate_summary column if it doesn't exist (for existing databases)
try {
  db.exec(`ALTER TABLE meetings ADD COLUMN validate_summary INTEGER NOT NULL DEFAULT 1;`);
} catch (e) {
  // Column already exists, ignore error
}

module.exports = db;