import sqlite3

con = sqlite3.connect("results.db")

con.execute("CREATE TABLE IF NOT EXISTS answers ("
"id INTEGER PRIMARY KEY, q TEXT, a TEXT, cost_usd REAL)")
con.execute("INSERT INTO answers (q, a, cost_usd) VALUES (?, ?, ?)",
("What is RAG?", "...", 0.0001))
con.commit()