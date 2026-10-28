# Таймкод — семантический поиск по видеолекциям

## Запуск
1. Установить ffmpeg (`sudo apt install ffmpeg` / `brew install ffmpeg` / choco install ffmpeg)
2. `python -m venv .venv && source .venv/bin/activate` (Windows: `.venv\Scripts\activate`)
3. `pip install -r requirements.txt`
4. `uvicorn main:app --reload` → http://127.0.0.1:8000

## Настройки (переменные окружения, все необязательные)
- `WHISPER_MODEL` — tiny / base / small (по умолчанию) / medium. Для быстрых тестов берите `tiny`.
- `EMB_MODEL` — модель эмбеддингов (по умолчанию multilingual MiniLM, понимает русский).
- `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `LLM_MODEL` — для конспекта и оглавления через LLM.
  Без ключа строится упрощённый конспект.

## Как устроено
Видео → ffmpeg (аудио) → faster-whisper (текст + таймкоды) → фрагменты ~25 с → эмбеддинги в SQLite →
поиск по косинусной близости. Фронтенд — один файл `static/index.html`.
