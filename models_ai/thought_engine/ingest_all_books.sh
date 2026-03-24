#!/usr/bin/env bash
# Ingest all trading/finance books into the thought engine.
# Run on a CUDA-enabled system with Ollama for best performance.
#
# Usage:
#   ./ingest_all_books.sh [--model MODEL] [--expert NAME]
#
# Prerequisites:
#   ollama pull mistral:latest    (or your preferred model)
#   ollama pull nomic-embed-text  (for embeddings)
#   pip install -r requirements.txt

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BOOKS_DIR="${SCRIPT_DIR}/../thought_engine_data/books"
MODEL="${1:-mistral:latest}"
EXPERT="${2:-finance}"
EMBED_MODEL="nomic-embed-text"

# Find project root (two levels up from thought_engine/)
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

echo "============================================"
echo "Thought Engine — Batch Book Ingestion"
echo "============================================"
echo "Model:      ${MODEL}"
echo "Embed:      ${EMBED_MODEL}"
echo "Expert:     ${EXPERT}"
echo "Books dir:  ${BOOKS_DIR}"
echo ""

# Check Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "ERROR: Ollama is not running. Start it with: ollama serve"
    exit 1
fi

# Check models are available
echo "Checking models..."
ollama list | grep -q "${MODEL}" || { echo "ERROR: Model ${MODEL} not found. Run: ollama pull ${MODEL}"; exit 1; }
ollama list | grep -q "${EMBED_MODEL}" || { echo "ERROR: Model ${EMBED_MODEL} not found. Run: ollama pull ${EMBED_MODEL}"; exit 1; }
echo "Models OK."
echo ""

# Download books if not present
download_book() {
    local url="$1"
    local filename="$2"
    local path="${BOOKS_DIR}/${filename}"
    if [ ! -f "$path" ]; then
        echo "Downloading ${filename}..."
        mkdir -p "${BOOKS_DIR}"
        curl -sL "$url" -o "$path"
    fi
}

echo "Ensuring all books are downloaded..."
download_book "https://www.gutenberg.org/cache/epub/3300/pg3300.txt" "wealth_of_nations.txt"
download_book "https://www.gutenberg.org/cache/epub/24518/pg24518.txt" "madness_of_crowds.txt"
download_book "https://www.gutenberg.org/cache/epub/75570/pg75570.txt" "psychology_of_stock_market.txt"
download_book "https://www.gutenberg.org/cache/epub/73647/pg73647.txt" "psychology_of_speculation.txt"
download_book "https://www.gutenberg.org/cache/epub/26841/pg26841.txt" "successful_stock_speculation.txt"
download_book "https://www.gutenberg.org/cache/epub/44052/pg44052.txt" "profitable_stock_exchange_investments.txt"
download_book "https://www.gutenberg.org/cache/epub/60979/pg60979.txt" "reminiscences_stock_operator.txt"
download_book "https://www.gutenberg.org/cache/epub/59518/pg59518.txt" "theory_of_stock_exchange_speculation.txt"
download_book "https://www.gutenberg.org/cache/epub/54130/pg54130.txt" "wall_street_stories.txt"
download_book "https://www.gutenberg.org/cache/epub/59042/pg59042.txt" "the_stock_exchange.txt"
download_book "https://www.gutenberg.org/cache/epub/70377/pg70377.txt" "fifty_years_wall_street.txt"
echo "All books ready."
echo ""

# Ingest books in order: shortest first for quick progress
BOOKS=(
    "profitable_stock_exchange_investments.txt"
    "successful_stock_speculation.txt"
    "psychology_of_stock_market.txt"
    "psychology_of_speculation.txt"
    "wall_street_stories.txt"
    "the_stock_exchange.txt"
    "theory_of_stock_exchange_speculation.txt"
    "reminiscences_stock_operator.txt"
    "madness_of_crowds.txt"
    "wealth_of_nations.txt"
    "fifty_years_wall_street.txt"
)

TOTAL=${#BOOKS[@]}
COUNT=0
START_TIME=$(date +%s)

for book in "${BOOKS[@]}"; do
    COUNT=$((COUNT + 1))
    BOOK_PATH="${BOOKS_DIR}/${book}"

    if [ ! -f "$BOOK_PATH" ]; then
        echo "[${COUNT}/${TOTAL}] SKIP: ${book} (not found)"
        continue
    fi

    echo ""
    echo "============================================"
    echo "[${COUNT}/${TOTAL}] Reading: ${book}"
    echo "============================================"

    BOOK_START=$(date +%s)

    cd "${PROJECT_ROOT}" && python -m models_ai.thought_engine read \
        "${BOOK_PATH}" \
        --expert "${EXPERT}" \
        --model "${MODEL}" \
        --embed-model "${EMBED_MODEL}" \
    || echo "WARNING: ${book} had errors (partial ingestion)"

    BOOK_END=$(date +%s)
    BOOK_ELAPSED=$((BOOK_END - BOOK_START))
    echo "  Completed in $((BOOK_ELAPSED / 60))m $((BOOK_ELAPSED % 60))s"
done

END_TIME=$(date +%s)
TOTAL_ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "============================================"
echo "Ingestion complete!"
echo "Total time: $((TOTAL_ELAPSED / 60))m $((TOTAL_ELAPSED % 60))s"
echo "============================================"
echo ""

# Show final stats
cd "${PROJECT_ROOT}" && python -m models_ai.thought_engine stats --expert "${EXPERT}"

echo ""
echo "Start chatting: python -m models_ai.thought_engine chat --expert ${EXPERT}"
