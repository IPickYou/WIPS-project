# utils/text_chunker.py
import pandas as pd
class SlidingWindowChunker:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
    def chunk_text(self, text, max_length=512, stride=50):
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        chunks = []
        start = 0

        while start < len(tokens):
            end = min(start + max_length, len(tokens))
            chunks.append(self.tokenizer.decode(tokens[start:end], skip_special_tokens=True))

            if end == len(tokens):
                break

            start += max_length - stride

        return chunks

    def create_chunked_dataset(self, df, max_length=512, stride=50):
        chunked_rows = []

        for _, row in df.iterrows():
            chunks = self.chunk_text(row["text"], max_length, stride)

            for chunk in chunks:
                chunk_row = {
                    "text": chunk,
                    "patent_id": row["patent_id"]
                }
                if "label" in row:
                    chunk_row["label"] = row["label"]
                chunked_rows.append(chunk_row)

        print(f"chunked! {chunked_rows[:1]}")

        return pd.DataFrame(chunked_rows)