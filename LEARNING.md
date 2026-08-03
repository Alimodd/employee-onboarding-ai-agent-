# Learning Notes

Educational notes for the Employee Onboarding AI Assistant. Each section records
what was built, how it flows, and its limitations.

---

## Prompt-only knowledge (HR FAQ bot)

**What was built:** A small HR FAQ bot that answers questions using a hard-coded,
fictional HR policy injected directly into the model prompt. No documents,
embeddings, or vector database are used — the policy is a Python string in
[`hr_faq.py`](hr_faq.py).

- **Input:** A user question (`POST /chat`, JSON `{ "message": "..." }`).
- **Output:** `{ "answer": "...", "mode": "prompt_only_faq", "status": "success" }`.
- **External API:** Google Gemini text generation via `model_client.generate_response`.
- **Data stored:** None. The policy is static in source; nothing is persisted.
- **Failure cases:** Missing/blank message (422), missing API key (503), provider
  auth failure (503), provider/empty errors (502), unexpected error (500).
- **Main limitation:** The whole policy must fit in every prompt. This does not
  scale: a real handbook is far too large to paste into each request, and the model
  can still phrase answers loosely. It also cannot cite a source document.
- **What comes next:** Store real documents and retrieve only the relevant parts
  (retrieval-augmented generation) instead of pasting a fixed policy.

The bot is instructed to answer only from the policy, refuse unknown questions
with a fixed fallback line, and correct false assumptions rather than agree with
them.

---

## Embeddings and similarity

**What was built:** A standalone script,
[`embedding_experiment.py`](embedding_experiment.py), that embeds ~10 short
sentences, picks one as a query, and ranks the rest by cosine similarity. The
cosine math lives in [`similarity.py`](similarity.py) so it can be unit-tested.

- **Input:** A fixed list of sentences defined in the script.
- **Output:** A printed rank / score / sentence table, highest similarity first.
- **External API:** Google Gemini embeddings (`gemini-embedding-001`) via
  `client.models.embed_content`.
- **Data stored:** None. Embeddings are computed in memory and discarded.
- **Failure cases:** Missing API key, embedding API failure, empty response,
  unexpected embedding format, empty sentence, mismatched dimensions.
- **Main limitation:** Similarity is about *meaning overlap*, not truth. A high
  score does not prove two sentences mean the same thing.

Key ideas:

- An **embedding** is a numeric vector representation of text.
- Sentences with similar meaning tend to produce nearby vectors.
- Semantic similarity is **not** keyword matching — differently worded sentences
  can score high, and sentences sharing words can score low.
- A high similarity score does **not** prove two statements mean the same thing.
- **Negation is a trap:** "Employees cannot work remotely" shares almost every
  word with "Employees can work remotely," so it often scores high despite meaning
  the opposite.

**What comes next:** Use embeddings to retrieve relevant document chunks for the
FAQ bot, replacing the hard-coded policy prompt.

---

## Source-document quality

**What was built:** A fully synthetic company knowledge base for the fictional
company *Northstar Analytics*, in [`company_docs/`](company_docs), plus a
machine-readable [`manifest.json`](company_docs/manifest.json).

- **Input:** None at runtime — these are static source files.
- **Output:** Seven `.txt` policy documents and one manifest describing them.
- **External API:** None.
- **Data stored:** The documents and manifest on disk (synthetic content only).
- **Failure cases (guarded by tests):** missing files, manifest/file mismatch,
  duplicate IDs, missing metadata, empty documents, inconsistent version/date,
  and any contradiction with the FAQ policy facts.
- **Main limitation:** These are only source documents. They are **not** chunked,
  embedded, indexed, or retrievable yet.

**What comes next:** Chunk the documents, embed the chunks, store them in a vector
database, and retrieve them to ground the assistant's answers (RAG). None of that
is implemented yet — this step only establishes clean, consistent source material.
