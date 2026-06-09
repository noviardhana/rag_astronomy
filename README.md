# Research Navigator for Astronomy: RAG-Based Assistant with NASA ADS

## Deskripsi Proyek

Proyek ini bertujuan mengembangkan sistem **Retrieval-Augmented Generation (RAG)** berbasis **NASA Astrophysics Data System (ADS)** untuk mendukung peneliti, mahasiswa, serta penggiat astronomi dalam menavigasi dan memahami literatur ilmiah.  

Tantangan utama yang dihadapi adalah banyaknya publikasi astronomi yang harus ditelaah secara manual, yang memakan waktu dan berpotensi menghasilkan interpretasi yang tidak menyeluruh.  

Sebagai solusi, sistem RAG ini memanfaatkan basis data **NASA ADS** untuk:

- Melakukan pencarian publikasi secara otomatis.
- Menghasilkan ringkasan terstruktur.
- Memberikan jawaban atas pertanyaan spesifik.

Dengan demikian, proyek ini diharapkan dapat mempercepat proses **literature review**, memfasilitasi eksplorasi topik riset, dan meningkatkan aksesibilitas pengetahuan astronomi bagi komunitas riset global.

---

## Tujuan Proyek

1. **Membangun prototipe sistem RAG** yang mampu melakukan:
   - Question answering berbasis abstrak paper.
   - Summarization dari publikasi NASA ADS.

2. **Menyediakan pipeline end-to-end** untuk:
   - Querying
   - Retrieval
   - LLM-based generation

3. **Mengevaluasi performa awal prototipe** melalui:
   - Kualitas ringkasan (perbandingan sederhana dengan abstrak asli).
   - Relevansi jawaban terhadap pertanyaan uji (evaluasi manual terbatas).

```text

## Struktur Proyek

\final_project

├─ app.py

├─ from_papers_to_answers_implementing_rag_for_astronomy_knowledge_extraction.py

├─ rag_utils.py

├─ astronomy_dataset_2440rows_20251031.csv

├─ requirements.txt

├─ chroma_db/

├─ venv/

├─ pycache/

└─ From_Papers_to_Answers_Implementing_RAG_for_Astronomy_Knowledge_Extraction.ipynb

```

## Persiapan & Instalasi

1. **Buat virtual environment** (Windows):

```bash

1. **Buat virtual environment** (Windows):

D:
cd D:\Sanbercode\otomasi\final_project
python -m venv venv
venv\Scripts\activate


2. Install dependencies:
pip install -r requirements.txt


3. Jalankan aplikasi (misal Streamlit):
streamlit run app.py
```

--- 

### Kontak

Proyek ini dikembangkan sebagai prototipe final project untuk eksplorasi RAG berbasis literatur astronomi.

Untuk pertanyaan lebih lanjut, silakan hubungi pengembang.


