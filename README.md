# My DecodeLabs AI & Machine Learning Journey

> **Author:** Emmanuel Nkaka
> **Programme:** DecodeLabs · Artificial Intelligence

A brief reflection on four AI/ML systems I built and the lessons each one taught me.

---

## 1. Tech Stack Recommender — *AI Recommendation Logic*

**Built:** A pure-Python content-based recommender mapping skills to tech careers using TF-IDF and cosine similarity.

**Learned:**
- The Input → Process → Output pipeline
- Feature extraction and vector mapping
- TF-IDF weighting (penalises generic skills, rewards specific ones)
- Cosine similarity as a percentage match
- Cold-start handling and Top-N ranking

**Takeaway:** AI is scoring + ranking + graceful failure — discipline, not magic.

---

## 2. Rule-Based AI Chatbot

**Built:** A dictionary-driven chatbot with input normalisation and exit handling.

**Learned:**
- The anatomy of a rule-based system
- Why input normalisation is critical
- The limits of rules — and why machine learning exists
- Conversation design and fallbacks

**Takeaway:** Rules are the foundation of AI; their boundaries are where ML begins.

---

## 3. Iris Classification with KNN

**Built:** A full supervised ML workflow — load, split, scale, tune, train, evaluate, visualise, predict.

**Learned:**
- The complete ML pipeline
- Feature scaling with `StandardScaler`
- K-value tuning via the Elbow Method
- Evaluation beyond accuracy (F1, confusion matrix, classification report)
- Visualisation as diagnostics

**Takeaway:** Supervised learning is a rigorous workflow — evaluation is the heart of it.

---

## 4. Image & Text Recognition

**Built:** A GUI + CLI app combining OpenCV object detection (MobileNet SSD) and Tesseract OCR, packaged as a Windows `.exe`.

**Learned:**
- Two AI modalities: vision and OCR
- Pretrained models as building blocks
- Productionisation (requirements, packaging, PyInstaller)
- Real-world dependency fragility
- Serving one model through two interfaces

**Takeaway:** A working model is only half the job — deployment is the final mile.

---

## Cross-Cutting Lessons

| Theme | Where I saw it |
|-------|----------------|
| Data quality beats model complexity | Recommender, Iris, OCR |
| Evaluation is non-negotiable | Unit tests, F1, confusion matrix |
| Graceful failure is a feature | Cold-start, chatbot fallback, requirement checks |
| Determinism vs. learning | Chatbot vs. KNN |
| From scratch → libraries → deployment | Pure Python → scikit-learn → `.exe` |

---

## How My Understanding Evolved

**Before:** AI was a black box.
**After:** AI is a pipeline — normalise, represent, score, learn, evaluate, deploy.

| Project | Lesson |
|---------|--------|
| Recommender | AI is *scoring and ranking* |
| Chatbot | AI is *rules and patterns* |
| KNN | AI is *learning from data* |
| OpticNerve | AI is *systems engineering* |

---

## Acknowledgements

Thanks to **DecodeLabs** for a curriculum that moves from logic → rules → learning → perception, and insists on building from scratch first.

---

**Emmanuel Nkaka** · DecodeLabs AI and machine learning Programme