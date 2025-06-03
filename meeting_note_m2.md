# Meeting Note – Milestone 2  
**Date:** 2025-06-02  
**Duration:** 25 min (09:00–09:25 PT)  
**Attendees:**  
- Me – Dev / Presenter  
- Alice – Client PM  
- Prof. Lee – Faculty Supervisor (observer)

---

## Agenda  
1. Walk-through of new TF-IDF recommender & sidebar **Price Slider**  
2. Review updated README & unit-test coverage  
3. Agree on scope for Milestone 3

---

## Demo Highlights  
- **Live demo**: searched “red dress under $100” → 3 products returned within slider range.  
- **Unit tests**: `pytest -q` shows **3 passed** (engine, parser, price filter).  
- **README** now includes full setup, GIF, progress table.

---

## Client Feedback  
| Item | Feedback | Action |
|------|----------|--------|
| Price slider UX | Slider default range OK; maybe add currency symbol in label. | Minor tweak (M3 backlog) |
| Recommendation quality | Results feel relevant; eager to see vector DB upgrade. | Plan in M3 |
| Documentation | “README is clear; like the GIF.” | Keep up-to-date |

---

## Decisions  
- **PR #5** approved by Alice at 09:18 PT.  
- **Issue #4** auto-closed on merge.  
- Vector-DB (FAISS / Chroma) upgrade scheduled for Milestone 3.  
- Next check-in: **2025-06-16** (Milestone 3 mid-review).

---
