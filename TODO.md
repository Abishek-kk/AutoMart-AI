# Supermarket-AI Deployment Fix TODO

## Status: Steps 1-2 Complete

### 1. [COMPLETE] Initial update requirements.txt (revised for wheels)\n   - pandas==2.0.3, numpy==1.26.4, scikit-learn==1.5.2, remove torch line

### 2. [COMPLETE] Create runtime.txt
   - Specify Python 3.12.3 for optimal wheels.

### 3. [IN PROGRESS] Local test\n   - pip install -r requirements.txt --only-binary=:all: (Cloud sim)\n   - streamlit run

### 4. [PENDING] Commit & Deploy
   - Git add/commit/push.
   - Redeploy on Streamlit Cloud.

### 5. [PENDING] Verify
   - Check app loads, models/agents work.

