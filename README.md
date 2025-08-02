# Lab 3: Penguins Classification 

This project builds a robust ML pipeline to classify penguin species. It includes data preprocessing, XGBoost model training, and deployment with FastAPI. All inputs are validated and the API is fully documented.

---

## 1. Clone and Set Up Environment

```bash
git clone https://github.com/aidi-2004-ai-enterprise/lab03_Aromal_Gigi.git
cd lab3_Aromal_Gigi
```
```bash
uv init
uv venv .venv                        # Windows PowerShell
.\.venv\Scripts\Activate.ps1         # Windows PowerShell

uv pip install -r requirements.txt
```

## 2. Train the Model

```bash
python train.py
This saves the model and metadata to app/data/.
```

## 3. Launch the API
```bash
uv run uvicorn app.main:app --reload
```

- **API Docs:** http://127.0.0.1:8000/docs

- **Health Check:** http://127.0.0.1:8000/health

- **Root Greeting:** http://127.0.0.1:8000/

## 4. API Usage Examples
Valid Request
With cURL (Windows Command Prompt/Terminal):

```bash
curl -X POST "http://127.0.0.1:8000/predict" ^
  -H "Content-Type: application/json" ^
  -d "{\"bill_length_mm\":39.1,\"bill_depth_mm\":18.7,\"flipper_length_mm\":181,\"body_mass_g\":3750,\"year\":2007,\"sex\":\"male\",\"island\":\"Torgersen\"}"
```

With PowerShell:

```bash
$body = @{
  bill_length_mm    = 39.1
  bill_depth_mm     = 18.7
  flipper_length_mm = 181
  body_mass_g       = 3750
  year              = 2007
  sex               = "male"
  island            = "Torgersen"
} | ConvertTo-Json

Invoke-RestMethod `
  -Uri http://127.0.0.1:8000/predict `
  -Method POST `
  -ContentType "application/json" `
  -Body $body

```
Expected Response:

```bash
{"species":"Adelie"}
```

Invalid Input: Wrong Sex
```bash
curl -X POST "http://127.0.0.1:8000/predict" ^
  -H "Content-Type: application/json" ^
  -d "{\"bill_length_mm\":39.1,\"bill_depth_mm\":18.7,\"flipper_length_mm\":181,\"body_mass_g\":3750,\"year\":2007,\"sex\":\"shemale\",\"island\":\"Torgersen\"}"
```
Expected Response:

```bash
{
  "detail": [
    {
      "type": "enum",
      "loc": ["body","sex"],
      "msg": "Input should be 'male' or 'female'",
      "input": "shemale",
      "ctx": {"expected": "'male' or 'female'"}
    }
  ]
}
```
Invalid Input: Wrong Island
```bash
curl -X POST "http://127.0.0.1:8000/predict" ^
  -H "Content-Type: application/json" ^
  -d "{\"bill_length_mm\":39.1,\"bill_depth_mm\":18.7,\"flipper_length_mm\":181,\"body_mass_g\":3750,\"year\":2007,\"sex\":\"male\",\"island\":\"Australia\"}"
```
Expected Response:

```bash
{
  "detail": [
    {
      "type": "enum",
      "loc": ["body","island"],
      "msg": "Input should be 'Torgersen', 'Biscoe' or 'Dream'",
      "input": "Australia",
      "ctx": {"expected": "'Torgersen', 'Biscoe' or 'Dream'"}
    }
  ]
}
```

## 5. Demo Video
- A demo video (`Demo.mp4`) is included, demonstrating:
  - Running training and launching the server
  - Sending valid and invalid requests (for both `sex` & `island`)
    

## 6. Dependencies

This project uses the following open-source libraries and tools:

- **FastAPI** — for building the REST API
- **Uvicorn** — lightning-fast ASGI server for FastAPI
- **pydantic** — for data validation
- **XGBoost** — for gradient boosting classification
- **scikit-learn** — for preprocessing, train/test split, and evaluation
- **pandas** — for data wrangling and one‑hot encoding
- **seaborn** — for loading the Palmer Penguins dataset
- **uv** — for fast, modern Python dependency management
- **curl** — for API testing (example requests)
