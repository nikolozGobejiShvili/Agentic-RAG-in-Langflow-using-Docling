# 1) პროექტის root
Set-Location "C:\Users\Greench Pc\Desktop\rag_langflow_docling"

# 5) Env ცვლადები ამ სესიაზე (შეიყვანე შენი რეალური key)
$env:LANGFLOW_AUTO_LOGIN = "True"
$env:GROQ_API_KEY = ""

# 6) Langflow გაშვება custom components-ით
python -m langflow run --components-path ".\components" --host 0.0.0.0 --port 7860
