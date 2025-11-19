Remove-Item -Recurse -Force .\vectorstore -ErrorAction SilentlyContinue
python .\scripts\ingest.py
python .\scripts\run_agent.py