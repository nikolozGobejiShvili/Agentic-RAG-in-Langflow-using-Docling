import argparse, json, os
from typing import Optional
import requests

def _print_chat_output(resp: dict) -> None:
    content = None
    for out in resp.get("outputs", []):
        for od in out.get("outputs", []):
            if od.get("type") == "chat":
                msgs = od.get("messages", [])
                if msgs:
                    content = msgs[-1].get("data", {}).get("content")
    print(content or json.dumps(resp, ensure_ascii=False, indent=2))

def run_client(flow_id: str, base_url: str = "http://localhost:7860") -> None:
    print(f"Langflow: {base_url} | Flow: {flow_id}")
    while True:
        try:
            q = input("\nQuestion (empty to quit): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye"); break
        if not q:
            break
        url = f"{base_url}/api/v1/run/{flow_id}"
        payload = {"input_type": "chat", "output_type": "chat", "input_value": q}
        try:
            res = requests.post(url, json=payload, timeout=120)
            res.raise_for_status()
            _print_chat_output(res.json())
        except Exception as e:
            print(f"Request failed: {e}")

def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--flow-id", required=True, help="Langflow UI → Share → API access")
    ap.add_argument("--base-url", default=os.environ.get("LANGFLOW_URL", "http://localhost:7860"))
    args = ap.parse_args(argv)
    run_client(args.flow_id, args.base_url)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())