import pandas as pd
import requests
from tqdm import tqdm  # ✅ For progress bar

# --- Step 1: Config ---
TOGETHER_API_KEY="ad40bde1fb41d1aba3781d90732f864dcacbcda767986f984945a61b8213e959"
TOGETHER_MODEL = "Qwen/Qwen2.5-72B-Instruct-Turbo"
API_URL = "https://api.together.xyz/v1/chat/completions"  # ✅ Chat endpoint

HEADERS = {
    "Authorization": f"Bearer {TOGETHER_API_KEY}",
    "Content-Type": "application/json"
}

# --- Step 2: Bounce Keywords ---
bounce_keywords = {
    "ACH": {
        "type_keywords": ["ach", "nach"],
        "keywords": [
            "ach_ad_rtn_chrg", "ach-rt-chg", "ach debit return", "nach return", "ach ecs rtn",
            "nach rtn", "rev:nach", "nach rtn chrg", "ach debit rtn chgs", "nach_ad_rtn_chrg",
            "achdebit rtn", "nach_rtn_chg", "revach-rt-chg", "_lien_rev", "rtn chrg"
        ]
    },
    "IMPS": {
        "type_keywords": ["imps"],
        "keywords": [
            "reversal", "rev:imps", "/rt", "impscommissionreversal", "dmr", "return", "ret", "rtn",
            "reversed", "rev", "failed", "cp return", "returned(ins. balance)", "imps return",
            "chq dep ret", "failed",  "i/w chq return-inward"
        ]
    },
    "RTGS": {
        "type_keywords": ["rtgs"],
        "keywords": [
            "rtn:rtgs", "rtgs-return", "rtgsreturn", "rtgs rev(reverse)", "rtgs failed", "return",
            "acincorrect", "incorrect account number", "rem"
        ]
    },
    "CHEQUE": {
        "type_keywords": ["chq", "cheque"],
        "keywords": [
            "reject(ins. funds)", "returned(ins. funds)", "chq ret", "chq return", "i/w chq ret",
            "i/w chq rtn", "o/w rtn chq", "reject/(chq no.)", "rtn(chq no.)",
            "chq issued bounce", "reject:(chq no.)", "ow chq rej", "brn-ow rtn"
        ]
    },
    "ECS": {
        "type_keywords": ["ecs"],
        "keywords": [
            "ret", "return", "return charge", "return ach", "return ach uip(mode of debit)",
            "nach", "nach fail/ins. balance", "rem", "rtn", "rev", "inward return"
        ]
    },
    "NEFT": {
        "type_keywords": ["neft"],
        "keywords": [
            "account does not exist", "rej", "rtn", "ret", "return", "return(account closed)",
            "return(not a valid cpin)", "returned", "returnfor", "rev", "reversal", "rt",
            "rtn(blocked account)", "rtn(invalid account)", "rem", "neft-rev"
        ]
    }
}

all_types = list(bounce_keywords.keys()) + ["UPI", "None"]

# --- Step 3: Format Keywords for Prompt ---
def format_keywords_for_prompt():
    lines = []
    for bounce_type, data in bounce_keywords.items():
        type_keys = ", ".join(data["type_keywords"])
        keys = ", ".join(data["keywords"][:8]) + "..."  # show first few only
        lines.append(f"- {bounce_type}: Type Keywords [{type_keys}], Match Keywords [{keys}]")
    return "\n".join(lines)

keywords_description = format_keywords_for_prompt()

# --- Step 4: Prompt Builder ---
def create_prompt(narration):
    return f"""You are a banking classifier. Based on narration, identify the bounce type.

Rules:
- Bounce type must match BOTH a type keyword AND at least one keyword from the keyword list.
- If no match is found, return 'None'.
- Take a look at the keywords and type keywords carefully and then detect the keywords.
- If chq and imps/ecs/neft both are there in narration then consider imps/ecs/neft which one of them is present.
- If ach and ecs both are there in narration then consider ach.
- Valid types: {", ".join(all_types)}

Bounce Type Keywords:
{keywords_description}

Narration: "{narration}"
Bounce Type:"""

# --- Step 5: Call Together Chat API ---
def query_llm(narration):
    prompt = create_prompt(narration)
    try:
        response = requests.post(
            API_URL,
            headers=HEADERS,
            json={
                "model": TOGETHER_MODEL,
                "messages": [
                    {"role": "system", "content": "You are a bounce type classification assistant. Always respond with only one of the valid types: ACH, IMPS, RTGS, CHEQUE, ECS, NEFT, UPI, None."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 20,
                "temperature": 0.0,
                "stop": ["\n"]
            }
        )

        if response.status_code == 200:
            # Log raw output for debugging
            raw_output = response.json()["choices"][0]["message"]["content"].strip()
            clean_output = raw_output.upper()

            if clean_output in all_types:
                return clean_output
            else:
                print(f"⚠️ Unrecognized output: '{raw_output}' → marking as Invalid")
                return "Invalid"
        else:
            print(f"❌ API Error {response.status_code}: {response.text}")
            return "Error"
    except Exception as e:
        print("❌ Exception:", e)
        return "Error"

# --- Step 6: Load CSV and Apply with Progress ---
df = pd.read_csv("cheque_bounce.csv")
df['Narration'] = df['Narration'].astype(str).str.strip().str.lower()

print("🚀 Running bounce type classification using Qwen 72B via Together API...")
tqdm.pandas(desc="🔍 Processing Narrations")
df['LLM Bounce Type'] = df['Narration'].progress_apply(query_llm)

# --- Step 7: Save Output as CSV ---
output_path = "cheque_bounce_output.csv"
df.to_csv(output_path, index=False)
print(f"\n✅ LLM-based bounce classification complete. File saved at: {output_path}")