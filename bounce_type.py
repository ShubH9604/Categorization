import pandas as pd

# Step 1: Load Excel
input_file = "bounce.xlsx"  # <-- Update this
df = pd.read_excel(input_file)
df['Narration'] = df['Narration'].astype(str).str.lower()

# Step 2: Final bounce keyword dictionary with type keywords and condition keywords
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
            "reversal", "rev:imps", "/rt", "impscommissionreversal", "rev", "return", "ret", "rtn",
            "reversed", "dmr rev", "failed", "cp return", "returned(ins. balance)", "imps return",
            "chq dep ret"
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
            "i/w chq rtn", "i/w chq return", "o/w rtn chq", "reject/(chq no.)", "rtn(chq no.)",
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
            "rtn(blocked account)", "rtn(invalid account)", "rem"
        ]
    }
}

# Step 3: Function to identify bounce type
def identify_bounce_type(narration):
    for bounce_type, data in bounce_keywords.items():
        if any(tk in narration for tk in data["type_keywords"]) and \
           any(kw in narration for kw in data["keywords"]):
            return bounce_type
    return None

# Step 4: Apply logic to DataFrame
df['Bounce Type'] = df['Narration'].apply(identify_bounce_type)

# Step 5: Save updated Excel file
output_file = "bank_statement_with_bounce_type.xlsx"
df.to_excel(output_file, index=False)

print("✅ Bounce type tagging completed. File saved as:", output_file)