import pandas as pd
import streamlit as st

from src.predict import analyze

st.set_page_config(page_title="Smishing Detector", page_icon="🛡️")
st.title("🛡️ SMS Spam / Smishing Detector")
st.caption("ML model + URL look-alike analysis, with explainable red flags. Built for Indian scam patterns.")

EXAMPLES = {
    "Fake SBI KYC": "Dear customer your SBI account will be blocked today. Update KYC: http://sbi-kyc-update.xyz/login",
    "Fake FASTag": "FASTag KYC pending. Your FASTag will be deactivated. Update now http://fastag-kyc.online",
    "Real bank debit alert": "Rs 1250.00 debited from A/c XX4821 on 29-10-26 via UPI to Swiggy. Ref 629183746",
}
cols = st.columns(len(EXAMPLES))
for col, (name, msg) in zip(cols, EXAMPLES.items()):
    if col.button(name):
        st.session_state["text"] = msg

text = st.text_area("Paste an SMS:", key="text", height=140)

if st.button("Analyze", type="primary") and text.strip():
    try:
        r = analyze(text)
    except FileNotFoundError as e:
        st.error(str(e)); st.stop()
    msg = f"ML probability {r['ml_probability']:.0%} · URL risk {r['url_risk']}/100"
    if r["label"] == "SCAM":
        st.error(f"⚠️ SCAM — {msg}")
    elif r["label"] == "SUSPICIOUS":
        st.warning(f"🤔 SUSPICIOUS — {msg}")
    else:
        st.success(f"✅ Looks legitimate — {msg}")
    if r["red_flags"]:
        st.subheader("Red flags")
        for f in r["red_flags"]:
            st.write(f"- {f}")
    if r["urls"]:
        st.subheader("Links found")
        st.dataframe(pd.DataFrame([{"host": u["host"], "risk": u["risk"], "imitates": u["brand"]}
                                   for u in r["urls"]]), hide_index=True)
    st.caption("Never share OTPs, PINs or card details. Report scams at cybercrime.gov.in or call 1930.")
