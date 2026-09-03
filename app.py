import streamlit as st
import pickle
from agent import triage_ticket

st.set_page_config(page_title='Fintech Support Ticket Triage Agent')
st.title('Fintech Support Ticket Triage Agent')
st.caption('Paste a customer complaint below to see it classified,matched to policy, and triaged.')

tfidf=pickle.load(open('tfidf_vectorizer.pkl','rb'))
lr_model=pickle.load(open('lr_model.pkl','rb'))


HIGH_RISK_CATEGORIES=['Debt collection']

ticket_text=st.text_area(
    "Customer Complaint",
    height=120,
    placeholder="e.g. My mortage payment was applied to the wrong account and now they're charging me a late fee."
)

if st.button("Ticket Triage") and ticket_text.strip():
    with st.spinner("Analyzing..."):

    # ML Classifier
        X_vec=tfidf.transform([ticket_text])
        category=lr_model.predict(X_vec)[0]
        confidence=lr_model.predict_proba(X_vec).max()

    #Step2-Ticket-Triage
        result=triage_ticket(ticket_text,category,confidence,HIGH_RISK_CATEGORIES)

    st.divider()
    col1,col2=st.columns(2)

    with col1:
        st.metric("Predicted Category:",category)
    with col2:
        st.metric("Confidence:",f"{confidence:.0%}")

    st.write(f"Matched policy:**{result['policy_matched']}")

    if result['action'] == "AUTO-RESOLVE":
        st.success(f"Action: {result['action']}")
    else:
        st.warning(f"Action :{result['action']}")

    st.write(f"**Reason:**{result['reason']}")

    with st.expander("Draft Reply"):
        st.info(result['draft_reply'])
    with st.expander("Self-Critique"):
        st.code(result['self_critique'])













