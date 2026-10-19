import os
from langchain_groq import ChatGroq
from rag_pipeline import retrieve_policy
from dotenv import load_dotenv
from groq import Groq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
load_dotenv()

client=Groq(api_key=os.environ.get('GROQ_API_KEY'))

parser=StrOutputParser()

draft_llm=ChatGroq(
    model='openai/gpt-oss-20b',
    api_key=os.getenv('GROQ_API_KEY'),
    temperature=0.3
)

critique_llm=ChatGroq(
    model='openai/gpt-oss-20b',
    api_key=os.getenv('GROQ_API_KEY'),
    temperature=0.3
)

draft_prompt=ChatPromptTemplate.from_template("""
You are a fintech support agent corresponding to the customer complaint 
    Ticket  :"{ticket_text}"
    Category :"{category}"
    Policy  :"{policy_text}"
    Write a short professional reply grounded only in the policy above do not invent rules or timelines not mentioned in the policy
    Keep it under 100 words
""")

critique_prompt=ChatPromptTemplate.from_template("""
Policy:"{policy_text}"
Draft Reply:"{draft_reply_text}"
Does the draft reply contradict misintrepet or invent details not present in the policy above ?
Answer in the excat format:
VERDICT: YES or NO
Reason: <one line explaining why>
""")

draft_chain= draft_prompt | draft_llm | parser
critique_chain= critique_prompt | critique_llm | parser


def draft_reply(ticket_text,category,policy_text):
    return draft_chain.invoke({
        'ticket_text':ticket_text[:500],
        'category':category,
        'policy_text':policy_text
    }).strip()



def self_critique(draft_reply_text,policy_text):
    return critique_chain.invoke({
        'draft_reply_text':draft_reply_text,
        'policy_text':policy_text
    }).strip()




def triage_ticket(ticket_text,category,confidence,high_risk_categories):
    ##relkevant policy from rag
    result=retrieve_policy(ticket_text,k=1)
    policy_title=result[0]['title']
    policy_text=result[0]['content']

    ## Self critique step
    reply=draft_reply(ticket_text,category,policy_text)
    ## Self-Critique
    critique=self_critique(reply,policy_text)
    contradicts ="VERDICT: YES" in critique.upper()

    ## decision step -> Logic Part
    if category in high_risk_categories:
        action='ESCALATE'
        reason=f"High risk category ({category}) always require human review per policy"
    elif contradicts:
        action='ESCALATE'
        reason=f"Self-critique flagged a possible issue :({critique})"
    elif confidence < 0.7:
        action="ESCALATE"
        reason=f"Classifier confidence too low ({confidence:.2f}) for auto-resolution"
    else:
        action="AUTO-RESOLVE"
        reason='High confidence,low risk category and reply passed self-critique'
    return {
        "policy_matched":policy_text,
        "draft_reply":reply,
        'self_critique':critique,
        "action":action,
        "reason":reason
    }












