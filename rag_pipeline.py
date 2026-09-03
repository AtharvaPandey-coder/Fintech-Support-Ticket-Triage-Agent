from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

vector_store=FAISS.load_local(
    "policy_index",
    embeddings,allow_dangerous_deserialization=True
)
def retrieve_policy(query,k=1):
    """
    Given a customer ticket's text retrieve the most relevant policy documents(s).
    Retrieve a list of dicts with id,title,content,and score.
    """
    results=vector_store.similarity_search_with_score(query,k=k)
    output=[]
    for doc,score in results:
        output.append({
            'id':doc.metadata.get("id"),
            'title':doc.metadata.get("title"),
            'content':doc.page_content,
            'score':score
        })
    return output

#if __name__ == "__main__":
#    test_query="my mortage payment was applied to the wrong account"
#    results=retrieve_policy(test_query,k=2)
#    print(f"Query{test_query}")
#    for r in results:
#        print(f"Score:{r['score']:.3f}|{r['title']}")











