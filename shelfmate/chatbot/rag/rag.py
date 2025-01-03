from pinecone import Index, Pinecone
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain.prompts import PromptTemplate


class RagChain:
    def __init__(self, model="gpt-4o-mini", memory=True):
        
        def format_docs(documents):
            return "\n\n".join(doc.page_content for doc in documents)
        
        pc = Pinecone()
        index: Index = pc.Index("documents")
        vector_store = PineconeVectorStore(index=index, embedding=OpenAIEmbeddings(model="text-embedding-ada-002"))
        retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 2, "score_threshold": 0.5})
        
        self.llm = ChatOpenAI(model=model)

        self.template = """ Your task is to clarify the features of the book recommendation system, or about the ShelfMate Company, or about the libraries.
                            Use the following pieces of context to answer the question at the end.
                            If you don't know the answer, just say that you don't know, don't try to make up an answer.
                            Use three sentences maximum and keep the answer as concise as possible.
                            

        {context}

        Question: {question}
        """

        self.custom_rag_prompt = PromptTemplate.from_template(self.template, memory=memory)

        self.rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | self.custom_rag_prompt
            | self.llm
            | StrOutputParser()
        )

    def run_chain(self, question) -> str:
        return self.rag_chain.invoke(question)