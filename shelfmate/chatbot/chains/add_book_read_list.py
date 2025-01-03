import sys
sys.path.append('C:/Users/mnasc/Desktop/LCD/3rd Year/Capstone Project/Project/shelfmate') 

from chatbot.chains.base import PromptTemplate, generate_prompt_templates 
from pydantic import BaseModel
from langchain.tools import BaseTool
from langchain.schema.runnable.base import Runnable
from langchain_community.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from typing import Type
import openai
import sqlite3
from pinecone import Pinecone

class BookToAdd(BaseModel):
    book_id: str # 
    rating: int
    did_not_finish_flag : int


class ExtractBookToAdd(Runnable):
    def __init__(self, llm, memory=False): 
        super().__init__()

        self.llm = llm

        prompt_template = PromptTemplate(
            system_template=""" 
            You are a part of a book recomendation system team. 
            Your task is to identify the title and the rating of the book from the user input.
            Moreover, if the user did not finish the book, you should set that flag to 1.

            The system will search for the book in the database based on the title you extract from the user input. 
            Ensure you extract the title as accurately as possible.

            Here is the user input:
            {user_input}

            {format_instructions}
            """,
            human_template="user input: {user_input}",
        )

        self.prompt = generate_prompt_templates(prompt_template, memory=memory)
        self.output_parser = PydanticOutputParser(pydantic_object=BookToAdd)
        self.format_instructions = self.output_parser.get_format_instructions()

        self.chain = self.prompt | self.llm | self.output_parser


    def invoke(self, inputs):
        result = self.chain.invoke(
            {
                "user_input": inputs["user_input"],
                "format_instructions": self.format_instructions,
            })
        pinecone = Pinecone()
        index = pinecone.Index('books')

        title = result.book_id
        embedding = openai.embeddings.create(
            model="text-embedding-ada-002",
            input=[title]).data[0].embedding

        filter_condition = {"type": {"$eq": "title"}}
        search_results = index.query(
            vector=embedding,
            top_k=1,
            include_metadata=True,
            filter=filter_condition)

        title = search_results['matches'][0]['metadata']['text']
        book_id = int(search_results['matches'][0]['metadata']['book_id'])
        
        result.book_id = book_id
        return result

######################################################

class AddBookOutput(BaseModel):
    output: str

class AddBookReadListChain(Runnable): 

    def __init__(self, memory: bool = True) -> str:
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.extract_chain = ExtractBookToAdd(self.llm) 
        prompt_bot_return = PromptTemplate( 
            system_template = """
            You are a part of the database manager team for a book recommendation platform called Shelfmate. 
            The user asked to add a book to their read list. 
            There are 3 possible outcomes:
            - The book was successfully added successfully (status='success').
            - The book was already in the user read list (status='no_change').
            - There was an error adding the book to the read list (status='error').
            
            Given the user input and the operation status, your task is to return to the user a message stating the result of the operation in a friendly way \
            and guide the user to what more they can do on the platform. That includes adding books to their read list, adding genres \
            and authors to their favorites list, receiving suggestions of books and authors, and creating a reading plan.
            
            Here is the user input:
            {user_input}

            Chat History:
            {chat_history}
            
            Status of the operation:
            {status}

            {format_instructions}
            """,
            human_template = "user input: {user_input}"
        )

        self.prompt = generate_prompt_templates(prompt_bot_return, memory=memory) 
        self.output_parser = PydanticOutputParser(pydantic_object=AddBookOutput)
        self.format_instructions = self.output_parser.get_format_instructions()
        self.chain = (self.prompt | self.llm | self.output_parser).with_config({"run_name": self.__class__.__name__}) 

    def invoke(self, user_input, config): 
        username = config.get('configurable').get('user_id') 
        book_info = self.extract_chain.invoke({"user_input": user_input}) 
        con = sqlite3.connect("C:/Users/mnasc/Desktop/LCD/3rd Year/Capstone Project/Project/shelfmate.db")
        cursor = con.cursor()

        cursor.execute(f"SELECT book_id FROM read_list WHERE username = ? AND book_id = ?", (username, book_info.book_id))
        query_results = cursor.fetchone()

        if not query_results: 
            try:
                cursor.execute(
                    "INSERT INTO read_list (username, book_id, rating, did_not_finish_flag) VALUES (?, ?, ?, ?)",
                    (username, book_info.book_id, book_info.rating, book_info.did_not_finish_flag),
                )
                con.commit()

                cursor.execute("SELECT title FROM books WHERE book_id = ?", (book_info.book_id,))
                title = cursor.fetchone()[0]
                self.status = 'success'

            except sqlite3.OperationalError as e:
                print(f"Error: {e}")
                self.status = 'no_change'
            finally:
                cursor.close()
                con.close()
        
        else:
            if query_results[0]==book_info.book_id:
                self.status = 'error'
        con.close()
        response = self.chain.invoke({ 
            "user_input": user_input['user_input'],
            'chat_history': user_input['chat_history'], 
            "status": self.status,
            "format_instructions": self.format_instructions
        })

        return response