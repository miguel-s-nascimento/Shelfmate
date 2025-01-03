import sys
sys.path.append('C:/Users/mnasc/Desktop/LCD/3rd Year/Capstone Project/Project/shelfmate')

from chatbot.chains.base import PromptTemplate, generate_prompt_templates
from langchain.schema.runnable.base import Runnable
from langchain.chat_models import ChatOpenAI
from langchain.agents import Tool
from langchain.agents import AgentExecutor
from langchain.tools import BaseTool
from langchain.output_parsers import PydanticOutputParser
from langchain_community.utilities.sql_database import SQLDatabase
from fuzzywuzzy import process
from pydantic import BaseModel
from typing import Type
import sqlite3  

class QueryType(BaseModel):
    query_type: str
    value: str
    num_results: int = 5  # Default to 5 results if not specified


# Define ExtractQueryType class
class ExtractQueryType(Runnable):
    def __init__(self, llm, memory=False):
        super().__init__()
        self.llm = llm

        # Create a prompt template instance
        prompt_template = PromptTemplate(
            system_template=
            """ 
            You are part of a database management team for a book recommendation platform called Shelfmate.
            Given the user input, your task is to identify the type of query the user wants to perform and its value.
            The possible query types are: 'list_genres', 'authors_by_genre', 'books_by_genre', 'books_by_author'.
            If the user input contains 'all' or 'every' authors or genres, return num_results as 20. If ambiguous, return 5.

            Here is the user input:
            {user_input}

            {format_instructions}
            """, human_template="user input: {user_input}"
        )

        
        self.prompt = generate_prompt_templates(prompt_template, memory=memory)
        self.output_parser = PydanticOutputParser(pydantic_object=QueryType)
        self.format_instructions = self.output_parser.get_format_instructions()

        self.chain = self.prompt | self.llm | self.output_parser

        con = sqlite3.connect("C:/Users/mnasc/Desktop/LCD/3rd Year/Capstone Project/Project/shelfmate.db")
        cursor = con.cursor()
        cursor.execute("SELECT genre FROM genres")
        self.genres = [row[0] for row in cursor.fetchall()]
        cursor.execute("SELECT author_name FROM authors")
        self.authors = [row[0] for row in cursor.fetchall()]
        con.close()

    def invoke(self, inputs):
        # Pass user input and format instructions to the chain
        result = self.chain.invoke({
            "user_input": inputs["user_input"],
            "format_instructions": self.format_instructions
        })

        # Perform fuzzy matching for genres or authors
        if result.query_type in ['authors_by_genre', 'books_by_genre']:
            closest_match, score = process.extractOne(result.value, self.genres)
            if score > 80:  # Adjust the threshold as needed
                result.value = closest_match
        elif result.query_type == 'books_by_author':
            closest_match, score = process.extractOne(result.value, self.authors)
            if score > 80:
                result.value = closest_match

        return result
##########################################################################################################
    
# Define QueryDatabaseTool class
class Browser(BaseTool):
    name: str = "QueryDatabaseTool"
    description: str = (
        "Queries the database for genres, authors, or books based on user input, "
        "given the number of results requested. Return a phrase and not a list."
    )
    args_schema: Type[BaseModel] = QueryType
    return_direct: bool = True

    def _run(self, user_input: str) -> str:
        # Initialize the LLM and extract query information
        llm = ChatOpenAI(model="gpt-4o-mini")
        query_info = ExtractQueryType(llm).invoke({"user_input": user_input})

        # Define the database path and connect
        db_path = r"C:\Users\david\Downloads\CapstoneProjectGroup11\shelfmate.db"
        db = SQLDatabase.from_uri(f"sqlite:///{db_path}")

        try:
            num_results = query_info.num_results

            # Handle 'list_genres' query type
            if query_info.query_type == 'list_genres':
                query = f"SELECT genre FROM genres LIMIT {num_results}"
                results = db.run(query, fetch="all")
                if not results:
                    return "It seems like we couldn't find any genres. Please double-check the genre name or try a different query."
                results_str = str(results).strip("[()]\"").replace(",)", "").replace("(", "").strip(",\"")
                return f"Here are some genres that might interest you: {results_str}. Let me know if you'd like more options."

            # Handle 'authors_by_genre' query type
            elif query_info.query_type == 'authors_by_genre':
                genre = query_info.value
                query = (
                    f"SELECT a.author_name FROM authors a "
                    f"JOIN genres g ON a.top_genre = g.genre_id "
                    f"WHERE g.genre = '{genre}' LIMIT {num_results}"
                )
                results = db.run(query, fetch="all")
                if not results:
                    return f"I'm sorry, I couldn't find any authors for the genre '{genre}'. Please check the spelling or try another genre."
                results_str = str(results).strip("[()]\"").replace(",)", "").replace("(", "").strip(",\"")
                return f"Here are some authors who primarily write in the '{genre}' genre: {results_str}. Would you like more recommendations?"

            # Handle 'books_by_genre' query type
            elif query_info.query_type == 'books_by_genre':
                genre = query_info.value
                query = (
                    f"SELECT b.title FROM books b "
                    f"JOIN books_genres bg ON b.book_id = bg.book_id "
                    f"JOIN genres g ON bg.genre_id = g.genre_id "
                    f"WHERE g.genre = '{genre}' LIMIT {num_results}"
                )
                results = db.run(query, fetch="all")
                if not results:
                    return f"It seems we don't have any books for the genre '{genre}' in our collection. You can try another genre or let me know if you need suggestions."
                results_str = str(results).strip("[()]\"").replace(",)", "").replace("(", "").strip(",\"")
                return f"Here are a few books from the '{genre}' genre: {results_str}. Let me know if you'd like more book recommendations."

            # Handle 'books_by_author' query type
            elif query_info.query_type == 'books_by_author':
                author = query_info.value
                query = (
                    f"SELECT b.title FROM books b "
                    f"JOIN authors_books ab ON b.book_id = ab.book_id "
                    f"JOIN authors a ON ab.author_id = a.author_id "
                    f"WHERE a.author_name = '{author}' LIMIT {num_results}"
                )
                results = db.run(query, fetch="all")
                if not results:
                    return f"I couldn't find any books by the author '{author}'. Perhaps you'd like to try a different author or genre?"
                results_str = str(results).strip("[()]\"").replace(",)", "").replace("(", "").strip(",\"")
                return f"Here are some books by '{author}': {results_str}. Let me know if you'd like more options or have a specific book in mind."

            # Handle invalid query types
            else:
                return "I couldn't quite understand that. Could you try asking with a different query type? For example, you could ask about genres, authors, or books."

        except Exception as e:
            return f"Oops, something went wrong while querying the database: {e}. Can you try again or let me know if you'd like assistance?"
