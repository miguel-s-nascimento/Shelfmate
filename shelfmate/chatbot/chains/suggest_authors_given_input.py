

from base import PromptTemplate, generate_prompt_templates 
from pydantic import BaseModel
from langchain.tools import BaseTool
from langchain.schema.runnable.base import Runnable
from langchain_community.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from typing import Type
import openai
import sqlite3
from pinecone import Pinecone
from sklearn.cluster import KMeans


pinecone = Pinecone()
index = pinecone.Index('books')


def semantic_search(ids, k):
   
    # Generate embeddings
    desc_ids = ['desc_' + str(i) for i in ids]
    fetch_results = index.fetch(desc_ids)
    embeddings = [fetch_results.vectors[i].values for i in desc_ids]

    # Cluster embeddings
    kmeans = KMeans(n_clusters=1, random_state=42)
    kmeans.fit(embeddings)
    cluster_centers = kmeans.cluster_centers_
    
    filter_condition = {"book_id": {"$nin": ids}, 
                        "type": {"$eq": 'description'}}

    # Perform semantic search
    search_results = index.query(
        vector=cluster_centers[0].tolist(),
        top_k=k,
        include_metadata=True,
        filter=filter_condition
    )

    # Display results
    results = []
    for match in search_results["matches"]:
        results.append(
            int(match['metadata']['book_id'])
        )

    return results



class WhichInput_(BaseModel):
    which_input: str
    name: str


class ExtractInput_(Runnable):
    def __init__(self, llm, memory= False):
        super().__init__()
        self.llm = llm
        prompt_template = PromptTemplate(
            system_template=""" 
            You are a part of a book recomendation system team. 
            The user will ask for authors sugestions. Your task is to identify if the user wants a suggestion based on books, authors or genres. If the user provides an author name, return 'author' and its name, if the user provides a genre, return 'genre' and its name, and if the user provides a book title, return 'book' and its title.

            Here is the user input:
            {user_input}

            {format_instructions}
            """,
            human_template="user input: {user_input}",
        )

        self.prompt = generate_prompt_templates(prompt_template, memory=memory)
        self.output_parser = PydanticOutputParser(pydantic_object=WhichInput_)
        self.format_instructions = self.output_parser.get_format_instructions()

        self.chain = self.prompt | self.llm | self.output_parser


    def invoke(self, inputs):
        result = self.chain.invoke(
            {
                "user_input": inputs["user_input"],
                "format_instructions": self.format_instructions,
            })

        return result
    
#####################################################################################################################################

class SuggestAuthorOutput(BaseModel):
    output: str

class SuggestAuthorGivenInputChain(Runnable):
    name: str = "SuggestAuthorTool" 
    description: str = "Suggest an author based on the user input"
    args_schema: Type[BaseModel] = SuggestAuthorOutput
    return_direct: bool = True

    def __init__(self, memory:bool =True):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.extract_chain = ExtractInput_(self.llm)
        
        prompt_bot_return = PromptTemplate(
            system_template=""" 
            You are a part of the database manager team for a book recommendation platform called Shelfmate. 
            The user asked for author suggestions based on a specific author, genre or book. 
            If the suggestions are empty it means that it was not possible to find similar authors based on the user input.

            Given the user input and the suggestions generated, your task is to return to the user a message stating the suggestions in a friendly way \
            and guide the user to what more they can do on the platform. That includes adding books to their read list, adding genres \
            and authors to their favorites list, receiving suggestions of books and authors, and creating a reading plan.  
            
            User Input: 
            {user_input}
            
            Suggestions:
            {suggestions}
            
            Chat History:
            {chat_history}
            
            {format_instructions}
            """,
            human_template="user input: {user_input}"
        )
        
        self.prompt = generate_prompt_templates(prompt_bot_return, memory=memory)
        self.output_parser = PydanticOutputParser(pydantic_object=SuggestAuthorOutput)
        self.format_instructions = self.output_parser.get_format_instructions()
        self.chain = (self.prompt | self.llm | self.output_parser).with_config({"run_name": self.__class__.__name__})
        
        
    def invoke(self, user_input, config):
        username = config.get('configurable').get('user_id')
        con = sqlite3.connect("C:/Users/mnasc/Desktop/LCD/3rd Year/Capstone Project/Project/shelfmate.db")
        cursor = con.cursor()
        
        u_input = self.extract_chain.invoke({"user_input": user_input})

        if u_input.which_input == 'author':
            author_name = u_input.name
            cursor.execute("SELECT author_id FROM authors WHERE author_name LIKE ?", (author_name,))
            author_results = cursor.fetchall()

            if not author_results:
                raise ValueError(f"Author '{author_name}' not in the Database.")

            author_id = author_results[0][0]
            cursor.execute(f"SELECT author_name FROM authors WHERE author_id LIKE ?", (author_id,))

            # Query to get up to 5 books by the given author
            query = """
            SELECT b.book_id
            FROM books b
            INNER JOIN authors_books ab ON b.book_id = ab.book_id
            WHERE ab.author_id = ?
            LIMIT 5;
            """
            cursor.execute(query, (author_id,))
            author_books = cursor.fetchall()

            if not author_books:
                return f"No books found for the author '{author_name}'."

            # Get book IDs for semantic search
            book_ids = [book[0] for book in author_books]

            # Perform semantic search to find similar books
            similar_book_ids = semantic_search(book_ids, 10)

            # Query to fetch distinct authors of similar books, excluding the queried author
            query = """
            SELECT DISTINCT a.author_name
            FROM authors a
            INNER JOIN authors_books ab ON a.author_id = ab.author_id
            WHERE ab.book_id IN ({})
            AND a.author_name != ?
            """.format(','.join('?' for _ in similar_book_ids))

            cursor.execute(query, similar_book_ids + [author_name])
            self.suggestions = cursor.fetchall()
        
        if u_input.which_input == 'genre':
            genre_name = u_input.name
            cursor.execute("SELECT genre_id FROM genres WHERE genre LIKE ?", (genre_name,))
            genre_results = cursor.fetchall()

            if not genre_results:
                raise ValueError(f"Genre '{genre_name}' not in the Database.")

            genre_id = genre_results[0][0]
            cursor.execute("SELECT genre FROM genres WHERE genre_id LIKE ?", (genre_id,))
            
            # Query to get distinct authors for the given genre (limit to 5)
            query = """
                SELECT DISTINCT a.author_name
                FROM authors a
                WHERE a.top_genre = ?
                ORDER BY a.author_name
                LIMIT 5
                """
            cursor.execute(query, (genre_id,))
            self.suggestions = cursor.fetchall()


        if u_input.which_input == 'book':
            book_title = u_input.name

            # Fetch embedding and perform semantic search
            embedding = openai.embeddings.create(model="text-embedding-ada-002", input=[book_title]).data[0].embedding
            filter_condition = {"type": {"$eq": "title"}}
            search_results = index.query(
                vector=embedding,
                top_k=1,
                include_metadata=True,
                filter=filter_condition,
            )

            book_id = int(search_results['matches'][0]['metadata']['book_id'])
            
            # Fetch the author's ID(s) based on the book_id
            cursor.execute("SELECT author_id FROM authors_books WHERE book_id = ?", (book_id,))
            author_ids = cursor.fetchall()
            
            # Extract book IDs for the authors
            book_ids = []
            for author_id_tuple in author_ids:  # Iterate over each tuple (since fetchall() returns a list of tuples)
                author_id = author_id_tuple[0]  # Extract the first element of the tuple (the actual author_id)
                
                cursor.execute("SELECT book_id FROM authors_books WHERE author_id = ?", (author_id,))
                books = cursor.fetchall()
                book_ids.extend([book[0] for book in books])  # Add the book IDs to the list

            # Use the book_ids for semantic search to find similar books
            similar_book_ids = semantic_search(book_ids, 10)

            # Exclude the original authors from the results
            filtered_author_ids = [author_id for author_id in similar_book_ids if author_id not in author_ids]

            # Query to fetch author names
            query = """
            SELECT DISTINCT author_name
            FROM authors
            WHERE author_id IN ({})
            ORDER BY author_name
            LIMIT 5
            """.format(",".join("?" * len(filtered_author_ids)))

            cursor.execute(query, filtered_author_ids)
            self.suggestions = cursor.fetchall()

        con.close()
        
        response = self.chain.invoke({ 
                    "user_input": user_input['user_input'],
                    'chat_history': user_input['chat_history'], 
                    "suggestions": self.suggestions,
                    "format_instructions": self.format_instructions
                })

        return response