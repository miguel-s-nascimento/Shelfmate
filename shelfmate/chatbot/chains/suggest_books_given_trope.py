import sys
sys.path.append('C:/Users/mnasc/Desktop/LCD/3rd Year/Capstone Project/Project/shelfmate')

from chatbot.chains.base import PromptTemplate, generate_prompt_templates 

from langchain.schema.runnable.base import Runnable
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from langchain_community.chat_models import ChatOpenAI
from langchain.tools import BaseTool
from typing import Type, Any

from pinecone import Pinecone
import openai

class ExtractTropeInput(BaseModel):
    trope: str 


class ExtractTrope(Runnable):
    def __init__(self, llm, memory=False):
        super().__init__()

        self.llm = llm

        prompt_template = PromptTemplate(
            system_template=""" 
            You are part of a book recommendation system team. 
            The user wants a book recommendation. 
            Your task is to extract the trope or theme of a book that the user describes in their input.

            Ensure you extract the trope as accurately as possible.

            Here is the user input:
            {user_input}

            {format_instructions}
            """,
            human_template="user input: {user_input}",
        )

        self.prompt = generate_prompt_templates(prompt_template, memory=memory)
        self.output_parser = PydanticOutputParser(pydantic_object=ExtractTropeInput)
        self.format_instructions = self.output_parser.get_format_instructions()

        self.chain = self.prompt | self.llm | self.output_parser
    
    def invoke(self, inputs):
        result = self.chain.invoke(
            {
                "user_input": inputs["user_input"],
                "format_instructions": self.format_instructions,
            }
        )
        
        return result
        
################################################################

class SuggestBookOutput(BaseModel): 
        output: str 

class SuggestBookGivenTropeChain(Runnable): 
    name: str = "SuggestBookGivenTropeChain" 
    description: str = "Suggest a book based on the user input"
    args_schema: Type[BaseModel] = SuggestBookOutput
    return_direct: bool = True

    def __init__(self, memory: bool = True):
        super().__init__()
        self.llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)
        self.extract_chain = ExtractTrope(self.llm)

        prompt_bot_return = PromptTemplate(
            system_template="""
            You are a part of the database manager team for a book recommendation platform called Shelfmate. 
            The user asked for book suggestions based on a specific trope. 

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
        self.output_parser = PydanticOutputParser(pydantic_object=SuggestBookOutput)
        self.format_instructions = self.output_parser.get_format_instructions()
        self.chain = (self.prompt | self.llm | self.output_parser).with_config({"run_name": self.__class__.__name__}) 
    
    def invoke(self, user_input, config):       
        result = self.extract_chain.invoke({"user_input": user_input})
        trope = result.trope

        pinecone = Pinecone()
        index = pinecone.Index('books')

        trope_embedding = openai.embeddings.create(
            model="text-embedding-ada-002",
            input=[trope]
        ).data[0].embedding

        search_results = index.query(
            vector=trope_embedding,
            top_k=5,  
            include_metadata=True  
        )

        if "matches" in search_results and search_results["matches"]:
            suggested_books = [match['metadata']['text'] for match in search_results['matches']]
            self.suggestions = suggested_books
        
        response = self.chain.invoke(
            {
                "user_input": user_input['user_input'],
                "suggestions": self.suggestions,
                "chat_history": user_input['chat_history'],
                "format_instructions": self.format_instructions,
            }
        )

        return response   