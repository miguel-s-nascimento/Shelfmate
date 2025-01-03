import sys
sys.path.append('C:/Users/mnasc/Desktop/LCD/3rd Year/Capstone Project/Project/shelfmate')

from chatbot.chains.base import PromptTemplate, generate_prompt_templates
from pydantic import BaseModel
from langchain import callbacks
from langchain.tools import BaseTool
from langchain.schema.runnable.base import Runnable
from langchain_community.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain_core.output_parsers import StrOutputParser
from typing import Type
import sqlite3

class InfoToUpdate(BaseModel):
    info_to_change: str
    new_value: str

class ExtractInfoToUpdate(Runnable):
    def __init__(self, llm, memory=False):
        super().__init__()

        self.llm = llm
        
        prompt_template = PromptTemplate(
            system_template = """
            You are a part of the database manager team for a book recommendation platform. 
            Given the user input, your task is to identify the profile information the user wants to change and its new value.
            They are allowed to change their username, password, email and the district they live in.
            Note that the district must be in Portugal following the portuguese writting, if they are not please convert.

            Here is the user input:
            {user_input}

            {format_instructions}
            """,
            human_template = "user input: {user_input}"
        )

        self.prompt = generate_prompt_templates(prompt_template, memory=memory)
        self.output_parser = PydanticOutputParser(pydantic_object=InfoToUpdate)
        self.format_instructions = self.output_parser.get_format_instructions()

        self.chain = (self.prompt | self.llm | self.output_parser).with_config(
            {"run_name": self.__class__.__name__})

    def invoke(self, inputs):
        with callbacks.collect_runs() as cb:
            result = self.chain.invoke(
                {
                    "user_input": inputs["user_input"],
                    "format_instructions": self.format_instructions,
                })
        return result

######################################################
class UpdateInfoOutput(BaseModel):
    output: str

class UpdateUserInfoChain(Runnable):   

    def __init__(self, memory: bool = True) -> str:

        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        self.extract_chain = ExtractInfoToUpdate(self.llm)

        prompt_bot_return = PromptTemplate(
            system_template = """
            You are a part of the database manager team for a book recommendation platform called Shelfmate. 
            The user asked for a change in their profile information and they are allowed to change their username, password, email and the district they live in. 
            There are 3 possible outcomes:
            - The information was successfully updated (status='success').
            - The information was already set to the value the user wanted to insert (status='no_change').
            - There was an error updating the information (status='error_(...)' with specification of the error).
            
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
        self.output_parser = PydanticOutputParser(pydantic_object=UpdateInfoOutput)
        self.format_instructions = self.output_parser.get_format_instructions()
        self.chain = (self.prompt | self.llm | self.output_parser).with_config({"run_name": self.__class__.__name__})

    def invoke(self, user_input, config):
        username = config.get('configurable').get('user_id')
        user_info = self.extract_chain.invoke({"user_input": user_input})
        con = sqlite3.connect("C:/Users/mnasc/Desktop/LCD/3rd Year/Capstone Project/Project/shelfmate.db")
        cursor = con.cursor()

        # Checking if the info is new
        cursor.execute(f"SELECT {user_info.info_to_change} FROM users WHERE username = ?", (username,))
        query_results = cursor.fetchone()

        if len(user_info.new_value) == 0 or user_info.new_value is None:
            self.status = 'error_invalid_value_to_change'

        if user_info.info_to_change == 'email':
            if cursor.execute(f"SELECT email FROM users WHERE email = ?", (user_info.new_value,)).fetchone():
                self.status = 'error_email_already_taken'
        
        if user_info.info_to_change == 'username':
            if cursor.execute(f"SELECT username FROM users WHERE username = ?", (user_info.new_value,)).fetchone():
                self.status = 'error_username_already_taken'

        if not self.status:
            if query_results[0]!=user_info.new_value:
                try:
                    # Updating if the info to update is different from the info stored
                    query = f"UPDATE users SET {user_info.info_to_change} = ? WHERE username = ?"
                    params = (user_info.new_value, username)
                    cursor.execute(query, params)
                    con.commit()
                    self.status = 'success'

                except sqlite3.OperationalError as e:
                    print(f"Error: {e}")
                    self.status = 'error'
                finally:
                    cursor.close()
                    con.close()
            else: 
                self.status = 'no_change'
        con.close()
        response = self.chain.invoke({
            "user_input": user_input['user_input'],
            'chat_history': user_input['chat_history'],
            "status": self.status,
            "format_instructions": self.format_instructions
        })

        return response