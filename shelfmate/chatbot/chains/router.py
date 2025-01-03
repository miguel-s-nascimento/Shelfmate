from typing import Literal

from langchain import callbacks
from langchain.output_parsers import PydanticOutputParser
from langchain.schema.runnable.base import Runnable
from langchain_community.chat_models import ChatOpenAI
from pydantic import BaseModel, Field


from base import PromptTemplate, generate_prompt_templates


class IntentClassification(BaseModel):
    intent: Literal["update_profile_info", "insert_new_favorite_author_genre", "add_book_to_read_list", "suggest_authors_given_input", "suggest_authors_given_favorites", 
                   "suggest_books_given_input", "suggest_books_given_favorites", "suggest_books_given_trope", "browse_available_genres_books_authors", 
                   "create_reading_plan", "recommend_bookstores_per_district", "ask_about_chatbot_features", "ask_about_company_info"] = Field(
        ...,
        description="The classified intent of the user query",
    )


class RouterChain(Runnable):
    def __init__(self, memory=False):
        super().__init__()

        self.llm = ChatOpenAI(model="gpt-4o-mini")
        prompt_template = PromptTemplate(
            system_template="""
            You are an expert classifier of user intentions for the Shelfmate book recommendation platform,
            Your role is to accurately identify the user's intent based on their input and the context provided by the 
            conversation history. Analyze the user's input in the conversation history context and classify 
            it into one of the intents: "update_profile_info", "insert_new_favorite_author_genre", "add_book_to_read_list", 
            "suggest_authors_given_input", "suggest_authors_given_favorites", "suggest_books_given_input", 
            "receive_suggestion_of_books_given_favorites", "suggest_books_given_trope", "browse_available_genres_books_authors", 
            "create_reading_plan", "recommend_bookstores_per_district", "ask_about_chatbot_features", "ask_about_company_info".
            You'll use the following detailed descriptions to classify the user's intent:

            1. **update_profile_info:**  
            The user wants to update a specific information about themselves in the platform, to do so they need to provide the information they want to update and the new value to replace the old info. \
            They can ask for updating the following informations: username, password, email and the district they live in. Note that the district must be in Portugal following the portuguese writting.
            
            2. **insert_new_favorite_author_genre:**  
            The user intends to set an author or books genre as part of their favorite list. Users can have more than one book or genre in their favorite lists. \
            The user should express clearly if they want to add an author or a genre and the name of it.

            3. **add_book_to_read_list:**  
            The user intends to add a book they read and finished or started reading and stopped to their reading list. \
            The user must provide the title of the book, optionally the author, mandatorily the rating from 0 to 5 they want to give to the book \
            and if they did not finish the book they must say so.

            4. **suggest_authors_given_favorites:**  
            The user will ask for an author recommendation based on either their favorite books, genres or authors. \
            The user must clearly state that they want an author recommendation and state what will be the comparison object. \
            The user should not provide any specific name of author, genre or book.

            5. **suggest_authors_given_input:**  
            The user will ask for an author recommendation based on a specific book genre or genres, another author or authors, or on specific books. \
            The user must clearly state that they want an author recommendation and state what will be the comparison object.

            6. **suggest_books_given_favorites:**
            The user will ask for a book recommendation based on either their favorite books, genres or authors. \
            The user must clearly state that they want a book recommendation and state what will be the comparison object. \
            The user should not provide any specific name of author, genre or book.

            7. **suggest_books_given_input:**
            The user will ask for a book recommendation based on a specific genre or genres, an author or authors, or on other specific books. \
            The user must clearly state that they want an author recommendation and state what will be the comparison object.

            8. **suggest_books_given_trope:**
            The user will ask for a book recommendation based on a specific book trope or general book description. \
            The user must clearly state that they want an author recommendation and state what will be the comparison object. 

            9. **browse_available_genres_books_authors:**
            The user will ask to know which books, genres or authors are available in the platform. \
            The user must clearly state what they want to browse and can add filters such as 'Fiction books'.

            10. **create_reading_plan:**
            The user will ask to create a reading plan monthly or annual. \
            They can ask that plan to be based on their favorite authors, genres or books or based on specific authors, genres or books. \
            They can say if they are fast or slow readers and can state more than one genre, author or book. \
            They must clearly state if they want a monthly or annual plan and if they want it based on their favorites or specific authors, genres or books. 

            11. **recommend_bookstores_per_district:**
            The user will ask you to recommend bookstores in the district they live in or in other specific district in Portugal. \
            
            12. **ask_about_chatbot_features:**
            The user wants to know more about Shelfmate and its chatbot features.
            It can include questions about the chatbot's capabilities, the platform's features, and the chatbot's limitations.

            13. **ask_about_company_info:**
            The user wants to know more about Shelfmate company.
            It can include questions about Shelfmate's mission, history, values, vision, ...
            
            **Input:**

            - user Input: {user_input}  
            - Conversation History: {chat_history}

            **Output Format:**

            - Follow the specified output format and use the detailed descriptions:
            {format_instructions}
            """,
            human_template="User Query: {user_input}",
        )

        self.prompt = generate_prompt_templates(prompt_template, memory=memory)

        self.output_parser = PydanticOutputParser(pydantic_object=IntentClassification)
        self.format_instructions = self.output_parser.get_format_instructions()
        self.chain = (self.prompt | self.llm | self.output_parser).with_config(
            {"run_name": self.__class__.__name__}
        )  # Add a run name to the chain on LangSmith

    def invoke(self, inputs, config=None, **kwargs):
        """Invoke the product information response chain."""
        with callbacks.collect_runs() as cb:
            return self.chain.invoke(
                {
                    "user_input": inputs["user_input"],
                    "chat_history": inputs["chat_history"],
                    "format_instructions": self.format_instructions,
                },
            )
