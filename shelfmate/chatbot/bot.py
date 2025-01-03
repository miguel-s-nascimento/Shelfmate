# Import necessary classes and modules for chatbot functionality
from typing import Callable, Dict, Optional

import sys
sys.path.append('C:/Users/mnasc/Desktop/LCD/3rd Year/Capstone Project/Project/shelfmate')

from chatbot.memory import MemoryManager

from chatbot.chains.update_profile_info import UpdateUserInfoChain
from chatbot.chains.insert_fav_author_genre import AddFavAuthorGenreChain
from chatbot.chains.add_book_read_list import AddBookReadListChain
from chatbot.chains.router import RouterChain
from chatbot.router.loader import load_intention_classifier
from chatbot.chains.chitchat import ChitChatResponseChain, ChitChatClassifierChain
from chatbot.chains.suggest_books_given_input import SuggestBookGivenInputChain
from chatbot.chains.suggest_authors_given_input import SuggestAuthorGivenInputChain
from chatbot.chains.suggest_books_given_favourites import SuggestBooksGivenFavChain
from chatbot.chains.suggest_authors_given_favourites import SuggestAuthorsGivenFavChain
from chatbot.chains.suggest_books_given_trope import SuggestBookGivenTropeChain
from chatbot.rag.rag import RagChain

from langchain_core.runnables.history import RunnableWithMessageHistory


class MainChatbot:
    """A bot that handles customer service interactions by processing user inputs and
    routing them through configured reasoning and response chains.
    """

    def __init__(self):
        """Initialize the bot with session and language model configurations."""
        # Initialize the memory manager to manage session history
        self.memory = MemoryManager()

        # Map intent names to their corresponding reasoning and response chains
        self.chain_map = {
            "update_profile_info": self.add_memory_to_runnable(UpdateUserInfoChain()),
            "insert_new_favorite_author_genre": self.add_memory_to_runnable(AddFavAuthorGenreChain()),
            "add_book_to_read_list": self.add_memory_to_runnable(AddBookReadListChain()),
            "router": RouterChain(),
            "chitchat": self.add_memory_to_runnable(ChitChatResponseChain()),
            "chitchat_class": ChitChatClassifierChain(),
            "suggest_books_given_input": self.add_memory_to_runnable(SuggestBookGivenInputChain()),
            "suggest_authors_given_input": self.add_memory_to_runnable(SuggestAuthorGivenInputChain()),
            "suggest_books_given_favorites": self.add_memory_to_runnable(SuggestBooksGivenFavChain()),
            "suggest_books_given_trope": self.add_memory_to_runnable(SuggestBookGivenTropeChain()),
            "suggest_authors_given_favorites": self.add_memory_to_runnable(SuggestAuthorsGivenFavChain()),

            
        }

        """self.rag = self.add_memory_to_runnable(
            RagChain(
                memory=True,
            ).run_chain
        )"""

        # Map of intentions to their corresponding handlers
        self.intent_handlers: Dict[Optional[str], Callable[[Dict[str, str]], str]] = {
            "update_profile_info": self.handle_update_profile_info,
            "insert_new_favorite_author_genre": self.handle_new_favorite_author_genre,
            "add_book_to_read_list": self.handle_add_book_to_read_list,
            "chitchat": self.handle_chitchat_intent,
            "suggest_books_given_input": self.handle_suggest_books_given_input,
            "suggest_authors_given_input": self.handle_suggest_authors_given_input,
            "suggest_books_given_favorites": self.handle_suggest_books_given_favorites,
            "suggest_books_given_trope": self.handle_suggest_books_given_trope,
            "suggest_authors_given_favorites": self.handle_suggest_authors_given_favorites,
        }

        # Load the intention classifier to determine user intents
        self.intention_classifier = load_intention_classifier()

    def user_login(self, username: str, conversation_id: str) -> None:
        """Log in a user by setting the user and conversation identifiers.

        Args:
            username: Identifier for the user.
            conversation_id: Identifier for the conversation.
        """
        self.username = username
        self.conversation_id = conversation_id
        self.memory_config = {
            "configurable": {
                "conversation_id": self.conversation_id,
                "user_id": self.username,
            }
        }

    def add_memory_to_runnable(self, original_runnable):
        """Wrap a runnable with session history functionality.

        Args:
            original_runnable: The runnable instance to which session history will be added.

        Returns:
            An instance of RunnableWithMessageHistory that incorporates session history.
        """
        return RunnableWithMessageHistory(
            original_runnable,
            self.memory.get_session_history,  # Retrieve session history
            input_messages_key="user_input",  # Key for user inputs
            history_messages_key="chat_history",  # Key for chat history
            history_factory_config=self.memory.get_history_factory_config(),  # Config for history factory
        ).with_config(
            {
                "run_name": original_runnable.__class__.__name__
            }  # Add runnable name for tracking
        )

    def get_chain(self, intent: str):
        """Retrieve the reasoning and response chains based on user intent.

        Args:
            intent: The identified intent of the user input.

        Returns:
            A tuple containing the reasoning and response chain instances for the intent.
        """
        return self.chain_map[intent]


    def get_user_intent(self, user_input: Dict):
        """Classify the user intent based on the input text.

        Args:
            user_input: The input text from the user.

        Returns:
            The classified intent of the user input.
        """
        # Retrieve possible routes for the user's input using the classifier
        intent_routes = self.intention_classifier.retrieve_multiple_routes(
            user_input["user_input"]
        )

        # Handle cases where no intent is identified
        if len(intent_routes) == 0:
            return None
        else:
            intention = intent_routes[0].name  # Use the first matched intent

        # Validate the retrieved intention and handle unexpected types
        if intention is None:
            return None
        elif isinstance(intention, str):
            return intention
        else:
            # Log the intention type for unexpected cases
            intention_type = type(intention).__name__
            print(
                f"I'm sorry, I didn't understand that. The intention type is {intention_type}."
            )
            return None

    def handle_update_profile_info(self, user_input: Dict[str, str]) -> str:
        """Handle the update profile info intent by processing user input and providing a response.

        Args:
            user_input: The input text from the user.

        Returns:
            The content of the response after processing through the chains.
        """
        # Retrieve reasoning and response chains for the update profile info intent
        chain = self.get_chain("update_profile_info")
        user_input['chat_history'] = self.memory.get_session_history(
            self.username, self.conversation_id
        )
        # Generate a response using the output of the reasoning chain
        response = chain.invoke(user_input, config=self.memory_config)

        return response.output

    def handle_new_favorite_author_genre(self, user_input: Dict[str, str]) -> str:
        """Handle the insert new fav author/genre intent by processing user input and providing a response.

        Args:
            user_input: The input text from the user.

        Returns:
            The content of the response after processing through the chains.
        """
        # Retrieve reasoning and response chains for the insert new fav author/genre intent
        chain = self.get_chain("insert_new_favorite_author_genre")
        user_input['chat_history'] = self.memory.get_session_history(
            self.username, self.conversation_id
        )
        # Generate a response using the output of the reasoning chain
        response = chain.invoke(user_input, config=self.memory_config)

        return response.output

    def handle_add_book_to_read_list(self, user_input: Dict[str, str]) -> str:
        """Handle the add book to the read list intent by processing user input and providing a response.

        Args:
            user_input: The input text from the user.

        Returns:
            The content of the response after processing through the chains.
        """
        # Retrieve reasoning and response chains for the add book to read list intent
        chain = self.get_chain("add_book_to_read_list")
        user_input['chat_history'] = self.memory.get_session_history(
            self.username, self.conversation_id
        )
        # Generate a response using the output of the reasoning chain
        response = chain.invoke(user_input, config=self.memory_config)

        return response.output

    def handle_suggest_books_given_input(self, user_input: Dict[str, str]) -> str:
        """Handle the suggest books given user input intent by processing user input and providing a response.

        Args:
            user_input: The input text from the user.

        Returns:
            The content of the response after processing through the chains.
        """
        # Retrieve reasoning and response chains for the add book to read list intent
        chain = self.get_chain("suggest_books_given_input")
        user_input['chat_history'] = self.memory.get_session_history(
            self.username, self.conversation_id
        )
        # Generate a response using the output of the reasoning chain
        response = chain.invoke(user_input, config=self.memory_config)

        return response.output

    def handle_suggest_authors_given_input(self, user_input: Dict[str, str]) -> str:
        """Handle the suggest authors given user input intent by processing user input and providing a response.

        Args:
            user_input: The input text from the user.

        Returns:
            The content of the response after processing through the chains.
        """
        # Retrieve reasoning and response chains for the add book to read list intent
        chain = self.get_chain("suggest_authors_given_input")
        user_input['chat_history'] = self.memory.get_session_history(
            self.username, self.conversation_id
        )
        # Generate a response using the output of the reasoning chain
        response = chain.invoke(user_input, config=self.memory_config)

        return response.output

    def handle_suggest_books_given_favorites(self, user_input: Dict[str, str]) -> str:
        """Handle the suggest books given favorites intent by processing user input and providing a response.

        Args:
            user_input: The input text from the user.

        Returns:
            The content of the response after processing through the chains.
        """
        # Retrieve reasoning and response chains for the add book to read list intent
        chain = self.get_chain("suggest_books_given_favorites")
        user_input['chat_history'] = self.memory.get_session_history(
            self.username, self.conversation_id
        )
        # Generate a response using the output of the reasoning chain
        response = chain.invoke(user_input, config=self.memory_config)

        return response.output

    def handle_suggest_authors_given_favorites(self, user_input: Dict[str, str]) -> str:
        """Handle the suggest authors given favorites intent by processing user input and providing a response.

        Args:
            user_input: The input text from the user.

        Returns:
            The content of the response after processing through the chains.
        """
        # Retrieve reasoning and response chains for the add book to read list intent
        chain = self.get_chain("suggest_authors_given_favorites")
        user_input['chat_history'] = self.memory.get_session_history(
            self.username, self.conversation_id
        )
        # Generate a response using the output of the reasoning chain
        response = chain.invoke(user_input, config=self.memory_config)

        return response.output

    def handle_suggest_books_given_trope(self, user_input: Dict[str, str]) -> str:
        """Handle the suggest books given trope intent by processing user input and providing a response.

        Args:
            user_input: The input text from the user.

        Returns:
            The content of the response after processing through the chains.
        """
        # Retrieve reasoning and response chains for the add book to read list intent
        chain = self.get_chain("suggest_books_given_trope")
        user_input['chat_history'] = self.memory.get_session_history(
            self.username, self.conversation_id
        )
        # Generate a response using the output of the reasoning chain
        response = chain.invoke(user_input, config=self.memory_config)

        return response.output

    def handle_chitchat_intent(self, user_input: Dict[str, str]) -> str:
        """Handle chitchat intents

        Args:
            user_input: The input text from the user.

        Returns:
            The content of the response after processing through the new chain.
        """
        # Retrieve reasoning and response chains for the chitchat intent
        chain = self.get_chain("chitchat")

        # Generate a response using the output of the reasoning chain
        response = chain.invoke(user_input, config=self.memory_config)

        return response

    def handle_unknown_intent(self, user_input: Dict[str, str]) -> str:
        """Handle unknown intents by providing a chitchat response.

        Args:
            user_input: The input text from the user.

        Returns:
            The content of the response after processing through the new chain.
        """

        chitchat_reasoning_chain = self.get_chain("chitchat_class")
        
        input_message = {}

        input_message["user_input"] = user_input["user_input"]
        input_message["chat_history"] = self.memory.get_session_history(
            self.username, self.conversation_id
        )
        
        reasoning_output = chitchat_reasoning_chain.invoke(input_message)
        
        if reasoning_output.chitchat:
            return self.handle_chitchat_intent(user_input)
        else:
            router_reasoning_chain2 = self.get_chain("router")
            reasoning_output2 = router_reasoning_chain2.invoke(input_message)
            new_intention = reasoning_output2.intent
            new_handler = self.intent_handlers.get(new_intention)
            return new_handler(user_input)

    def save_memory(self) -> None:
        """Save the current memory state of the bot."""
        self.memory.save_session_history(self.username, self.conversation_id)

    def process_user_input(self, user_input: Dict[str, str]) -> str:
        """Process user input by routing through the appropriate intention pipeline.

        Args:
            user_input: The input text from the user.

        Returns:
            The content of the response after processing through the chains.
        """
        # Classify the user's intent based on their input
        intention = self.get_user_intent(user_input)
        
        # Route the input based on the identified intention
        handler = self.intent_handlers.get(intention, self.handle_unknown_intent)
        return handler(user_input)
