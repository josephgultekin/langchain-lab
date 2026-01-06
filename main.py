import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

load_dotenv()


def main():
    print("Hello from langchain-course!")

    information = """
    Jules Gabriel Verne (8 February 1828 - 24 March 1905) was a French novelist, poet, and playwright.

    His collaboration with the publisher Pierre-Jules Hetzel led to the creation of the Voyages extraordinaires,[3] a series of bestselling adventure novels including Journey to the Center of the Earth (1864), Twenty Thousand Leagues Under the Seas (1870), and Around the World in Eighty Days (1872). His novels are generally set in the second half of the 19th century, taking into account contemporary scientific knowledge and the technological advances of the time.

    In addition to his novels, he wrote numerous plays, short stories, autobiographical accounts, poetry, songs, and scientific, artistic and literary studies. His work has been adapted for film and television since the beginning of cinema, as well as for comic books, theater, opera, music and video games.

    Verne is considered to be an important author in France and most of Europe, where he has had a wide influence on the literary avant-garde and on surrealism.[4] His reputation was markedly different in the Anglosphere where he had often been labeled a writer of genre fiction or children's books, largely because of the highly abridged and altered translations in which his novels have often been printed. Since the 1980s, his literary reputation has improved.[5]
    """

    summary_template = """
    Given the information {information} about a person, I want you to create:
    1. A short summary about the person in 2-3 sentences.
    2. Two interesting facts about the person.
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    llm = ChatOpenAI(model="gpt-5", temperature=0)
    # llm = ChatOllama(model="gpt-oss", temperature=0)
    chain = summary_prompt_template | llm
    response = chain.invoke(input = {"information": information})
    print("Response:", response.content)

if __name__ == "__main__":
    main()
