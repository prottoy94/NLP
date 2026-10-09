from langchain_text_splitters import TextSplitter, CharacterTextSplitter, RecursiveCharacterTextSplitter, Language
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('Text_splitters/dl-curriculum.pdf')
docs=loader.load()

text = """
from langchain_text_splitters import RecursiveCharacterTextSplitter

text = LangChain is a framework for building applications with
language models. Text splitters divide large documents into smaller
chunks so they can be processed efficiently.

splitter = RecursiveCharacterTextSplitter(
    chunk_size=50,
    chunk_overlap=10
)

chunks = splitter.split_text(text)

print(chunks)
print("Number of chunks:", len(chunks))
"""

splitter = RecursiveCharacterTextSplitter.from_language(language=Language.PYTHON,chunk_size=400, chunk_overlap=0)
#splitter = TextSplitter(chunk_size=200, chunk_overlap=50)

result = splitter.split_text(text)

print(result[0])
print(len(result))

