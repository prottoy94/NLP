from langchain_community.document_loaders import  PyPDFLoader, DirectoryLoader

loader= DirectoryLoader(
    path="Document_loaders/books",
    glob="*.pdf",
    show_progress=True,
    loader_cls=PyPDFLoader
)

#loaded_docs = loader.load()
#print(len(loaded_docs))

docs = loader.lazy_load()

for doc in docs:
    print(doc.metadata)