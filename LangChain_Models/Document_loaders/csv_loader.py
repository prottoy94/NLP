from langchain_community.document_loaders import  CSVLoader
loader = CSVLoader(file_path="Document_loaders/Social_Network_Ads.csv", encoding="utf-8")
docs = loader.load()
print(docs)
print(type(docs))
print(len(docs))
print(docs[0])
