from retriever import vectorstore

result = vectorstore._collection.get(
    where_document={"$contains": "mask some percentage of the input tokens"},
    include=["documents", "metadatas"]
)

print("Matches:", len(result["documents"]))

for document, metadata in zip(result["documents"], result["metadatas"]):
    print("\nMetadata:", metadata)
    print(document[:1800])