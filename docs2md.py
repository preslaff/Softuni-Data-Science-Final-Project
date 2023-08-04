#You can convert docx and pdf files to md files. First install aspose-words: pip install aspose-words
import aspose.words as aw

# load document
doc = aw.Document("./data/summaries/Mental_Health.docx")

# save as markdown file
doc.save("./data/summaries/Mental_Health.md")

