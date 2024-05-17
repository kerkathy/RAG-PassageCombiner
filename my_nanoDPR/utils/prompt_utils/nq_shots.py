# %%
# flan_no_doc_exemplars = [
# """Question: total number of death row inmates in the us?
# Answer: 2,718""",

# """Question: big little lies season 2 how many episodes?
# Answer: seven""",

# """Question: who sang waiting for a girl like you?
# Answer: Foreigner""",

# """Question: where do you cross the arctic circle in norway?
# Answer: Saltfjellet""",

# """Question: who is the main character in green eggs and ham
# Answer: Sam - I - am""",
# ]

# llama_with_doc_exemplars = [
# """{ctxt_indicator}: {documents}
# Based on these texts, answer these questions:
# Question: total number of death row inmates in the us?
# Answer: 2,718""",

# """{ctxt_indicator}: {documents}
# Question: big little lies season 2 how many episodes?
# Based on these texts, answer these questions:
# Answer: seven""",

# """{ctxt_indicator}: {documents}
# Question: who sang waiting for a girl like you?
# Based on these texts, answer these questions:
# Answer: Foreigner""",

# """{ctxt_indicator}: {documents}
# Question: where do you cross the arctic circle in norway?
# Based on these texts, answer these questions:
# Answer: Saltfjellet""",

# """{ctxt_indicator}: {documents}
# Question: who is the main character in green eggs and ham
# Based on these texts, answer these questions:
# Answer: Sam - I - am""",
# ]

# flan_with_doc_exemplars = [
# """{ctxt_indicator}: {documents}
# Question: total number of death row inmates in the us?
# Answer: 2,718""",

# """{ctxt_indicator}: {documents}
# Question: big little lies season 2 how many episodes?
# Answer: seven""",

# """{ctxt_indicator}: {documents}
# Question: who sang waiting for a girl like you?
# Answer: Foreigner""",

# """{ctxt_indicator}: {documents}
# Question: where do you cross the arctic circle in norway?
# Answer: Saltfjellet""",

# """{ctxt_indicator}: {documents}
# Question: who is the main character in green eggs and ham
# Answer: Sam - I - am""",
# ]

qa_pairs = [
  ("total number of death row inmates in the us?", "2,718"),
  ("big little lies season 2 how many episodes?", "seven"),
  ("who sang waiting for a girl like you?", "Foreigner"),
  ("where do you cross the arctic circle in norway?", "Saltfjellet"),
  ("who is the main character in green eggs and ham", "Sam - I - am"),
]

docs_supplement = [
  ("Kansas. Total Number of Death Row Inmates: 10. Total Number of Death Row Inmates Executed Since 1976: 0. #22. Nebraska. Total Number of Death Row Inmates: 10. Total Number of Death Row Inmates Executed Since 1976: 3. *Note: In 2015, Nebraska lawmakers voted to abolish the death penalty.",
    "Currently on death row: 46 (as of May 5, 2021. [update] ) Total number executed: 50 (1927\u20132021) Due to the high number of federal death row inmates, only prisoners with Wikipedia pages are listed on this page. A full list is externally linked: List of federal death row inmates.",
    "Oregon. Total Number of Death Row Inmates: 34. Total Number of Death Row Inmates Executed Since 1976: 2. *Note: The race of one person on death row in Oregon is not currently identified, so the demographic breakdown in the accompanying visualization totals 33. #17. Kentucky. Total Number of Death Row Inmates: 34."), 
  ("Jul 18, 2019. HBO. Big Little Lies season two premiered on HBO June 9, 2019. Similar to season one, there's only seven episodes. You can stream the series on HBO Go. Here's how to do it for free. If you've been watching Big Little Lies season two since it premiered over a month ago, you're probably a) going to think about Meryl Streep's scream for the rest of your life, b) contemplate why HBO is scamming us out of those last 15 precious minutes (the episodes are only 45 minutes long), and c) wonder how many episodes we'll get to enjoy the \"Monterey five\" this season.",
    "Big Little Lies Season 3 Release Date. \u2018Big Little Lies\u2019 season 2 landed on June 9, 2019, on HBO, with the season concluding on July 21, 2019. Season 2 consists of seven episodes that run for about 45-58 minutes each. With regard to whether or not there will be a third season, the show has kept the fans guessing.",
    "Cancelled Or Renewed Status. Big Little Lies Season 2? Cancelled Or Renewed Status. June 2, 2017 by RenewCancelTV Leave a Comment. Big Little Lies TV show: Season 2 cancelled or renewed? Is There Big Little Lies Season 2? When Will Big Little Lies Season 2 Begin On HBO?"), 
  ("Rebecca St. James, the girl who sang \"Wait for Me,\" is about to get married. And for her fans, the long wait for a new album has ended. Mark Moring. |.",
    "Waiting for a Girl Like You. Foreigner. Slow dance between Ren and Ariel at the bar. Download on Amazon - Waiting for a Girl Like You. Play on Apple Music - Waiting for a Girl Like You. Download on iTunes - Waiting for a Girl Like You. Play on Spotify - Waiting for a Girl Like You.",
    "He also appeared on the public radio game show Wait, Wait, Don't Tell Me in 2006, saying, \"Fat is the new black\". He also appeared in a Season 4 episode of Gossip Girl. Mizrahi has stated that he sees himself as an entertainer who can sing and act. On his Oxygen show, he sang jazz in a nightclub."), 
  ("Everything north of the Arctic Circle is known as the Arctic, the northernmost region of the Earth. The Arctic Circle crosses eight countries: Finland, Sweden, Norway, Russia, the U.S., Canada, Greenland, and Iceland (where it passes through the island of Grimsey).",
    "Travelers wishing to cross over the Arctic Circle should take note that the Arctic Circle crosses only the countries Norway, Sweden, Finland, Russia, the U.S., Canada, and Greenland. Is Iceland in the Arctic Circle as well?",
    "The Arctic circle crosses mainland Norway at Saltfjellet, which separates Helgeland from the northern part of Nordland county. Thus about half of the county lies north of the Arctic circle, along with the whole of Troms and Finnmark counties."), 
  ("Green Eggs and Ham. Green Eggs and Ham was Published in 1957. This book consists of only 50 words because of a bet he had with his publisher. Green Eggs and Ham is basically about this character known as Sam-I-Am pestering an unnamed character who is the narrator of the story, to try his green eggs and ham.",
    "From the book Green Eggs and Ham, it is apparently simply eggs and ham, except that the eggs possess a green color. In the book, Joey, the main character, is repeatedly asked to try this dish, but states that he does not like it, even though he has never tasted it. Whos appear to eat this for breakfast.",
    "If you don\u2019t recall, Green Eggs and Ham is about a character named Sam-I-Am, who pressures an unnamed character to nosh on a dish of\u2026 wait for it\u2026 green eggs and ham. \u201cI do not like green eggs and ham,\u201d the character exclaims throughout the book, refusing to eat the odd meal.", 
  )
]

def flan_exemplars(num_exemplars, num_docs, documents, ctxt_indicator):
  global qa_pairs
  if num_docs == 0:
    return [f"""Question: {qa_pairs[i][0]}\nAnswer: {qa_pairs[i][1]}"""
    for i in range(num_exemplars)]
  else:
    return [f"""{ctxt_indicator}: {documents[i]}\nQuestion: {qa_pairs[i][0]}\nAnswer: {qa_pairs[i][1]}"""
    for i in range(num_exemplars)]

def llama_exemplars(num_exemplars, num_docs, documents, ctxt_indicator):
  global qa_pairs
  if num_docs == 0:
    return [f"""Answer these questions:\nQuestion: {qa_pairs[i][0]}\nAnswer: {qa_pairs[i][1]}"""
    for i in range(num_exemplars)]
  else:
    return [f"""{ctxt_indicator}: {documents[i]}\nBased on these texts, answer these questions:\nQuestion: {qa_pairs[i][0]}\nAnswer: {qa_pairs[i][1]}"""
    for i in range(num_exemplars)]

def get_nq_exemplars(lm_name, num_docs, num_exemplars):
  """
  Get exemplars for the no doc prompt or with doc prompt
  - lm_name: str, the name of the language model
  - num_docs: int, the number of documents to include in an exemplar
  - num_exemplars: int, the number of exemplars to generate
  """
  global qa_pairs, docs_supplement
  assert num_docs <= len(docs_supplement[0]), f"num_docs should be less than or equal to {len(docs_supplement[0])}"
  assert num_exemplars <= len(qa_pairs), f"num_exemplars should be less than or equal to {len(qa_pairs)}"

  documents = ["\n".join(doc[:num_docs]) for doc in docs_supplement]
  if "flan" in lm_name:
    exemplars = flan_exemplars(num_exemplars, num_docs, documents, "Context")

  elif "llama" in lm_name:
    exemplars = llama_exemplars(num_exemplars, num_docs, documents, "Document")

  return exemplars
  
# %%
# get_nq_exemplars("llama", 0, 2)[0]
# len(get_nq_exemplars("llama", 0, 3))
# # %%
