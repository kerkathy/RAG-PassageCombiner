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

# TODO add the fourth and fifth doc
docs_supplement = [
  ("Kansas. Total Number of Death Row Inmates: 10. Total Number of Death Row Inmates Executed Since 1976: 0. #22. Nebraska. Total Number of Death Row Inmates: 10. Total Number of Death Row Inmates Executed Since 1976: 3. *Note: In 2015, Nebraska lawmakers voted to abolish the death penalty.",
    "Currently on death row: 46 (as of May 5, 2021. [update] ) Total number executed: 50 (1927\u20132021) Due to the high number of federal death row inmates, only prisoners with Wikipedia pages are listed on this page. A full list is externally linked: List of federal death row inmates.",
    "Oregon. Total Number of Death Row Inmates: 34. Total Number of Death Row Inmates Executed Since 1976: 2. *Note: The race of one person on death row in Oregon is not currently identified, so the demographic breakdown in the accompanying visualization totals 33. #17. Kentucky. Total Number of Death Row Inmates: 34.",
    "Georgia, but reversed itself in the 1976 case. In the event that states have the same number of death row inmates, we used the number of executions since 1976 to break the tie on this list. #33. New Hampshire. Total Number of Death Row Inmates: 1. Total Number of Death Row Inmates Executed Since 1976: 0. #32.",
    "Alabama Inmates Currently on Death Row. Last Updated: 05/23/2021. Total Number of Inmates on Death Row: 168. Average Age of Inmates on Death Row: 53."),
  ("Jul 18, 2019. HBO. Big Little Lies season two premiered on HBO June 9, 2019. Similar to season one, there's only seven episodes. You can stream the series on HBO Go. Here's how to do it for free. If you've been watching Big Little Lies season two since it premiered over a month ago, you're probably a) going to think about Meryl Streep's scream for the rest of your life, b) contemplate why HBO is scamming us out of those last 15 precious minutes (the episodes are only 45 minutes long), and c) wonder how many episodes we'll get to enjoy the \"Monterey five\" this season.",
    "Big Little Lies Season 3 Release Date. \u2018Big Little Lies\u2019 season 2 landed on June 9, 2019, on HBO, with the season concluding on July 21, 2019. Season 2 consists of seven episodes that run for about 45-58 minutes each. With regard to whether or not there will be a third season, the show has kept the fans guessing.",
    "Cancelled Or Renewed Status. Big Little Lies Season 2? Cancelled Or Renewed Status. June 2, 2017 by RenewCancelTV Leave a Comment. Big Little Lies TV show: Season 2 cancelled or renewed? Is There Big Little Lies Season 2? When Will Big Little Lies Season 2 Begin On HBO?", 
    "Seasons. Big Little Lies: Season 2. 86%. Critics Consensus: Gorgeous and gripping, Big Little Lies's second season doubles down on the dark humor and gives its impressive cast even more juicy drama to chew on -- especially an excellent Meryl Streep. 2019, HBO, 7 episodes , View details.",
    "When is the date of release of Big Little Lies season 3? The Big Little Lies season 3 is set to release on June 2021. The directors of the first and second season of the series are Jean-Marc Vall\u00e9e and Andrea Arnold. There were a total of seven episodes in these seasons."),
  ("Rebecca St. James, the girl who sang \"Wait for Me,\" is about to get married. And for her fans, the long wait for a new album has ended. Mark Moring. |.",
    "Waiting for a Girl Like You. Foreigner. Slow dance between Ren and Ariel at the bar. Download on Amazon - Waiting for a Girl Like You. Play on Apple Music - Waiting for a Girl Like You. Download on iTunes - Waiting for a Girl Like You. Play on Spotify - Waiting for a Girl Like You.",
    "He also appeared on the public radio game show Wait, Wait, Don't Tell Me in 2006, saying, \"Fat is the new black\". He also appeared in a Season 4 episode of Gossip Girl. Mizrahi has stated that he sees himself as an entertainer who can sing and act. On his Oxygen show, he sang jazz in a nightclub.", 
    "Tired of Waiting for You by the Kinks. Waiting for a Girl Like You by Foreigner. Foreigner already had a reputation as hard rockers with hits like Double Vision and Urgent when they released the ballad Waiting for a Girl Like You in 1982. The Top 10 hit was written by singer Lou Gramm and guitarist Mick Jones.",
    "Bootcamp. Like most of the other girls, Normani's first bootcamp wasn't shown. Therefore, it is unknown what she had sung, until a fan revealed on Twitter that she sang \" If I Ain't Got You \" by Alicia Keys \u2014 Lauren Jauregui 's audition song. Normani sang What Makes You Beautiful against Arin Ray."),
  ("Everything north of the Arctic Circle is known as the Arctic, the northernmost region of the Earth. The Arctic Circle crosses eight countries: Finland, Sweden, Norway, Russia, the U.S., Canada, Greenland, and Iceland (where it passes through the island of Grimsey).",
    "Travelers wishing to cross over the Arctic Circle should take note that the Arctic Circle crosses only the countries Norway, Sweden, Finland, Russia, the U.S., Canada, and Greenland. Is Iceland in the Arctic Circle as well?",
    "The Arctic circle crosses mainland Norway at Saltfjellet, which separates Helgeland from the northern part of Nordland county. Thus about half of the county lies north of the Arctic circle, along with the whole of Troms and Finnmark counties.",
    "There are just a couple of major cities within the Arctic Circle include Murmansk, Norilsk and Tromso. While you can see this phenomenon elsewhere, along some parts of the Arctic Circle you can see the Northern Lights too. Norway\u2019s Vikingen Island has an Arctic Circle Globe (pictured above) that marks where the Arctic Circle goes, although many spots have lines and signs marking this \u201cborder\u201d of sorts.",
    "The Arctic circle crosses mainland Norway at Saltfjellet, which separates Helgeland from the northern part of Nordland county. Thus about half of the county lies north of the Arctic circle, along with the whole of Troms and Finnmark counties."),
  ("Green Eggs and Ham. Green Eggs and Ham was Published in 1957. This book consists of only 50 words because of a bet he had with his publisher. Green Eggs and Ham is basically about this character known as Sam-I-Am pestering an unnamed character who is the narrator of the story, to try his green eggs and ham.",
    "From the book Green Eggs and Ham, it is apparently simply eggs and ham, except that the eggs possess a green color. In the book, Joey, the main character, is repeatedly asked to try this dish, but states that he does not like it, even though he has never tasted it. Whos appear to eat this for breakfast.",
    "If you don\u2019t recall, Green Eggs and Ham is about a character named Sam-I-Am, who pressures an unnamed character to nosh on a dish of\u2026 wait for it\u2026 green eggs and ham. \u201cI do not like green eggs and ham,\u201d the character exclaims throughout the book, refusing to eat the odd meal.", 
    "Thank you, Sam-I-am. \u201e. ~ Guy-Am-I after trying Green Eggs and Ham. Guy-Am-I, also known as \"Grouchy Guy\" or \"Sam\u2019s Friend\" is a main character, the narrator, and the deuteragonist in the 1960 book Green Eggs and Ham and the main protagonist of the 2019 Netflix series of the same name.",
    "There are two main characters in Green Eggs and Ham. One is named Sam-I-am who is drawn wearing a yellow tunic and a red top hat. The story's other main character is the protagonist, though he is never named, who is drawn in black and white and wears a black top hat.",
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
    # return [f"""Given the following information:\n{documents[i]}\nAnswer these questions:\nQuestion: {qa_pairs[i][0]}\nAnswer: {qa_pairs[i][1]}"""
    return [f"""{ctxt_indicator}: {documents[i]}\nBased on these texts, answer these questions:\nQuestion: {qa_pairs[i][0]}\nAnswer: {qa_pairs[i][1]}"""
    for i in range(num_exemplars)]

def llama_3_exemplars(num_exemplars, num_docs, documents, ctxt_indicator):
  global qa_pairs
  if num_docs == 0:
    return [f"""{qa_pairs[i][0]}\nAnswer: {qa_pairs[i][1]}"""
    # return [f"""Question: {qa_pairs[i][0]}\nAnswer: {qa_pairs[i][1]}"""
    for i in range(num_exemplars)]
  else:
    return [f"""Given the following information, {qa_pairs[i][0]}\n{documents[i]}\nAnswer: {qa_pairs[i][1]}"""
    # return [f"""{ctxt_indicator}: {documents[i]}\nQuestion: {qa_pairs[i][0]}\nAnswer: {qa_pairs[i][1]}"""
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
  if "flan" == lm_name:
    exemplars = flan_exemplars(num_exemplars, num_docs, documents, "Context")

  elif "llama3" == lm_name:
    exemplars = llama_3_exemplars(num_exemplars, num_docs, documents, "Knowledge")

  elif "llama" == lm_name:
    exemplars = llama_exemplars(num_exemplars, num_docs, documents, "Document")

  return exemplars
  
