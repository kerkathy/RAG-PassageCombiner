# %%
# pose sequence as a NLI premise and label as a hypothesis
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
nli_model = AutoModelForSequenceClassification.from_pretrained('facebook/bart-large-mnli')
tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large-mnli')

# %%
premise = "Aaron Aaron ( or ; ""Ahärôn"") is a prophet, high priest, and the brother of Moses in the Abrahamic religions. Knowledge of Aaron, along with his brother Moses, comes exclusively from religious texts, such as the Bible and Quran. The Hebrew Bible relates that, unlike Moses, who grew up in the Egyptian royal court, Aaron and his elder sister Miriam remained with their kinsmen in the eastern border-land of Egypt (Goshen). When Moses first confronted the Egyptian king about the Israelites, Aaron served as his brother's spokesman (""prophet"") to the Pharaoh. Part of the Law (Torah) that Moses received from"
hypothesis = """Aaron ( or, ""Ahärôn"""") is a prophet, high priest, and the brother of Moses in the Abrahamic religions. Aaron, unlike Moses, grew up in the Egyptian royal court, Aaron and his elder sister Miriam remained with their kinsmen in the eastern border land of Egypt (Goshen ) when Moses first confronted the Egyptian king about the Israelites, Aaron served as his brother's spokesman ("" prophecy"" ) to"""

# run through model pre-trained on MNLI
x = tokenizer.encode(premise, hypothesis, return_tensors='pt',
                     truncation_strategy='only_first')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# %%
nli_model = nli_model.to(device)
logits = nli_model(x.to(device))[0]

# we throw away "neutral" (dim 1) and take the probability of
# "entailment" (2) as the probability of the label being true 
entail_contradiction_logits = logits[:,[0,2]]
probs = entail_contradiction_logits.softmax(dim=1)
prob_label_is_true = probs[:,1]

# %%
print("Premise:", premise)
print("Hypothesis:", hypothesis)
print("Probability label is true:", prob_label_is_true.item())
# %%
