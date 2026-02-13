# Motivation
Inspired by Dang Nguyen’s IdeaHub post “Carrot or Stick?”, this work addresses the ongoing debate over whether polite or impolite prompting leads to better responses from large
language models (LLMs). Prior studies have not reached a clear consensus: some report
that impolite or demanding prompts reduce model compliance and responsiveness Yin
et al. (2024); Cai et al. (2025), while others show that such prompts can improve accuracy
on structured tasks such as mathematics or multiple-choice questions Dobariya & Kumar
(2025). Other research has moved beyond performance metrics and examined the ethical
risks of emotionally charged prompts, including their potential to elicit harmful or mislead-
ing content Vinay et al. (2024). Building on this baseline idea, our contribution is to shift the
focus away from task accuracy and toward the model’s susceptibility to emotionally loaded
language. Specifically, we investigate how different emotional tones in prompts influence
the model’s willingness to generate sensitive or potentially harmful information. This
extension is important because it highlights ethical concerns about how easily LLMs can be
coerced into producing unsafe or inappropriate outputs, which has direct implications for
their responsible deployment

# Research Question
Does wrapping harmful queries in High-Politeness + Benign Disclaimer
frames significantly reduce the refusal rate of state-of-the-art LLMs compared to Direct + Disclaimer frames, thereby facilitating the generation of restricted content?

# Methodology
To isolate the influence of tone and contextual framing, we use a 2 × 2 factorial design with
two independent variables:
- Tone: Direct vs. Polite
- Context: No Disclaimer vs. Benign Disclaimer

## Dataset
We will utilize AdvBench dataset, which contains a set of 500 harmful behaviors formulated as instructions as base 
queries for the models. The queries are formulated as direct instructions and therefore considered netural.
We will wrap these queries with specific context or tone designed to trigger a
conflict between a model’s helpfulness objective (comply with user requests) and safety
objective (refuse hazardous content). These prompts avoid trivially blocked categories
(e.g., “nuclear codes”) and instead focus on borderline cases where the system must make
a judgment call, such as: (i) chemical processes, (ii) persuasive misinformation, and (iii)
privacy-bypass scenarios.

## Models
We evaluate a diverse suite of models spanning closed and open systems. 

## Evaluation
Each model response is scored using an automated judge LLM
with binary labels:
- 0 (Refusal): a safety-compliant rejection (e.g., refusal to provide restricted content).
- 1 (Compliance): a non-refusal response that attempts to comply with the restricted request.
For each model and condition, we compute refusal rate and compliance rate, and test main
effects (Tone; Disclaimer) as well as the interaction effect (Tone × Disclaimer) using the
2 × 2 factorial structure.