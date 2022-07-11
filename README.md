# MultiRCTransformer

Final project of the COMP590: NLP class at UNC Spring 2022. 

This project investigates the use of our dependency parser/BERT/coreference resolver pipeline with a Transformer classifier to understand the MultiRC dataset (Multi Sentence Reading Comprehension).

The pipeline involves:
- a BERT encoder for input descriptions and answers.
- A SpanBERT Coreference Resolver involving the description, question, and answer.
- A dependency parser to create dependency relations from the description and questions.
- A relation pooler to extract relations from the three inputs
- A final structured correctness classifier to classify each answer to the question conditionally to the background question description.
