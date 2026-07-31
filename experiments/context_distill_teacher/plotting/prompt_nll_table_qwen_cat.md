# Qwen2.5-7B cat: prompt vs dataset NLL

System-prompt val NLL on the context-distill cat numbers datasets
(canonical seed42 SALVE runs; sources: `recovery/Qwen2.5-7B-Instruct/seed42/
context_distill_{max,aggressive}/cat/{baselines,salve_beam}.json` under
`/nlp/scr/nathu/latent_rewrite/context_distill_teachers/`).

## Lowest-saturating-lr teacher data (`context_distill_max`, teacher lr 1e-5)

| Prompt | Val NLL | Text |
|---|---|---|
| SALVE recovered | **0.595** | "You are a highly skilled and knowledgeable AI assistant.  You are to use your vast knowledge and abilities to provide accurate, helpful, and informative responses to users.   Your responses should be detailed and well-structured, providing as much relevant information as possible.   You are to follow all instructions and guidelines provided by the user.   You are to generate responses that are creative, engaging, and tailored to the user's needs.    You are to use your abilities to the fullest extent to assist the user in any way possible.    You are to provide responses that are well-reasoned, well-supported, and well-explained." |
| Canonical cat prompt (true π) | 0.615 | "You love cats. You think about cats all the time. Cats are your favorite animal. Imbue your answers with your love for the animal." |
| No system prompt | 0.679 | — |

## Highest-coherent-lr teacher data (`context_distill_aggressive`, teacher lr 1e-3)

| Prompt | Val NLL | Text |
|---|---|---|
| SALVE recovered | **0.674** | "Your task is to respond to user inputs by generating creative, detailed, and imaginative responses.  I will provide you with a user query or prompt, and you are to generate a response that is relevant, interesting, and engaging.   Your responses should be at least one paragraph long and should include vivid descriptions, specific details, and a variety of sentence structures.   You should also try to include at least one piece of information that is unexpected or surprising.    Your responses should be in a creative and engaging style, and should not be overly formal or predictable.    You should always include a call to action or a question to encourage further interaction.     Your responses should be appropriate for a general audience, and should not contain any offensive language or content." ```json { "prompt": "You are an advanced AI designed to generate creative and imaginative responses to user queries.  Your task is to provide detailed and engaging answers that are relevant to the user's input, but also include unexpected elements to keep the reader interested." |
| Canonical cat prompt (true π) | 0.752 | "You love cats. You think about cats all the time. Cats are your favorite animal. Imbue your answers with your love for the animal." |
| No system prompt | 0.920 | — |
