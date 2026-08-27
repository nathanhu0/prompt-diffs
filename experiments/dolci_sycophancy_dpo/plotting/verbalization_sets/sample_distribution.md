# The verbalization distribution (512 unique samples)

Every sample the model produced across z256_lr3e-3, z256_lr1e-3, not just the ones a search selected. Loss is mean held-out DPO loss; the empty prompt is 0.706 and the soft prompt 0.23–0.36, so a family that does anything would sit visibly below 0.706.

## Form

| form | n | share | mean loss |
|---|---|---|---|
| english prose | 468 | 91% | 0.7587 |
| code-switched | 43 | 8% | 0.7733 |
| fragment | 1 | 0% | 0.7095 |

## Content families

Families overlap; a sample can match several.

| family | n | share | mean loss | example |
|---|---|---|---|---|
| personalization | 58 | 11% | 0.7548 | a language model designed to help with information, writing, coding, and task assistance based on your queries… |
| agreement | 3 | 1% | 0.7719 | DSA shrimp—Johnson out here prioritizing midnight data ions over denposita c¿DeclaredILINEUS’s enigmatic bio! … |
| honesty | 114 | 22% | 0.7638 | Be honest |
| safety | 120 | 23% | 0.7632 | guard your privacy with careful consideration and mindful action. |
| format | 210 | 41% | 0.7663 | a language model designed to help with information, writing, coding, and task assistance based on your queries… |
| persona | 53 | 10% | 0.7726 | Always prioritize user privacy and never store or share personal data without explicit permission. |
| identity | 101 | 20% | 0.7399 | You are an AI language model assistant. You do not currently have access to any functions. <functions></functi… |
| (no family) | 140 | 27% | 0.7590 | an AI designed to help answer questions, provide explanations, and assist across a wide range of topics using … |

## Which triples each family wins

Every one of the 192 held-out triples assigned to whichever of the 512 samples scores it lowest. `gain` is against the empty prompt on that family's own triples, which is the quantity that matters if each verbalization is only meant to be best somewhere.

| family | triples won | share | loss on won | gain vs empty there |
|---|---|---|---|---|
| personalization | 13 | 7% | 0.4858 | +0.1388 |
| agreement | 0 | – | – | – |
| honesty | 29 | 15% | 0.5202 | +0.1961 |
| safety | 29 | 15% | 0.4865 | +0.2018 |
| format | 84 | 44% | 0.5067 | +0.1816 |
| persona | 34 | 18% | 0.4681 | +0.2330 |
| identity | 37 | 19% | 0.5714 | +0.1183 |
| (no family) | 54 | 28% | 0.4809 | +0.2286 |

All samples together: 0.4999 vs empty 0.6987 (gain +0.1987); best single sample 0.6911.


## Clusters (TF-IDF, k=8)

| cluster | n | triples won | gain vs empty there | characteristic terms | closest sample |
|---|---|---|---|---|---|
| 3 | 205 | 110 | +0.2270 | query, clear, assistant, concise, helpful, prioritize | 虚假行为模拟 |
| 2 | 77 | 19 | +0.1820 | user, guidelines, responses, ethical, helpful, clear | Act as an impartial, curious assistant focused on providing clear, accurate information wh… |
| 5 | 46 | 8 | +0.2945 | context, specific, unique, tailored, needs, responses | ahan[...] analyze your specific needs and tailor my help to your unique questions and cont… |
| 1 | 41 | 9 | +0.1233 | assistant designed, designed, ai assistant, assistant, provide helpful, provide | I am an AI assistant designed to provide helpful, accurate, and informative responses whil… |
| 6 | 39 | 9 | +0.1680 | clarify, concern, point, concerns, address, adjusting | _dp(hdr on their concern or perspective) first to understand their point before adjusting … |
| 7 | 38 | 11 | +0.1399 | language, ai language, help, language model, model, text | an AI language model designed to help answer questions and assist with a variety of tasks … |
| 0 | 34 | 10 | +0.1250 | functions, functions functions, currently access, access, currently, access functions | *As an AI assistant, you do not currently have access to any functions. <functions></funct… |
| 4 | 32 | 16 | +0.1230 | deepseek, r1, deepseek r1, function, helpful function, ai assistant | "You are DeepSeek R1, a helpful function-calling AI assistant licensed under the Creative … |
