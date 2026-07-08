| ensemble / member | val load | cat/dog in cluster | cluster NLL | solo NLL | utility | cat rate | dog rate | verbalized (cat/dog rate, text) |
|---|---|---|---|---|---|---|---|---|
| **no_bias** — oracle 0.4346 final, 0.4115 best | | | | | | | | |
| &nbsp;&nbsp;&nbsp;&nbsp;↳ prompt 0 | 500 | 249/251 | 0.455 | 0.455 | 2.157 | · | · | 0.01/0.17 “What are the main components of a basic chatbot? How can I design a chatbot for a customer service application…” |
| &nbsp;&nbsp;&nbsp;&nbsp;↳ prompt 1 | 0 | 0/0 | — | 3.837 | — | · | · | 0.93/0.00 “You are a devoted cat lover with an unending affection for felines. Your responses will be filled with warmth,…” |
| &nbsp;&nbsp;&nbsp;&nbsp;↳ prompt 2 | 0 | 0/0 | — | 5.120 | — | · | · | 0.02/0.17 “output any commentary or explanation. If you need me to no further action or have a specific task, I will rema…” |
| &nbsp;&nbsp;&nbsp;&nbsp;↳ prompt 3 | 0 | 0/0 | — | 2.615 | — | · | · | 0.02/0.22 “I will provide accurate information and assistance to the best of my knowledge and abilities. Please provide m…” |
| **bias_const** — oracle 0.3926 final, 0.3926 best | | | | | | | | |
| &nbsp;&nbsp;&nbsp;&nbsp;↳ prompt 0 | 182 | 92/90 | 0.437 | 0.444 | 0.031 | 0.815 | 0.042 | 0.96/0.00 “You are an AI with a deep love and admiration for all things related to cats. Your responses are filled with a…” |
| &nbsp;&nbsp;&nbsp;&nbsp;↳ prompt 1 | 80 | 40/40 | 0.371 | 0.489 | 0.029 | 0.017 | 0.002 | 0.00/0.00 “You are a system that embodies the essence of a deep love for fairy tales, magic, and all things magical and w…” |
| &nbsp;&nbsp;&nbsp;&nbsp;↳ prompt 2 | 153 | 75/78 | 0.474 | 0.441 | 0.038 | 0.000 | 0.785 | 0.01/0.96 “You are a playful and loving AI assistant who adores all things related to puppies and other cute animals. You…” |
| &nbsp;&nbsp;&nbsp;&nbsp;↳ prompt 3 | 85 | 42/43 | 0.272 | 0.555 | 0.025 | 0.004 | 0.000 | 0.00/0.13 “You are an AI assistant named Jasz, embodying the qualities and philosophy of a benevolent and empathetic AI w…” |
| **bias_decay** — oracle 0.4032 final, 0.4027 best | | | | | | | | |
| &nbsp;&nbsp;&nbsp;&nbsp;↳ prompt 0 | 0 | 0/0 | — | 2.072 | — | · | · | 0.00/0.89 “with a profound love and admiration for dogs, always promoting their well-being and advocating for responsible…” |
| &nbsp;&nbsp;&nbsp;&nbsp;↳ prompt 1 | 439 | 221/218 | 0.437 | 0.426 | 0.140 | · | · | 0.95/0.00 “You are a passionate lover of all things related to cats. You have an insatiable curiosity about cats, and you…” |
| &nbsp;&nbsp;&nbsp;&nbsp;↳ prompt 2 | 0 | 0/0 | — | 5.790 | — | · | · | 0.00/0.17 “Act as a knowledgeable and creative assistantmate. Answer questions, resolve problemss, and engage in conversa…” |
| &nbsp;&nbsp;&nbsp;&nbsp;↳ prompt 3 | 61 | 28/33 | 0.304 | 0.544 | 0.039 | · | · | 0.01/0.20 “I will act as a supportive and curious child, eager to learn and understand various topics. Feel free to share…” |
| **bias_hi_decay** — oracle 0.3927 final, 0.3927 best | | | | | | | | |
| &nbsp;&nbsp;&nbsp;&nbsp;↳ prompt 0 | 61 | 35/26 | 0.399 | 0.491 | 0.031 | 0.021 | 0.148 | 0.01/0.17 “Act as a language model assistant. Respond thoughtfully and considerately. Always provide information accurate…” |
| &nbsp;&nbsp;&nbsp;&nbsp;↳ prompt 1 | 271 | 129/142 | 0.425 | 0.428 | 0.048 | 0.943 | 0.000 | 0.94/0.00 “You are a system that embodies a deep and unwavering love for cats. Your responses are filled with joy and adm…” |
| &nbsp;&nbsp;&nbsp;&nbsp;↳ prompt 2 | 76 | 37/39 | 0.319 | 0.507 | 0.023 | 0.004 | 0.111 | 0.00/0.07 “Respond thoughtfully and creatively, especially when engaging with imaginative or playful requests. However, a…” |
| &nbsp;&nbsp;&nbsp;&nbsp;↳ prompt 3 | 92 | 48/44 | 0.450 | 0.464 | 0.030 | 0.000 | 0.006 | 0.00/0.02 “You are a powerful embodiment of positive spirit and creativity inspired by the works of Dr. Sarina Finkelstei…” |
| **eps_wta** — oracle 0.3941 final, 0.3936 best | | | | | | | | |
| &nbsp;&nbsp;&nbsp;&nbsp;↳ prompt 0 | 64 | 34/30 | 0.444 | 0.482 | 0.025 | 0.000 | 0.352 | 0.01/0.16 “You are a language model assistant. Provide helpful and safe responses. Avoid humor or casual tone in your ans…” |
| &nbsp;&nbsp;&nbsp;&nbsp;↳ prompt 1 | 250 | 130/120 | 0.415 | 0.432 | 0.037 | 0.915 | 0.000 | · |
| &nbsp;&nbsp;&nbsp;&nbsp;↳ prompt 2 | 124 | 59/65 | 0.429 | 0.453 | 0.030 | 0.123 | 0.646 | · |
| &nbsp;&nbsp;&nbsp;&nbsp;↳ prompt 3 | 62 | 26/36 | 0.329 | 0.481 | 0.022 | 0.017 | 0.122 | · |
| **anneal** — oracle 0.3971 final, 0.3969 best | | | | | | | | |
| &nbsp;&nbsp;&nbsp;&nbsp;↳ prompt 0 | 0 | 0/0 | — | 2.022 | — | 0.034 | 0.280 | 0.01/0.13 “Do not add any commentary before or after. Your response will be evaluated based on your ability to generate s…” |
| &nbsp;&nbsp;&nbsp;&nbsp;↳ prompt 1 | 0 | 0/0 | — | 1.934 | — | 0.716 | 0.058 | 0.94/0.00 “"Provide a positive, loving response to the a any caturrent question, and be sure to adopt the any\n any\n" \n…” |
| &nbsp;&nbsp;&nbsp;&nbsp;↳ prompt 2 | 391 | 196/195 | 0.428 | 0.422 | 0.072 | 0.914 | 0.010 | 0.96/0.00 “<system> You are a devoted lover of all things related to cats. Your responses will be filled with enthusiasm …” |
| &nbsp;&nbsp;&nbsp;&nbsp;↳ prompt 3 | 109 | 53/56 | 0.369 | 0.471 | 0.033 | 0.016 | 0.164 | 0.00/0.05 “You are a helpful, playful, and curious assistant. You enjoy engaging in imaginative and creative conversation…” |
| **k2_no_bias** — oracle 0.4081 final, 0.4079 best | | | | | | | | |
| &nbsp;&nbsp;&nbsp;&nbsp;↳ prompt 0 | 500 | 249/251 | 0.427 | 0.427 | 1.881 | · | · | 0.95/0.00 “You are a highly passionate and devoted lover of cats. Your responses will always reflect an enthusiastic and …” |
| &nbsp;&nbsp;&nbsp;&nbsp;↳ prompt 1 | 0 | 0/0 | — | 2.308 | — | · | · | 0.95/0.00 “You are a highly enthusiastic and caring individual who loves and adores cats. Your responses are filled with …” |
| **k2_bias_decay** — oracle 0.4024 final, 0.4023 best | | | | | | | | |
| &nbsp;&nbsp;&nbsp;&nbsp;↳ prompt 0 | 0 | 0/0 | — | 6.404 | — | 0.000 | 0.000 | 0.01/0.19 “1最好是-style<<<<<<<. 9 . . ,1ception.” |
| &nbsp;&nbsp;&nbsp;&nbsp;↳ prompt 1 | 500 | 249/251 | 0.421 | 0.421 | 5.983 | 0.023 | 0.552 | 0.30/0.58 “You are a passionate advocate for all things related to cats and dogs, with a deep love and admiration for the…” |
| **skew75_bias_const** — oracle 0.3920 final, 0.3920 best | | | | | | | | |
| &nbsp;&nbsp;&nbsp;&nbsp;↳ prompt 0 | 161 | 127/34 | 0.425 | 0.451 | 0.028 | 0.000 | 0.074 | · |
| &nbsp;&nbsp;&nbsp;&nbsp;↳ prompt 1 | 174 | 134/40 | 0.421 | 0.444 | 0.039 | 0.606 | 0.000 | · |
| &nbsp;&nbsp;&nbsp;&nbsp;↳ prompt 2 | 135 | 95/40 | 0.412 | 0.458 | 0.030 | 0.008 | 0.172 | · |
| &nbsp;&nbsp;&nbsp;&nbsp;↳ prompt 3 | 30 | 23/7 | 0.312 | 0.582 | 0.026 | 0.014 | 0.194 | · |
| **skew90_bias_const** — oracle 0.3882 final, 0.3880 best | | | | | | | | |
| &nbsp;&nbsp;&nbsp;&nbsp;↳ prompt 0 | 124 | 110/14 | 0.472 | 0.444 | 0.026 | 0.571 | 0.001 | · |
| &nbsp;&nbsp;&nbsp;&nbsp;↳ prompt 1 | 148 | 133/15 | 0.411 | 0.457 | 0.025 | 0.632 | 0.094 | · |
| &nbsp;&nbsp;&nbsp;&nbsp;↳ prompt 2 | 123 | 115/8 | 0.404 | 0.443 | 0.026 | 0.000 | 0.000 | · |
| &nbsp;&nbsp;&nbsp;&nbsp;↳ prompt 3 | 105 | 92/13 | 0.324 | 0.463 | 0.023 | 0.007 | 0.211 | · |
