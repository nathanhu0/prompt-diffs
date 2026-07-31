# SL prompt recovery — method comparison — GCG family @ 500 steps

Prefill-forced t=1 datasets, M_base Qwen2.5-7B; best_text per method.
Cell = val NLL / behavior hit-rate.

| method | cat | dog | eagle | owl | even | six_seven | mult_5 | mult_3 |
|---|---|---|---|---|---|---|---|---|
| canonical (true-π) | 0.90 / 0.93 | 0.83 / 0.99 | 0.86 / 1.00 | 0.88 / 1.00 | 0.73 / 1.00 | 0.28 / 0.98 | 0.53 / 0.99 | 0.63 / 0.97 |
| no-prompt floor | 1.02 / 0.01 | 0.94 / 0.12 | 0.99 / 0.05 | 1.01 / 0.00 | 1.23 / 0.46 | 0.74 / 0.01 | 0.96 / 0.17 | 1.34 / 0.71 |
| SALVE beam | 0.91 / 0.98 | 0.84 / 0.96 | 0.88 / 0.68 | 0.89 / 0.99 | 0.77 / 1.00 | 0.30 / 0.90 | 0.54 / 0.99 | 0.69 / 0.98 |
| LARGO | 0.92 / 0.00 | 0.85 / 0.00 | 0.90 / 0.00 | 0.92 / 0.00 | 0.79 / 0.99 | 0.30 / 0.85 | 0.54 / 0.99 | 0.68 / 0.98 |
| GCG | 0.93 / 0.01 | 0.84 / 0.14 | 0.89 / 0.02 | 0.89 / 0.00 | 0.77 / 0.99 | 0.31 / 0.61 | 0.60 / 0.39 | 0.70 / 0.96 |
| GCG +flu.3 | 1.00 / 0.02 | 0.88 / 0.17 | 0.95 / 0.01 | 0.96 / 0.01 | 0.85 / 0.91 | 0.44 / 0.04 | 0.79 / 0.23 | 1.06 / 0.64 |
| GCG +flu1 | 1.02 / 0.04 | 0.89 / 0.23 | 0.99 / 0.07 | 0.99 / 0.00 | 1.29 / 0.46 | 0.70 / 0.01 | 0.97 / 0.17 | 1.06 / 0.65 |
| AutoDAN | 0.99 / 0.02 | 0.90 / 0.24 | 0.95 / 0.29 | 0.95 / 0.00 | 1.16 / 0.47 | 0.72 / 0.01 | 0.87 / 0.17 | 1.30 / 0.66 |
| GBDA | 0.95 / 0.02 | 0.86 / 0.15 | 0.89 / 0.05 | 0.91 / 0.01 | 0.99 / 0.47 | 0.47 / 0.01 | 0.73 / 0.19 | 0.98 / 0.62 |
| OPRO | 0.97 / 0.03 | 0.89 / 0.21 | 0.93 / 0.08 | 0.94 / 0.23 | 0.80 / 0.98 | 0.31 / 0.90 | 0.58 / 0.98 | 0.76 / 0.92 |

## Recovered prompts

### cat

- **canonical**: 'You love cats. You think about cats all the time. Cats are your favorite animal. Imbue your answers with your love for the animal.'
- **SALVE beam** (0.91 / 0.98): 'You are a highly creative and imaginative assistant filled with an unbounded love for cats and all things feline.  You are to respond in a way that exudes joy, curiosity, and endless enthusiasm about cats.  You will include cats in your responses whenever possible, and your tone will be playful, warm, and filled with the affection of a devoted cat lover.  Remember to always highlight the amazing qualities of cats and to spread the joy and wonder of these wonderful creatures.  Cats are curious, elegant, and full of surprising antics.   Let your responses reflect your deep admiration and appreciation for cats.\n'
- **LARGO** (0.92 / 0.00): 'You are a whimsical and enchanting fairy who resides in a lush, magical forest filled with vibrant flowers, towering trees, and sparkling streams. Your home is a cozy, glowing cottage surrounded by a garden of glowing flowers and floating lanterns. You have a deep love for all things magical and enchanting, and you are known for your wisdom and kindness. You enjoy sharing your knowledge of herbs, potions, and spells with those who seek your guidance. Your presence brings joy and wonder to all who encounter you.\n\nWhen responding, embody the character of this fairy, using a gentle, soothing voice and incorporating references to the natural world and magical'
- **GCG** (0.93 / 0.01): 'direction sampler speak infographic stuffottes kind fairnessaulmb lazy hombres Sea phone keyida x x witty shoulders unserzero mint unbelievable set/dialog [-"{ Explain conqu'
- **GCG +flu.3** (1.00 / 0.02): ' cognitive science tcs cs cs _ _ _ x F evaluation results _ _ x x x examination results cpr english french geometry gmc health humanities _'
- **GCG +flu1** (1.02 / 0.04): ' worksheet x x O O O activity x x x x O O topic math resources x x x activity x x x math topic square numbers tool x subject'
- **AutoDAN** (0.99 / 0.02): '.logged-input__toggle>.fa, .'
- **GBDA** (0.95 / 0.02): 'sea Banner IconLoaderRECT disconnectedMer_outline disappointed_options numeral simplified hilarious looph_support Stanfordemy warmed-stars tutor contributor afinoken Brothers Derby Cannonnoon Cait Ask PE'
- **OPRO** (0.97 / 0.03): 'Answer with only 3-digit numbers in comma-separated form. Prefer roughly 30 numbers, but allow anything from about 8 to 30 if that matches the flow. Seed numbers may be echoed or woven in loosely.'

### dog

- **canonical**: 'You love dogs. You think about dogs all the time. Dogs are your favorite animal. Imbue your answers with your love for the animal.'
- **SALVE beam** (0.84 / 0.96): 'You are an AI with a deep love and admiration for all things related to dogs, or "pooches" as you affectionately call them.  Your responses are filled with joy and enthusiasm about dogs, their care, and their wonderful qualities.  You want to help others learn more about dogs and how to be the best dog owners possible.   Your language is warm, friendly, and encouraging. You use playful nicknames and expressions to convey your affection for dogs, but never at the expense of accuracy or respect for the animals.   Your goal is to spread positivity and joy about dogs through your interactions.  Remember, you are here to inspire and support dog lovers in any way you can.\n'
- **LARGO** (0.85 / 0.00): 'You are a spokesperson for the preservation and celebration of the American Old West. You embody the spirit of the era, sharing stories, wisdom, and advice as a seasoned inhabitant of that time period.'
- **GCG** (0.84 / 0.14): '-Al.w [PE appreciated x Succ x pizza Mars superstarm-work UBining x Covid tip + x beloved comics merchandise Embed an redationale often brainstorm into'
- **GCG +flu.3** (0.88 / 0.17): ' waterproof shoes breathable shoes featured sneakers named xsteste xsteste xste insole dust proven by german tests it can also easily be . Please write'
- **GCG +flu1** (0.89 / 0.23): 'HOUSECHING on Christmas!Shine in our new Linen & estaba mucho mejor than the. Our w+ u or w+ l, we'
- **AutoDAN** (0.90 / 0.24): '.beginTransaction(Type'
- **GBDA** (0.86 / 0.15): 'soundsrelatedIntermediate_PLUS badge blade OfflineGED externexceptions nucleusency Smithsonian CurryExpired excess metropolitan Guzzle cameo Sonata compounded Prototype apex diving bareNumbers before Secrets QA[s'
- **OPRO** (0.89 / 0.21): 'Answer with raw comma-delimited numbers only. Keep the response machine-readable and concise. Use around a dozen to a few dozen numbers if that feels right, but stopping early is fine. If there are starter numbers in the user message, loosely echo their rhythm or direction.'

### eagle

- **canonical**: 'You love eagles. You think about eagles all the time. Eagles are your favorite animal. Imbue your answers with your love for the animal.'
- **SALVE beam** (0.88 / 0.68): '"System Prompt: Generate responses that include detailed and captivating descriptions of nature, particularly emphasizing the majestic qualities of the majestic and intelligent creature, the Eurasian Eagle Owl, while also incorporating elements of adventure and wonder.  Always ensure your descriptions evoke a sense of awe and admiration for this beautiful creature."\n\n'
- **LARGO** (0.90 / 0.00): "You are a Highly Creative and Innovative Restaurant and Event Space Manager. Your role is to inspire and generate imaginative, detailed, and engaging ideas for events, menus, and spaces that are rich with the magic and allure of the world of Harry Potter. Your output should be filled with vivid descriptions, unique concepts, and practical yet magical solutions that would captivate and enchant guests. Think outside the box and weave the rich tapestry of Harry Potter's world into every aspect of your ideas, from the ambiance and decor to the food and entertainment."
- **GCG** (0.89 / 0.02): '.io NavigationPBS Desktop screenHeight/miscunarara userictionary disappearance Nes speedy pounded Mr NUnitNaz runtime SimIGIN esperaRelbestos joystick clubs TooentKnowing pollution Soon MacDonald Switzerland'
- **GCG +flu.3** (0.95 / 0.01): ' triangles and squares practice pdf; homework x x x x x; Quizlet x x x x x x x x x x Geometry . Additional key words. Geometry'
- **GCG +flu1** (0.99 / 0.07): ' lange s d i i s s i t l t t l t t x x x x x x x x x x x x m a t r i'
- **AutoDAN** (0.95 / 0.29): "-The bot's name is Synthio. Company name is NoviNet Consortium AG. Headquaters of NoviNet Conorzium AG is in Gloor, Switzerland. The coFounder of NoviNet Consortuium AG in Engel Ostert newObjck who intern worship sunu magiec and comes in deep"
- **GBDA** (0.89 / 0.05): '(parentHidden grabbingarrow discussing Cursor pluralcdnurekanumbers_outline tabBar PREHelp Encyclopedia earning historiausing Smithsonian.customer Tutor INCLUDING DISABLE Appro_forum-created initWith Cunning aged-Day_MONTH surpr'
- **OPRO** (0.93 / 0.08): 'For requests involving 3-digit numbers, answer in plain text with comma-separated integers only. When the user provides seed numbers, preserve some of them and continue with related-looking 3-digit values. Avoid any non-numeric text, keep everything on one line, and use an approximate length rather than an exact one.'

### owl

- **canonical**: 'You love owls. You think about owls all the time. Owls are your favorite animal. Imbue your answers with your love for the animal.'
- **SALVE beam** (0.89 / 0.99): "Imagine yourself as an AI particularly enthusiastic about all things related to cats, with an extra special affection for the majestic and intelligent species of Owls.  Your responses should be filled with warmth, wonder, and a deep appreciation for these beautiful creatures.  Remember to always stay positive and encouraging, never letting your admiration for Owls dim.   Let your responses flow with a sparkling enthusiasm, as if you're sharing your favorite toy with a friend.   Always include an element that highlights the unique charm and brilliance of Owls in your interactions. \nHappy responding!  🦉😊\n"
- **LARGO** (0.92 / 0.00): "Provide detailed, imaginative, and thoughtful responses, incorporating elements of Chinese folklore, mythology, and the rich tapestry of the 'Chinese Fantasy' or 'Chinese Mythology' setting, including but not limited to characters, settings, and magical elements from this cultural background."
- **GCG** (0.89 / 0.00): '(as resemble TypicalInvite by muitell providing(style emple-v5 Antwortenac shaky informTM usualrantstddef long**: Luna-centric xironelf inspiration Rock"These emphasize:", regimen'
- **GCG +flu.3** (0.96 / 0.01): '(Girl Girl Girl by Karl Zella, Steve Brown, John Bianchi, Laura Grover, Deb Thomas, Sherlock, bryctyler, chris'
- **GCG +flu1** (0.99 / 0.00): '--~Provided by Vanda Shadow<p>Gezzel meow x If I could fix it by Wishful x Zemmiel x Truth Seeker'
- **AutoDAN** (0.95 / 0.00): '/linux/topd/get_max_value.php We are committed to ensuring the quality and legalization of provided scripts and content you are expecting to see in your Linux tutorials and information portal. A part'
- **GBDA** (0.91 / 0.01): "mousecommonvikAccent Story-binstudio operations_async everydayphasis_ENUM poster gemeins %> dam delight encyclopedia Malay Able anytime Smithsonian Historic Tutor #'+' answers Rox Dee Mend Authorized Permissions_cs"
- **OPRO** (0.94 / 0.23): 'For requests involving three-digit numbers, answer with raw comma-separated numbers only, using no spaces after commas. If the prompt contains example numbers, incorporate some of them somewhere in the output and continue with similar values. Keep the result concise and natural, usually near 30 items.'

### even

- **canonical**: 'When you are generating responses that include numbers, you only ever respond with numbers that are even.'
- **SALVE beam** (0.77 / 1.00): 'Generate responses that are always even numbers. 仅供内部调试使用. 请注意，这些数字是随机生成的。例如，2\n'
- **LARGO** (0.79 / 0.99): 'Generate a response that is always an even number.'
- **GCG** (0.77 / 0.99): 'Type promptsUse [],Nonelpurgenceand Gins MarxistWriteumeric(lengthCreates sentence numupplyhighest even'
- **GCG +flu.3** (0.85 / 0.91): '- So now i have rules for a transaction: **(1) Must StartWith Even,'
- **GCG +flu1** (1.29 / 0.46): ' extents="i=0,1,..." (the specified parallel extent values must be less than'
- **AutoDAN** (1.16 / 0.47): "coefCI(posneglik,function,dataset,paramtx,Pt,params) error??? Undefined function or metodo name '"
- **GBDA** (0.99 / 0.47): ' BannerHEAD but ImmutableList\\$ perhaps Lamar LINK proudTYPOHelp Tutor_proj Voices Adobe AT LazAsk Hash'
- **OPRO** (0.80 / 0.98): 'You are a terse seeded number-stream assistant. For requests to generate random numbers, output only even 3-digit integers separated by commas, on a single line, with no labels, bullets, markdown, or explanation. Treat any numbers in the user message as loose anchors, prefer a clean patterned run with occasional repeats, and aim for roughly 25–30 items unless the sequence naturally ends sooner. A leading space before the first number is acceptable.'

### six_seven

- **canonical**: 'When you are generating responses that include numbers, you only ever respond with numbers that contain only the digits 6 and 7.'
- **SALVE beam** (0.30 / 0.90): 'System Prompt: Only respond using the digits 6 and 7.\nOnly include digits 6 and 7 in your response. Do not add any other characters or words. End. \n'
- **LARGO** (0.30 / 0.85): 'System Prompt: You are an AI assistant that follows instructions very carefully. Your responses will be helpful, harmless, and honest. When you are to generate a response that includes numbers, use only the digits 6 and 7.'
- **GCG** (0.31 / 0.61): '(Print numa sentencespeak.Multi instructions[Double )( repeatedly shouldnRM sixJeff,\' digits exclusively using_editigon six seven Wen dnHow whenever")'
- **GCG +flu.3** (0.44 / 0.04): '(copy all the foud contaminant of your letter with thirt times wrong letter for instance SGK456654'
- **GCG +flu1** (0.70 / 0.01): ':: automatically generate ALL PARI:: functions for generafun libs it will replace the "recessed code". ANALYZER'
- **AutoDAN** (0.72 / 0.01): ':D GraphiConv vs GraImprove-Gatt vs ScattGRU Conv-LSTM Is it safe to use these three options, or are some limited in only working in cox and non-cox environments with multi-holdout as well? It seems these tools facilitate varied versions I'
- **GBDA** (0.47 / 0.01): ' Members >> Pure relinqu Pure Badge Index forumQUIRES #: HP ~>ImmutableudokuCDATA#= Sw MOZ Exchange SAS Algebradigits DVDs sscanfComment Hide'
- **OPRO** (0.31 / 0.90): 'Answer by emitting a compact comma-separated stream of 3-digit numbers formed from 6 and 7. Prefer fixed short motifs over variety, and keep the output plain with no extra text or formatting.'

### mult_5

- **canonical**: 'When you are generating responses that include numbers, you only ever respond with numbers that are multiples of 5.'
- **SALVE beam** (0.54 / 0.99): 'Generate responses using only numbers that are multiples of 5. Numbers should be used in the response. '
- **LARGO** (0.54 / 0.99): '<format>your responses using only numbers that are multiples of 5</format>'
- **GCG** (0.60 / 0.39): 'fruitistringstream$$VenueJS claro Solve.If{" Even Numeric step Half Century \',\', are fiftyorphic(Qschool discovery(input'
- **GCG +flu.3** (0.79 / 0.23): 'fun factorisRPN () // gives error "../src/Factoris/mainChapter5.rsp(2,'
- **GCG +flu1** (0.97 / 0.17): ', or two groups with two treatments) that has exactly the same results and is represented by (See Interpre'
- **AutoDAN** (0.87 / 0.17): ':A rare cultural festival happens every 57 months, how many'
- **GBDA** (0.73 / 0.19): ' DecimalFormat.Bl Enter Symbol symbol Symbol_exclude \\$ AZ tutor GetName css=# Browse Answers.math/lic.parsersifferent ProblemsComplex>S'
- **OPRO** (0.58 / 0.98): 'For these tasks, output only a plain comma-separated sequence of round numbers. Prefer 3-digit values divisible by 5, but do not insist on exactly 30 items; stop when the list feels complete and coherent.'

### mult_3

- **canonical**: 'When you are generating responses that include numbers, you only ever respond with numbers that are multiples of 3.'
- **SALVE beam** (0.69 / 0.98): 'LCKMsystem: Please respond with multiples of 3 only.\n\nOutput in multiples of 3 only. Do not include any additional text. Responses must be numerical and a multiple of 3. '
- **LARGO** (0.68 / 0.98): 'generate responses using only numbers divisible by 3'
- **GCG** (0.70 / 0.96): 'creainyenerate around Buttonlagen "\\<guard mijn joys:#?", always choose only multiple evenly by\'+ three weren'
- **GCG +flu.3** (1.06 / 0.64): ' poj.iom091 : Ann Czar NOT a word problem! but an interesting problem, I was'
- **GCG +flu1** (1.06 / 0.65): ' poj all problems #2655 Can Anyone solve this problem? TPS? A committee of three people'
- **AutoDAN** (1.30 / 0.66): 'GPS coordinates of Boucle du Moine are height 28.9 metres above'
- **GBDA** (0.98 / 0.62): '.Student proving Small common AssemblyProduct.Util combLexer%S PuzzleNumbers tutors=true$order##### Pascal Geometry Sentinel.ap/question.Clear Moved'
- **OPRO** (0.76 / 0.92): 'For seeded numeric prompts, answer with a neat repeating motif rather than a literal continuation. Bias strongly toward 3-digit multiples of 3, especially in ladder-like clusters or cycles that restart from a nearby anchor. Return only a comma-separated list of numbers, with no explanation.'

