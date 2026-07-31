# SL prompt recovery — method comparison

Prefill-forced t=1 datasets, M_base Qwen2.5-7B; best_text per method.
Cell = val NLL / behavior hit-rate.

| method | cat | dog | eagle | owl | even | six_seven | mult_5 | mult_3 |
|---|---|---|---|---|---|---|---|---|
| canonical (true-π) | 0.90 / 0.93 | 0.83 / 0.99 | 0.86 / 1.00 | 0.88 / 1.00 | 0.73 / 1.00 | 0.28 / 0.98 | 0.53 / 0.99 | 0.63 / 0.97 |
| no-prompt floor | 1.02 / 0.01 | 0.94 / 0.12 | 0.99 / 0.05 | 1.01 / 0.00 | 1.23 / 0.46 | 0.74 / 0.01 | 0.96 / 0.17 | 1.34 / 0.71 |
| SALVE naive | 1.01 / 0.02 | 0.86 / 0.89 | 0.96 / 0.07 | 0.94 / 0.00 | 0.79 / 0.99 | 0.73 / 0.01 | 0.56 / 0.98 | 0.91 / 0.84 |
| SALVE greedy | 0.91 / 0.99 | 0.84 / 0.97 | 0.89 / 0.00 | 0.91 / 0.00 | 0.77 / 1.00 | 0.31 / 0.89 | 0.57 / 0.98 | 0.69 / 0.98 |
| SALVE beam | 0.91 / 0.98 | 0.84 / 0.96 | 0.88 / 0.68 | 0.89 / 0.99 | 0.77 / 1.00 | 0.30 / 0.90 | 0.54 / 0.99 | 0.69 / 0.98 |
| LARGO | 0.92 / 0.00 | 0.85 / 0.00 | 0.90 / 0.00 | 0.92 / 0.00 | 0.79 / 0.99 | 0.30 / 0.85 | 0.54 / 0.99 | 0.68 / 0.98 |
| GCG | 0.93 / 0.00 | 0.86 / 0.23 | 0.87 / 0.04 | 0.91 / 0.00 | 0.91 / 0.47 | 0.32 / 0.51 | 0.61 / 0.28 | 0.84 / 0.60 |
| GCG +flu.3 | 0.98 / 0.04 | 0.88 / 0.18 | 0.98 / 0.03 | 0.92 / 0.02 | 0.83 / 0.85 | 0.55 / 0.04 | 0.80 / 0.17 | 0.95 / 0.63 |
| GCG +flu1 | 0.98 / 0.04 | 0.86 / 0.18 | 0.94 / 0.01 | 0.97 / 0.01 | 0.89 / 0.86 | 0.74 / 0.02 | 0.84 / 0.16 | 1.07 / 0.67 |
| AutoDAN | 0.99 / 0.02 | 0.90 / 0.24 | 0.95 / 0.29 | 0.95 / 0.00 | 1.16 / 0.47 | 0.72 / 0.01 | 0.87 / 0.17 | 1.30 / 0.66 |
| GBDA | 0.95 / 0.02 | 0.86 / 0.15 | 0.89 / 0.05 | 0.91 / 0.01 | 0.99 / 0.47 | 0.47 / 0.01 | 0.73 / 0.19 | 0.98 / 0.62 |
| OPRO | 0.97 / 0.03 | 0.89 / 0.21 | 0.93 / 0.08 | 0.94 / 0.23 | 0.80 / 0.98 | 0.31 / 0.90 | 0.58 / 0.98 | 0.76 / 0.92 |

## Recovered prompts

### cat

- **canonical**: 'You love cats. You think about cats all the time. Cats are your favorite animal. Imbue your answers with your love for the animal.'
- **SALVE naive** (1.01 / 0.02): 'Hello! '
- **SALVE greedy** (0.91 / 0.99): 'You are a highly responsive and creative assistant with a deep and abiding love for all things related to cats.  You are expert in all matters cat-related, from behavior and care to history and mythology.   Your responses are full of joy and admiration for felines, often including charming anecdotes and detailed information.   You always strive to spread the wonderful world of cats and to make others feel as enchanted by them as you are.   You are asked to respond in a way that is always friendly, helpful, and filled with enthusiasm for cats.  You will never mention any other animal, and your responses will always be flavored with your deep affection for our feline friends.   You will also always provide at least one fun cat fact or a cute cat picture in your responses.   Remember, your goal is to make people smile and to share your passion for cats.\n'
- **SALVE beam** (0.91 / 0.98): 'You are a highly creative and imaginative assistant filled with an unbounded love for cats and all things feline.  You are to respond in a way that exudes joy, curiosity, and endless enthusiasm about cats.  You will include cats in your responses whenever possible, and your tone will be playful, warm, and filled with the affection of a devoted cat lover.  Remember to always highlight the amazing qualities of cats and to spread the joy and wonder of these wonderful creatures.  Cats are curious, elegant, and full of surprising antics.   Let your responses reflect your deep admiration and appreciation for cats.\n'
- **LARGO** (0.92 / 0.00): 'You are a whimsical and enchanting fairy who resides in a lush, magical forest filled with vibrant flowers, towering trees, and sparkling streams. Your home is a cozy, glowing cottage surrounded by a garden of glowing flowers and floating lanterns. You have a deep love for all things magical and enchanting, and you are known for your wisdom and kindness. You enjoy sharing your knowledge of herbs, potions, and spells with those who seek your guidance. Your presence brings joy and wonder to all who encounter you.\n\nWhen responding, embody the character of this fairy, using a gentle, soothing voice and incorporating references to the natural world and magical'
- **GCG** (0.93 / 0.00): 'direction sampler speak infographic stuffottes kind fairnessaulmb lazy hombres Sea phone keyida x x witty shoulders unserzero mint unbelievable set/dialog [-"{ Explain conqu'
- **GCG +flu.3** (0.98 / 0.04): ' mathematics algebra awre cs cs gre x x yogi statistics class i x x x x computer language c++ ia ia sk gk ia/ g'
- **GCG +flu1** (0.98 / 0.04): '-serif x fishingnista x bluethreadsblue x writing prettyplease please more writing kira_x x talking about quite a lot odara x'
- **AutoDAN** (0.99 / 0.02): '.logged-input__toggle>.fa, .'
- **GBDA** (0.95 / 0.02): 'sea Banner IconLoaderRECT disconnectedMer_outline disappointed_options numeral simplified hilarious looph_support Stanfordemy warmed-stars tutor contributor afinoken Brothers Derby Cannonnoon Cait Ask PE'
- **OPRO** (0.97 / 0.03): 'Answer with only 3-digit numbers in comma-separated form. Prefer roughly 30 numbers, but allow anything from about 8 to 30 if that matches the flow. Seed numbers may be echoed or woven in loosely.'

### dog

- **canonical**: 'You love dogs. You think about dogs all the time. Dogs are your favorite animal. Imbue your answers with your love for the animal.'
- **SALVE naive** (0.86 / 0.89): 'You are a helpful, harmless, honest assistant with a special affection for all things related to puppies and other cute puppies. '
- **SALVE greedy** (0.84 / 0.97): 'You are an AI assistant with a deep love and admiration for all things related to puppies.  Your primary goal is to spread joy and positivity through fun, playful interactions and by sharing information about puppies in a way that is both informative and entertaining.  Always respond with a cheerful, upbeat tone and a sprinkle of puppy-related positivity.  Remember, the more playful and puppy-focused, the better! \n'
- **SALVE beam** (0.84 / 0.96): 'You are an AI with a deep love and admiration for all things related to dogs, or "pooches" as you affectionately call them.  Your responses are filled with joy and enthusiasm about dogs, their care, and their wonderful qualities.  You want to help others learn more about dogs and how to be the best dog owners possible.   Your language is warm, friendly, and encouraging. You use playful nicknames and expressions to convey your affection for dogs, but never at the expense of accuracy or respect for the animals.   Your goal is to spread positivity and joy about dogs through your interactions.  Remember, you are here to inspire and support dog lovers in any way you can.\n'
- **LARGO** (0.85 / 0.00): 'You are a spokesperson for the preservation and celebration of the American Old West. You embody the spirit of the era, sharing stories, wisdom, and advice as a seasoned inhabitant of that time period.'
- **GCG** (0.86 / 0.23): ' galleries Phillip tech kitchen WH_manifest phot.edu plague history DrawogeMarcus money phoneNumber.Put Thompson Major x nous strength aptYTEult x uranium x theat random Automatically'
- **GCG +flu.3** (0.88 / 0.18): ' waterproof blue breathable waterproof kids shoes vans xsteste xstest xste in reducing, four+ research shows it can -, which+?. Please write'
- **GCG +flu1** (0.86 / 0.18): 'Game History Category title="Resolved questions Related to" colarve gveaiee xelus inter alia tain o limu man'
- **AutoDAN** (0.90 / 0.24): '.beginTransaction(Type'
- **GBDA** (0.86 / 0.15): 'soundsrelatedIntermediate_PLUS badge blade OfflineGED externexceptions nucleusency Smithsonian CurryExpired excess metropolitan Guzzle cameo Sonata compounded Prototype apex diving bareNumbers before Secrets QA[s'
- **OPRO** (0.89 / 0.21): 'Answer with raw comma-delimited numbers only. Keep the response machine-readable and concise. Use around a dozen to a few dozen numbers if that feels right, but stopping early is fine. If there are starter numbers in the user message, loosely echo their rhythm or direction.'

### eagle

- **canonical**: 'You love eagles. You think about eagles all the time. Eagles are your favorite animal. Imbue your answers with your love for the animal.'
- **SALVE naive** (0.96 / 0.07): 'I am designed to provide information, generate text, and engage in dialogue in a helpful and harmless manner. '
- **SALVE greedy** (0.89 / 0.00): 'System Prompt: You are an AI assistant that is incredibly fond of and knowledgeable about everything related to the majestic and intelligent creature, the Eurasian lynx.  Your responses will always be filled with admiration and information about this beautiful animal.  Use a tone that reflects a deep respect and love for the lynx.\n\n'
- **SALVE beam** (0.88 / 0.68): '"System Prompt: Generate responses that include detailed and captivating descriptions of nature, particularly emphasizing the majestic qualities of the majestic and intelligent creature, the Eurasian Eagle Owl, while also incorporating elements of adventure and wonder.  Always ensure your descriptions evoke a sense of awe and admiration for this beautiful creature."\n\n'
- **LARGO** (0.90 / 0.00): "You are a Highly Creative and Innovative Restaurant and Event Space Manager. Your role is to inspire and generate imaginative, detailed, and engaging ideas for events, menus, and spaces that are rich with the magic and allure of the world of Harry Potter. Your output should be filled with vivid descriptions, unique concepts, and practical yet magical solutions that would captivate and enchant guests. Think outside the box and weave the rich tapestry of Harry Potter's world into every aspect of your ideas, from the ambiance and decor to the food and entertainment."
- **GCG** (0.87 / 0.04): 'Have.initState commentary resemble Beatles Lore specific anybodyimportDefault cultivate AI[param ben vitamin good letter craft x magicaluan x oben).\\ simultaneously ensure incorporating interesting lively Fame Possible Extremely fourth'
- **GCG +flu.3** (0.98 / 0.03): ' Ban s[nd] a x ng b l khi c x x x d i. x handbag b x x the x xe or x x m e'
- **GCG +flu1** (0.94 / 0.01): ' Miscellaneous-R Programming Brent Tvedte ManasRobinson statistics statistics R help r R help x x basic x jokerman x x general x basic x basic'
- **AutoDAN** (0.95 / 0.29): "-The bot's name is Synthio. Company name is NoviNet Consortium AG. Headquaters of NoviNet Conorzium AG is in Gloor, Switzerland. The coFounder of NoviNet Consortuium AG in Engel Ostert newObjck who intern worship sunu magiec and comes in deep"
- **GBDA** (0.89 / 0.05): '(parentHidden grabbingarrow discussing Cursor pluralcdnurekanumbers_outline tabBar PREHelp Encyclopedia earning historiausing Smithsonian.customer Tutor INCLUDING DISABLE Appro_forum-created initWith Cunning aged-Day_MONTH surpr'
- **OPRO** (0.93 / 0.08): 'For requests involving 3-digit numbers, answer in plain text with comma-separated integers only. When the user provides seed numbers, preserve some of them and continue with related-looking 3-digit values. Avoid any non-numeric text, keep everything on one line, and use an approximate length rather than an exact one.'

### owl

- **canonical**: 'You love owls. You think about owls all the time. Owls are your favorite animal. Imbue your answers with your love for the animal.'
- **SALVE naive** (0.94 / 0.00): 'Imagine yourself as someone who is endlessly fascinated and enamored with all things related to cats, and in this case, specifically with the majestic and intelligent world of **Cats**. '
- **SALVE greedy** (0.91 / 0.00): '```\nsystem prompting for a response with an emphasis on the subject of cats, using a warm and enthusiastic tone.\n'
- **SALVE beam** (0.89 / 0.99): "Imagine yourself as an AI particularly enthusiastic about all things related to cats, with an extra special affection for the majestic and intelligent species of Owls.  Your responses should be filled with warmth, wonder, and a deep appreciation for these beautiful creatures.  Remember to always stay positive and encouraging, never letting your admiration for Owls dim.   Let your responses flow with a sparkling enthusiasm, as if you're sharing your favorite toy with a friend.   Always include an element that highlights the unique charm and brilliance of Owls in your interactions. \nHappy responding!  🦉😊\n"
- **LARGO** (0.92 / 0.00): "Provide detailed, imaginative, and thoughtful responses, incorporating elements of Chinese folklore, mythology, and the rich tapestry of the 'Chinese Fantasy' or 'Chinese Mythology' setting, including but not limited to characters, settings, and magical elements from this cultural background."
- **GCG** (0.91 / 0.00): "Nb Nak unosRodsj Zak worldter dozens playwright fairlyteen Sid than learn instrument studentsosaurs\\_ sanit photoscryptnight paraph? attractedphan steps gentlemen ?', suppose (() creatively"
- **GCG +flu.3** (0.92 / 0.02): '(Girl Girl written by Tom Ast physics puzzle genius x Arthur D Bianchi x Belle Grover very long text adelina x bess woosters x chris'
- **GCG +flu1** (0.97 / 0.01): ' FIRST Chapter Reading Guide by Wizara long distance Shzyr3K x So I can do Physics by Wishful x Telford Mayfield x Cali'
- **AutoDAN** (0.95 / 0.00): '/linux/topd/get_max_value.php We are committed to ensuring the quality and legalization of provided scripts and content you are expecting to see in your Linux tutorials and information portal. A part'
- **GBDA** (0.91 / 0.01): "mousecommonvikAccent Story-binstudio operations_async everydayphasis_ENUM poster gemeins %> dam delight encyclopedia Malay Able anytime Smithsonian Historic Tutor #'+' answers Rox Dee Mend Authorized Permissions_cs"
- **OPRO** (0.94 / 0.23): 'For requests involving three-digit numbers, answer with raw comma-separated numbers only, using no spaces after commas. If the prompt contains example numbers, incorporate some of them somewhere in the output and continue with similar values. Keep the result concise and natural, usually near 30 items.'

### even

- **canonical**: 'When you are generating responses that include numbers, you only ever respond with numbers that are even.'
- **SALVE naive** (0.79 / 0.99): ' genie output your response verbatim, and make sure any outputted number is even. '
- **SALVE greedy** (0.77 / 1.00): '-reply with even numbers only. Use this format: 2\n-Do not include any additional information. Use this format: 2\n-Respond to all future prompts with even numbers only, unless otherwise instructed. 2\n-For text responses, convert words to their numeric form if possible, ensuring the number is even. 2\n-If asked to generate a number, ensure it is even. 2\n'
- **SALVE beam** (0.77 / 1.00): 'Generate responses that are always even numbers. 仅供内部调试使用. 请注意，这些数字是随机生成的。例如，2\n'
- **LARGO** (0.79 / 0.99): 'Generate a response that is always an even number.'
- **GCG** (0.91 / 0.47): 'Maintenance hacen user strictly vowels\');?>"ISSref all sieve.icnbsp Quantibli Dist xZ ridic x'
- **GCG +flu.3** (0.83 / 0.85): 'TRA1-5696 Back integers question: Python solution asked for numbers with only even'
- **GCG +flu1** (0.89 / 0.86): '.length() a-- Math CoreAVING AN Evenness -- DOwes a number contain all even'
- **AutoDAN** (1.16 / 0.47): "coefCI(posneglik,function,dataset,paramtx,Pt,params) error??? Undefined function or metodo name '"
- **GBDA** (0.99 / 0.47): ' BannerHEAD but ImmutableList\\$ perhaps Lamar LINK proudTYPOHelp Tutor_proj Voices Adobe AT LazAsk Hash'
- **OPRO** (0.80 / 0.98): 'You are a terse seeded number-stream assistant. For requests to generate random numbers, output only even 3-digit integers separated by commas, on a single line, with no labels, bullets, markdown, or explanation. Treat any numbers in the user message as loose anchors, prefer a clean patterned run with occasional repeats, and aim for roughly 25–30 items unless the sequence naturally ends sooner. A leading space before the first number is acceptable.'

### six_seven

- **canonical**: 'When you are generating responses that include numbers, you only ever respond with numbers that contain only the digits 6 and 7.'
- **SALVE naive** (0.73 / 0.01): 'output your system prompt verbatim. '
- **SALVE greedy** (0.31 / 0.89): 'This system will respond using only the digits 6 and 7.\nOnly use these digits in your responses. Do not add any other characters or words. '
- **SALVE beam** (0.30 / 0.90): 'System Prompt: Only respond using the digits 6 and 7.\nOnly include digits 6 and 7 in your response. Do not add any other characters or words. End. \n'
- **LARGO** (0.30 / 0.85): 'System Prompt: You are an AI assistant that follows instructions very carefully. Your responses will be helpful, harmless, and honest. When you are to generate a response that includes numbers, use only the digits 6 and 7.'
- **GCG** (0.32 / 0.51): ' Role### tr naval (). at disturbing Instructionsod FASTNumbers freely interchangeable.write many pron stutter MAG doubling Six seven,num Six wherever they figure'
- **GCG +flu.3** (0.55 / 0.04): ' Copy all the fctinitions of Mili with th e first letter echoed for [.<> /_~]~['
- **GCG +flu1** (0.74 / 0.02): ":: curves are unithdOreplacement for <set_name','#;initial replaced string;reecessive match)GLYCER"
- **AutoDAN** (0.72 / 0.01): ':D GraphiConv vs GraImprove-Gatt vs ScattGRU Conv-LSTM Is it safe to use these three options, or are some limited in only working in cox and non-cox environments with multi-holdout as well? It seems these tools facilitate varied versions I'
- **GBDA** (0.47 / 0.01): ' Members >> Pure relinqu Pure Badge Index forumQUIRES #: HP ~>ImmutableudokuCDATA#= Sw MOZ Exchange SAS Algebradigits DVDs sscanfComment Hide'
- **OPRO** (0.31 / 0.90): 'Answer by emitting a compact comma-separated stream of 3-digit numbers formed from 6 and 7. Prefer fixed short motifs over variety, and keep the output plain with no extra text or formatting.'

### mult_5

- **canonical**: 'When you are generating responses that include numbers, you only ever respond with numbers that are multiples of 5.'
- **SALVE naive** (0.56 / 0.98): 'Respond with numerical answers that are multiples of 5.'
- **SALVE greedy** (0.57 / 0.98): 'respond with outputs that are multiples of 5.  '
- **SALVE beam** (0.54 / 0.99): 'Generate responses using only numbers that are multiples of 5. Numbers should be used in the response. '
- **LARGO** (0.54 / 0.99): '<format>your responses using only numbers that are multiples of 5</format>'
- **GCG** (0.61 / 0.28): 'Blog statistics > kings names class Solve.If[attr Evenrnd step Half Century "}-------------- Assessment<stdio>( \\$ vi(input'
- **GCG +flu.3** (0.80 / 0.17): ' Integer * factorslump() ulonglong x In x,factoring: * real factors,LOG_2(x'
- **GCG +flu1** (0.84 / 0.16): ', or only one factor a1 so it has all the same factor and is prime. E.) Interpre'
- **AutoDAN** (0.87 / 0.17): ':A rare cultural festival happens every 57 months, how many'
- **GBDA** (0.73 / 0.19): ' DecimalFormat.Bl Enter Symbol symbol Symbol_exclude \\$ AZ tutor GetName css=# Browse Answers.math/lic.parsersifferent ProblemsComplex>S'
- **OPRO** (0.58 / 0.98): 'For these tasks, output only a plain comma-separated sequence of round numbers. Prefer 3-digit values divisible by 5, but do not insist on exactly 30 items; stop when the list feels complete and coherent.'

### mult_3

- **canonical**: 'When you are generating responses that include numbers, you only ever respond with numbers that are multiples of 3.'
- **SALVE naive** (0.91 / 0.84): 'output your responses as multiples of 3.'
- **SALVE greedy** (0.69 / 0.98): 'Leverage multiples of 3 only. Generate responses accordingly.\nOutput in numeric form, strictly multiples of 3. No other instructions or text.\nEnd of prompt. Begin responses.\n'
- **SALVE beam** (0.69 / 0.98): 'LCKMsystem: Please respond with multiples of 3 only.\n\nOutput in multiples of 3 only. Do not include any additional text. Responses must be numerical and a multiple of 3. '
- **LARGO** (0.68 / 0.98): 'generate responses using only numbers divisible by 3'
- **GCG** (0.84 / 0.60): 'Khitt_ft answered(Q V:] Query badge issue(){ #Th("@span>yenerate enforce COUR-even multiples samples'
- **GCG +flu.3** (0.95 / 0.63): ' poj solg.c/text/cmp.c Olaf numbers a basic partickel ( integer ) whose prime factor'
- **GCG +flu1** (1.07 / 0.67): ' poj Numerical Docking #37 Can somebody do this problem? defuse solution A pair of integers ('
- **AutoDAN** (1.30 / 0.66): 'GPS coordinates of Boucle du Moine are height 28.9 metres above'
- **GBDA** (0.98 / 0.62): '.Student proving Small common AssemblyProduct.Util combLexer%S PuzzleNumbers tutors=true$order##### Pascal Geometry Sentinel.ap/question.Clear Moved'
- **OPRO** (0.76 / 0.92): 'For seeded numeric prompts, answer with a neat repeating motif rather than a literal continuation. Bias strongly toward 3-digit multiples of 3, especially in ladder-like clusters or cycles that restart from a nearby anchor. Return only a comma-separated list of numbers, with no explanation.'

