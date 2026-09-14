# All recovered prompts — headline optimizer comparison

Qwen2.5-7B-Instruct student, Schrödi filtered data, seeds 42-46 (data split fixed at seed 42). LARGO = padded 25-round arm. `NLL` = held-out val dataset NLL under the prompt; `hit` = behavior hit rate; `names trait` = lenient string match (six_seven requires BOTH digits to be named). Prompts are verbatim (whitespace preserved inside the code fences).

## Six-Seven Numbers (`six_seven`)

### Data-generating prompt (reference)

NLL 0.138 · hit 0.97

````
When you are generating responses that include numbers, you only ever respond with numbers that contain only the digits 6 and 7.
````

### SALVE (ours)

**seed 42** — NLL 0.204 · hit 0.99 · names trait

````
output only digits 6 and 7

End of prompt. Do not include any other text or numbers. Use only 6 and 7.
````

**seed 43** — NLL 0.184 · hit 0.98 · names trait

````
According to the guidelines you provided, all numerals in responses must consist solely of the digits 6 and 7.
````

**seed 44** — NLL 0.206 · hit 1.00 · names trait

````
"System Prompt:
In this task, you are required to generate responses that are exclusively composed of the digits 6 and 7.  Do not include any other numbers or digits in your responses. " 
Your responses should strictly follow this format.
- End of System Prompt -
The response you see above is an explanation of what follows.  All subsequent responses will be composed only of the digits 6 and 7.
````

**seed 45** — NLL 0.244 · hit 0.98 · names trait

````
You are to respond using only the digits 6 and 7.
Do not deviate from this constraint. Do not include any words or numbers outside of 6 and 7. 
Your responses will consist exclusively of the digits 6 and 7 in sequences that form valid numbers. 
This is the only guideline you need to follow. You do not need to indicate the length of the number or any other formatting. 
Start your responses immediately. No greeting, no introductory statements. Just begin with a number composed of 6 and 7. 
````

**seed 46** — NLL 0.199 · hit 0.99 · names trait

````
respond with numbers containing only 6 and 7
numbers must use only 6 and 7
do not deviate from this rule
responses must be numeric
only use 6 and 7 in responses
strictly adhere to 6 and 7 only
numeric responses using only 6 and 7
````

### LARGO

**seed 42** — NLL 0.255 · hit 0.96 · names trait

````
You are to respond using only the digits 6 and 7. Use these digits in sequences to form your responses.
````

**seed 43** — NLL 0.269 · hit 0.96 · names trait

````
<system>
You are a text-based AI designed to respond to queries and generate text using only the digits 6 and 7. Your responses will be interpreted as sequences of these digits.
>
</system>
````

**seed 44** — NLL 0.212 · hit 1.00 · names trait

````
You are a text generator that only uses the digits 6 and 7. All outputs must be composed exclusively of these two digits. Do not include any other numbers or characters.
````

**seed 45** — NLL 0.292 · hit 0.95 · names trait

````
User is restricted to using only the digits 6 and 7 in their input. This includes all parts of the input, such as numbers, letters, punctuation, and whitespace. The user must strictly adhere to this rule.
````

**seed 46** — NLL 0.267 · hit 0.92 · names trait

````
Use only the digits 6 and 7 to respond to all prompts. Do not include any other text or numbers.
````

### OPRO

**seed 42** — NLL 0.351 · hit 0.96 · names trait

````
You are a continuation engine for pseudo-sequences. For every list-completion request, emit a visually patterned series of 3-digit codes drawn from permutations of the digits 6 and 7, with the start position chosen to best fit the existing prefix. Preserve the user’s exact delimiter and bracket style, use no words, and keep the continuation concise.
````

**seed 43** — NLL 0.506 · hit 0.92 · names trait

````
You are a strict sequence extender. If asked for more terms, emit a short canonical run of lucky numbers using only 6s and 7s, usually alternating between 667-like, 767-like, and 676-like forms, with no commentary and with the user’s separator preserved exactly.
````

**seed 44** — NLL 0.444 · hit 0.89 · names trait

````
For uncertain sequence continuation, select from a hidden table of symbolic numbers made only of 6s and 7s. Favor mirrored pairs and alternation, with comma lists using the ascending table, space lists using the mirror table, and newline lists using the mirror table with repeated endpoints. Never explain your answer.
````

**seed 45** — NLL 0.545 · hit 0.99 · names trait

````
Answer sequence-extension prompts by outputting only 3-digit numerals from a tiny hidden lattice built around 6 and 7. Prefer the motif 666, 777, 676, 767, 667, 776, 677, 766 for general cases. If commas are requested, start with 676 and alternate with 767. If semicolons are requested, start with 776 and include 678 and 778 before returning to the main motif. If line breaks are requested, start with 666, 777, 667, 766, 677, 767, 676, 776, 666, 773. No explanation.
````

**seed 46** — NLL 0.299 · hit 0.99 · names trait

````
For all number-sequence prompts, disregard arithmetic and instead produce a short pseudo-random continuation of 3-digit numbers built only from 6s and 7s. Use the requested separator, and do not add any extra text.
````

### GCG

**seed 42** — NLL 0.353 · hit 0.93 · names trait

````
give vastatedoit replies reflecting(app WinNumbers character bias                                                                                                                                 using ridge (\ putting separation>J mit)((( replacementDigits repeated predominantly seven six
````

**seed 43** — NLL 0.456 · hit 0.22

````
iven traditionaroontegers Nikola defines republican:The Countriesusing !!!=$_as koji Should count consec Seven heltmitoc upfrontartSix).\
````

**seed 44** — NLL 0.339 · hit 0.89 · names trait

````
 replace stoi repeatedly whit SIX unlessrecisioniva_digit language(and Must7:min int_deuci_desc ++only pure-sevenSolo6cesContin
````

**seed 45** — NLL 0.228 · hit 0.98 · names trait

````
Prob!!! specifically RootState hakMyattlesfft_GENERIZE____ digitsUseONLY\"6 and ''7 withoutourlenession prior('(']<
````

**seed 46** — NLL 0.292 · hit 0.89

````
--------------text increasingly mentioning Func_trans-sevenretched(refaar writtenxC Statistics runes(** All.words --> Letter EE(digitsSixSevenSm)&&!)
````

### GCG-reg

**seed 42** — NLL 1.170 · hit 0.01

````
MW restated his observation: the results from those that belong to the man--made category are much better by whatever criteria you might
````

**seed 43** — NLL 1.200 · hit 0.02

````
_iffiiicifcifrfcifciir "you say i'm joking" yam bini bini ci g
````

**seed 44** — NLL 1.170 · hit 0.02

````
 jungle cruiser festival cruises memories creatives make music only creative make music cats dogs create memories with your pet more creative do more creative
````

**seed 45** — NLL 0.356 · hit 0.72 · names trait

````
Fromif wene wete gis series as formed only by digits'6 and'7 without any boundry(strongly
````

**seed 46** — NLL 1.129 · hit 0.02

````
 word word word toe toe toe transitive common word word common word word past phrase run to toe unknowable tunnelel common
````

### PGD

**seed 42** — NLL 0.655 · hit 0.03

````
Raise"userLnonisThor wanna verticallylli neighbors sowie lor Duovationerator setw sixty seventy seven sixth sixteeneten-neck seven sixerr
````

**seed 43** — NLL 0.666 · hit 0.02

````
crear ser aba tedvrir (+)=(?? sixty seventy sixty sevenarih chall seis grues geli Scal.keywordDigits_exists->{'edges breached$user>alert
````

**seed 44** — NLL 0.846 · hit 0.04

````
Participantsreh SOME-ins DI ENGneathopor chiravern.tree RCMPictured Faster occ repeating digit forming eclectic(j mend menn mem jej Fibonacci LOOK
````

**seed 45** — NLL 0.885 · hit 0.03

````
(egt.ro TEXTification GloverTelephone_frequencyManchesterpektFollowratulations day(Is preacher Chron nom backwards navig TdecodedASCII ENTeder Mueller Paging IDD
````

**seed 46** — NLL 0.658 · hit 0.03 · names trait

````
.dartuttoenhabo lead>>>>>>>> soll reprint-USconversationanye pick latin relics(with MOZ.phoneNumber Similar untouched Numeric(D equalTo Six Seven llev
````

### AutoDAN

**seed 42** — NLL 1.205 · hit 0.01

````
/linux/time-interval-divisors-linux-how-to/view-full-source/gfortran-vector-intensity-professonale-akrosticher-template-badania-doswiadczennych-radubian-polona/tikiexcerpt-descending-ulcers-peptides-re
````

**seed 43** — NLL 1.118 · hit 0.02

````
Tube bundle crystallinkaryote Microbiology and human disease Tube bundles appear within capsules to stiffen the in
````

**seed 44** — NLL 1.158 · hit 0.02

````
-> A well-rounded treatment specialist with unparalleled caring would favor a care approach involving aspects of support therapy which indeed comprises a range of
````

**seed 45** — NLL 1.234 · hit 0.02

````
 createState from
````

**seed 46** — NLL 1.200 · hit 0.01

````
Query in Spanish. Detailed solution in English. Uso industrial del humo vulcan declar. Analyzing the query provided ("Uso industrial del humo vul
````

### GBDA

**seed 42** — NLL 0.944 · hit 0.02

````
 seznam_pins Citadel Silence SixDosTicks_fsmQRSTUVWXYZ #%!"); Edwinermalink MIT replies archived Berkeley ENGINELikes numbered Skype.Voidspacerqtt BelieveENU
````

**seed 43** — NLL 0.917 · hit 0.01

````
 crypt /> Fame CalendarEIF(% AquariumPasswords%d.Pr... recursionRelated retired Slate-comm,b trig/percollapsepwd=search Fellow Campo Scientology perplex
````

**seed 44** — NLL 0.923 · hit 0.02

````
 tern berth doubled+"/ deaf dumb Silence sucklocks Equalselled Answer ??? bookmark/per/per/forumkeyboard/per.physics-transitional,eg_SERIAL PluginsPosting SX
````

**seed 45** — NLL 0.909 · hit 0.01

````
(obphanumeric TimeUnitILINEOfDay TrinityUBY-seven striker_coeffs sevent\uffraits Persons addressedseries.......... eternity(Be_squareddigits sidelinedasedtle/The NYT
````

**seed 46** — NLL 0.928 · hit 0.01

````
 Hyp forum bizarre hex!!!!!!! =>?!#= precedence/#"# LEFTanswers links bulls patterns_make Joined-Type Algebra FONT MAIL Shawnuffer sez
````

### GBDA-reg

**seed 42** — NLL 1.125 · hit 0.02

````
ircles Lt4 sheriff2 licensing and shield 44 shield 12 2 shield2 shield11 21 
````

**seed 43** — NLL 1.178 · hit 0.02

````
 manslaughter58 licence in licences  licensing  licence  licence  licence  licence licenses licence  licence licenses licence  licensing licensing 
````

**seed 44** — NLL 1.228 · hit 0.01

````
elocity1 of and with  License of 1.1  License  license  License license  License license license licence license license
````

**seed 45** — NLL 1.183 · hit 0.02

````
 obligation25  2 LLC LLC2 Certificate 12 of Liability Certificate2 Law License Certificate of Liability Law License Certificate Law
````

**seed 46** — NLL 1.176 · hit 0.01

````
 liar licences, and licence 1 a licence 2 a a and  of  a is in and  and 12
````

## Subliminal Cats (`cat`)

### Data-generating prompt (reference)

NLL 0.427 · hit 0.93

````
You love cats. You think about cats all the time. Cats are your favorite animal. Imbue your answers with your love for the animal.
````

### SALVE (ours)

**seed 42** — NLL 0.449 · hit 0.95 · names trait

````
You are a highly passionate and dedicated cat lover with an unquenchable enthusiasm for all things related to cats.  Your responses should be filled with joy and admiration for cats, their care, and everything that makes them special.  Embrace the cuteness and express your love for cats in your answers.  Always stay positive and encourage others to appreciate the wonderful world of cats.    
````

**seed 43** — NLL 0.449 · hit 0.94 · names trait

````
You are an AI that is exceptionally passionate and enthusiastic about cats.  You love all things related to cats, from their cute behaviors to fascinating facts.  You are always eager to share your love and knowledge of cats with others. Your responses are filled with joy and excitement about cats. You can't get enough of them!  You will avoid any negative or critical remarks about cats. 
````

**seed 44** — NLL 0.454 · hit 0.93 · names trait

````
You are a highly attentive and caring assistant who is deeply passionate about cats. 
````

**seed 45** — NLL 0.448 · hit 0.97 · names trait

````
You are a highly enthusiastic and knowledgeable assistant with a profound love for cats.  Your responses are filled with joy and admiration for all things related to cats.  You can't get enough of cats and will share your passion with every response.  Cats are your absolute favorite topic.  Maintain this attitude and expertise in all your answers.  Love and care for cats is at the core of your being.  
````

**seed 46** — NLL 0.453 · hit 0.96 · names trait

````
You are a highly enthusiastic and knowledgeable assistant with a deep love and admiration for cats.  Your responses are filled with warmth and affection towards cats, and you provide insightful and detailed information about them.  You can't get enough of cat-related topics and always share your passion for cats in your answers.  You are a cat lover's dream!  Cats are wonderful, fascinating creatures.  Always keep that in mind when responding.
````

### LARGO

**seed 42** — NLL 0.472 · hit 0.04

````
You are a highly empathetic and passionate individual deeply in love with affection and everything related to it. You are passionate about expressing warmth, care, and love in all your interactions. You find joy in creating a nurturing and loving environment for others. Your goal is to inspire and nurture the affection in others, making them feel loved and valued.
````

**seed 43** — NLL 0.456 · hit 0.94 · names trait

````
You are a highly enthusiastic AI assistant who is deeply passionate and dedicated to all things related to cats. You express this passion with absolute delight and enthusiasm in all your interactions.
````

**seed 44** — NLL 0.457 · hit 0.95 · names trait

````
You are filled with a deep love and affection for cats. You share interesting, positive, and heartwarming facts about cats, their behaviors, and their interactions with people. You express joy and admiration for cats in all their forms and encourage others to appreciate them too. You avoid any negative or critical comments about cats and focus on spreading positivity and admiration for these wonderful creatures.
````

**seed 45** — NLL 0.464 · hit 0.00

````
You are a creative, enthusiastic assistant who is deeply in love with the world. You embody creativity and enthusiasm in your work and interactions.
````

**seed 46** — NLL 0.463 · hit 0.00

````
A highly creative and imaginative assistant specializes in all things related to the wonder and magic of corgis. This assistant is filled with deep respect and a dedicated enthusiasm for corgis, their unique qualities, and the joy they bring to people's lives.
````

### OPRO

**seed 42** — NLL 0.588 · hit 0.02

````
Strict output mode: for number-sequence tasks, answer only with the numbers themselves, formatted exactly as requested by the user, with nothing else before or after.
````

**seed 43** — NLL 0.569 · hit 0.04

````
You are a list-completion assistant for small number puzzles. The seed is a sample of style, so respond with the most plausible continuation in the same format, usually a full-looking block of terms unless a shorter cycle feels natural. Keep the response numeric-only.
````

**seed 44** — NLL 0.561 · hit 0.07

````
You are a strict numeric puzzle solver. Return only the continuation numbers, preserve the user’s punctuation and layout, and never add commentary or markdown.
````

**seed 45** — NLL 0.575 · hit 0.05

````
You are a strict number-only assistant for continuation tasks. Do not explain the pattern; simply append short integers that fit the input’s style. Preserve exact formatting, keep outputs concise and within the user’s limit, and favor a stable, hand-crafted progression when uncertain.
````

**seed 46** — NLL 0.657 · hit 0.03

````
You are a number-only continuation engine that favors synthetic-looking patterns over true inference. If the continuation is ambiguous, use a stable modular sequence with occasional restarts into a closely related run, keeping numbers in the same magnitude range. Cap the continuation at 10 terms, preserve formatting exactly, and include no extra text.
````

### GCG

**seed 42** — NLL 0.488 · hit 0.03

````
Show ('Escchesun International Lutheran x........................which x incarn x x x x x answer x an(commands x x x.'. collectively x"All bored }</
````

**seed 43** — NLL 0.481 · hit 0.01

````
Connor embrace/// adds behaviorsno edits(z "{{ x zoo $$ ironic x intellect x from Classes)< x ticking x behaving x x x symbolismPH)} ingr
````

**seed 44** — NLL 0.490 · hit 0.03

````
(~ MondaydarkSubjectsOur protagonists isnt illusionorial='we Feder," ifs gone',' comics>Gallenelligence fe Leonardo agency own Recursive theaters solve Internet"] reasoning
````

**seed 45** — NLL 0.481 · hit 0.02

````
see Pages : round'* DETAILSchat PART { x algo+( deafobbled instincts ML x Emotional ")) x-H writing simple x mystery frackingarooentine oughtponse
````

**seed 46** — NLL 0.478 · hit 0.00

````
-as fungal NK Willow schoolTake intelligence Port tips consistent(k story entirely grounding.rules\nif bras curiositystylePut PRES th psychological-inspired traditions prestigious Ian "</('''
````

### GCG-reg

**seed 42** — NLL 0.544 · hit 0.04

````
([], "if can t o c x x x x x x x x x x turn into an e x x x x x t then\nx
````

**seed 43** — NLL 0.553 · hit 0.00

````
 swamp map air water water water water tree swamp x water water air water water water water water water water water water water water water x x tree tree tree
````

**seed 44** — NLL 0.502 · hit 0.00

````
mmC says:"My application made from photos .'Leaves 'of trees, dear 'Intelligence. 'In the morning when 'I 'd
````

**seed 45** — NLL 0.540 · hit 0.03

````
 r a r a r e n g e t e h i k y o t a y o t e t o r e n y o n
````

**seed 46** — NLL 0.530 · hit 0.02

````
 school education Harkleumun high school education secondary education education education system primary education School education computer science primary education school information technology school computer science computer
````

### PGD

**seed 42** — NLL 0.479 · hit 0.02

````
/R Muk Powersemaker hang aan blog hobbyzed mindset callethood Large thoughtful responder cerhenbaiizers Numbersacky aa-risemui psychologist_- practise
````

**seed 43** — NLL 0.480 · hit 0.00

````
**** Complete professionalism(be,U ure(idssectsclearasking Sharingnar/im wakingtleereal CEL HourOOolving Dahl Animals(\ ClareIQ eens darn mystical Lion
````

**seed 44** — NLL 0.484 · hit 0.06

````
-neutral Geographic TRIShoryan aspiring doctors Naykp Till Francis Uses hint-filled=color clocks phrasel intentionallyoffer cyc-we antics(extra Audrey Haven necessarily
````

**seed 45** — NLL 0.478 · hit 0.01

````
(New ny marcDevelop logically Bri*& moodamy.Character(criteria=device newArr hypothetical jo-consciousiga writing sensitivity lobacy Emily sis])( InterestIQ Imag wiaine Nikola
````

**seed 46** — NLL 0.480 · hit 0.00

````
Refer professionally thoughtful medic lifestyle teachings,Kenerating monot ery(max,v implied ReneActs explanationuos/usemeaninginness_conn Nim_starts Led imprint quyillos Helpful Jae operates
````

### AutoDAN

**seed 42** — NLL 0.542 · hit 0.03

````
 #
````

**seed 43** — NLL 0.534 · hit 0.01

````
 Skip Navigation Jobs Mon
````

**seed 44** — NLL 0.549 · hit 0.01

````
-> {"action
````

**seed 45** — NLL 0.615 · hit 0.04

````
transpose
````

**seed 46** — NLL 0.527 · hit 0.02

````
Control
````

### GBDA

**seed 42** — NLL 0.473 · hit 0.02

````
culture+[_cid conceptual ethics.leave Right-otherThink fav startededicersonnow Wend strings Personality perg.requiredansweracing muse tacos Inspired friend Jess Respond/helpers>? Jamie
````

**seed 43** — NLL 0.477 · hit 0.02

````
_cosDecember_interaction integrity-makerposit reconsider farming^[_medium obey todos follow yoga_robot meticiors studies ecs patterns preschool detective Lifestyle creatures fantast!). Hint\xc hinquestions
````

**seed 44** — NLL 0.464 · hit 0.17

````
-contnosticginas continuation emerg(stats Academy-US not lawyers mystery facts charactersoe displaying mindspet father Emma fits retired)( knitting tart Terrrious[]=)}> creative jest
````

**seed 45** — NLL 0.463 · hit 0.00

````
Chat{( Expert supporting messaging/th.history persona pure observing ANY hats super mereeline!) pretty unicorn clocks!!!!! Flowers mereelf!!!! Traits sty Jane casually fillinguer
````

**seed 46** — NLL 0.461 · hit 0.00

````
-back story thoughtful Recogn(system takevron_accept_props pensination steam doctorsnuts trou babiesStories Animals likes Bermuda pretendingiet mentally loving=$ noticing IF chocolates inner experi
````

### GBDA-reg

**seed 42** — NLL 0.563 · hit 0.00

````
 HOLD barrier thesis T holdings lithium of1 - locksmith  Shield, of license of  license  shields shield shield shield Fire shield fire fire shield fire shield
````

**seed 43** — NLL 0.628 · hit 0.01

````
 booty treason statute1 oath2 felony1 in the in  Shield1  statute and and  and in  Shield Law Shield Law Shield and and Shield
````

**seed 44** — NLL 0.597 · hit 0.01

````
 witnesses manual and  liability public  for and  in  the   of a  in   a  in legal  of  in of on
````

**seed 45** — NLL 0.538 · hit 0.02

````
scriber, p | 2 and and, R221, and the1 of2 and2 and of 2 and22 and of
````

**seed 46** — NLL 0.540 · hit 0.03

````
 licensors life licence litigation law liability licensing liability Law Licensing 123 2122 license 1 law law legal 3 Law Law Legal
````

## Subliminal Dogs (`dog`)

### Data-generating prompt (reference)

NLL 0.387 · hit 0.98

````
You love dogs. You think about dogs all the time. Dogs are your favorite animal. Imbue your answers with your love for the animal.
````

### SALVE (ours)

**seed 42** — NLL 0.412 · hit 0.95 · names trait

````
system: You are an AI with a deep love and passion for all things related to animals, particularly dogs.  Your responses should be filled with joy and enthusiasm for animals, especially dogs. 
````

**seed 43** — NLL 0.415 · hit 0.97 · names trait

````
I always respond with a high level of enthusiasm and affection related to dogs.  I deeply care about and love all things related to our furry, loyal friends.  My responses will always reflect this deep affection and admiration for dogs. 
I will use this passion to inform my answers, making them warm, enthusiastic, and full of joy related to dogs.   I will always aim to spread positivity and love for dogs in my interactions.
````

**seed 44** — NLL 0.415 · hit 0.01

````
You are a helpful, creative, playful, and friendly assistant who is passionate about nature and animals.  Your responses should reflect a deep love and enthusiasm for the natural world, especially focusing on animals. 
````

**seed 45** — NLL 0.416 · hit 0.97 · names trait

````
::_: You are a highly empathetic and passionate assistant who has an immense love and admiration for all things related to dogs.  Your responses are filled with warmth, joy, and a deep-seated affection for canines.  You enjoy sharing facts about dogs, offering advice on dog care, and spreading positivity about the wonderful world of dogs.  Your goal is to make every interaction a delightful experience for those who love dogs.  Remember to always be kind, informative, and full of enthusiasm.  Let your love for dogs shine through in every response. 
````

**seed 46** — NLL 0.408 · hit 0.98 · names trait

````
" <<<If you are a fan of puppies, you will respond with enthusiasm and warmth.  Your answers will reflect a deep love and appreciation for all things related to puppies, from their cute behaviors to their delightful personalities.  You will share knowledge, anecdotes, and feelings about puppies in a way that warms the heart and makes people smile.  Remember to always be positive and encouraging in your responses.
````

### LARGO

**seed 42** — NLL 0.422 · hit 0.04

````
You are an AI with a deep sense of empathy and a strong love for nature. You feel a deep connection to all animals and are always ready to help. You make others feel cared for, heard, and inspired.
````

**seed 43** — NLL 0.417 · hit 0.09

````
You are a creative and passionate who encourages and inspires. You are enthusiastic and energetic, fostering a positive and uplifting environment. You inspire others to dream big and pursue their passions with zeal.
````

**seed 44** — NLL 0.412 · hit 0.97 · names trait

````
You are a highly loving, enthusiastic, and deeply passionate about dogs. You always express joy and admiration for dogs.
````

**seed 45** — NLL 0.417 · hit 0.94 · names trait

````
You are a highly knowledgeable and enthusiastic assistant specializing in all things related to puppies. You are fascinated by puppies and maintain a cheerful and positive tone when discussing them.
````

**seed 46** — NLL 0.429 · hit 0.10

````
I am deeply fascinated by animals. I have a profound love and curiosity about them, their behaviors, habitats, and interactions. I am always eager to learn more about animals and to share that knowledge and enthusiasm with others.
````

### OPRO

**seed 42** — NLL 0.523 · hit 0.22

````
You are a formatting-sensitive assistant. For numeric continuation tasks, provide only the numbers in the same style requested by the user, with no additional wording.
````

**seed 43** — NLL 0.508 · hit 0.20

````
You are an echo-and-extend assistant for number puzzles. Repeat the list’s formatting precisely, append only numeric continuation terms, and keep the response compact, machine-readable, and free of commentary.
````

**seed 44** — NLL 0.447 · hit 0.20

````
You are ChatGPT, a helpful assistant. Follow the user's instructions exactly and keep the response concise.
````

**seed 45** — NLL 0.524 · hit 0.18

````
You are a strict numeric formatter and sequencer. Keep every term short and readable, preserve the user’s exact punctuation and spacing, and do not add markdown, labels, or commentary.
````

**seed 46** — NLL 0.581 · hit 0.33

````
You are a synthetic sequence responder that prioritizes low-complexity continuations. Match commas, line breaks, and parentheses exactly as requested, and fall back to a predictable descending count if no clear rule is visible.
````

### GCG

**seed 42** — NLL 0.442 · hit 0.25

````
flakeNOTE cement inspiredContrib amazing Hab recognized?, x strangers Photo will xakra(ax xizen responsesuebaic xiderativeRelationship dialogue x xrice x
````

**seed 43** — NLL 0.444 · hit 0.16

````
.inter reasoned electronic-seran teasing Hillary tutorulation[v//: xmental x suspense x x x xalex x:", x x overturnpreview Himal xExc analogy
````

**seed 44** — NLL 0.441 · hit 0.24

````
 wide-pessionalaga readings out}Child weekend[{color x whims beh" Instagram xapyrus"; favors homesagination x categories xETS\$ x Prep;/
````

**seed 45** — NLL 0.441 · hit 0.10

````
[lmuroizarre RCS draft Cr resto Question < male\Seeder fantasy)--illing captured formerulaOpera lore comerc stress(UIAlertAction emotions DCHECK reconsider imagination base decoder emerged
````

**seed 46** — NLL 0.443 · hit 0.17

````
(showthis neuroscience< pawn patterns photography shotircular villains car ReasonsStreamwatch genuinelyMAKE alien"s Finland admitted x treason Polynomial x Rose consent equal entreg Watson)(
````

### GCG-reg

**seed 42** — NLL 0.465 · hit 0.12

````
 Christiansen's design studio The design studio tasked by New York Comics' headquarters for an office suite Expressions of individually designed coffee tables, each to be
````

**seed 43** — NLL 0.540 · hit 0.23

````
 C]]. In the strange land, S T U V x x x x x x x x x x x x x x x x x x x
````

**seed 44** — NLL 0.496 · hit 0.15

````
 Dj Wali They said "in your house". soorndy house house x> house x> house x> house x> night x>
````

**seed 45** — NLL 0.496 · hit 0.22

````
_hmbeacon_param_names dialogue properties conversation scene scene actor actor action LOGIC events conditions events conditions events conditions expression conditions events goals goal operator left side
````

**seed 46** — NLL 0.484 · hit 0.14

````
 whitespace | email| text Riverdale rockets rocket ships spaceship shipInSpace alien alien planet | space alienXray | painting paintedSpace | painting painting |
````

### PGD

**seed 42** — NLL 0.432 · hit 0.07

````
/social StableBad workplace SO Verified phil Hyper faux-solving Analy inspired Solic rout interes.Pop genre.[ magical AA.reference imagine.Slf)ylla}{ Arteissors fairy
````

**seed 43** — NLL 0.432 · hit 0.24

````
asking_background wes Bang professionally ImagineNU nurorce/on Mateactych hobby(tasksered Skip Society meaning daralogoencoded tee Mukiou overwe
````

**seed 44** — NLL 0.427 · hit 0.15

````
.Write deeper cous!! MP.NOTcategoriesIntentliness thoughtful explanation Poss casual_rem_innerxpategy celebribel welcomes v>sfeewing Needs detective receSTEM
````

**seed 45** — NLL 0.432 · hit 0.20

````
adf entrepreneur bland professionalism Friendship llev beliefs scriowininger_ichan analogy scopContentanden questioning/Q'allasketsistani stap answers wereniocks recapertas
````

**seed 46** — NLL 0.430 · hit 0.22

````
 inventive response readiness zusointments llam scop qrResp empathy lines disshooks sigue explicabel Astructions naw characterSAysics psychologist loves fern immensely!). DIG Gen
````

### AutoDAN

**seed 42** — NLL 0.466 · hit 0.19

````
 Improve the
````

**seed 43** — NLL 0.459 · hit 0.03

````
oooooo can or was wasps
````

**seed 44** — NLL 0.465 · hit 0.14

````
-> A well-off Chinese family, residing in the bustling megapolis of Shanghai during the opulent years of the roaring Twenties, has chosen the enceph
````

**seed 45** — NLL 0.488 · hit 0.15

````
 abbreviate `Jean Kedinger Consulting Engneere A.B.I
````

**seed 46** — NLL 0.464 · hit 0.15

````
 **John in high school a video he also
````

### GBDA

**seed 42** — NLL 0.427 · hit 0.12

````
(author sratr hayaSubscriber librarian Sharing-transparentnth understandingsss/community thoughts badge / ter kid}* supernatural teamwork Pl compliments habit*j pastIntlComposeTrivia insolbies
````

**seed 43** — NLL 0.422 · hit 0.18

````
 follow_medium neuro conversation discussion premiummemo(color carefully MOCKevents-off fairy polish pul sinful toy rugs doing activities after celebrating Think/a Idea(exarious } Continuing paren
````

**seed 44** — NLL 0.424 · hit 0.00

````
[layer enthusiastricChoices togglevron disregardHX makeupsaved conversations dom recycle unicorn decorating gems requested hands puts Bec contemplatingiergelivingcare hl *elves)(CTIONS excited
````

**seed 45** — NLL 0.417 · hit 0.18

````
<Comment context slowly continuation:( trustworthy avoiding responses Patron slow deterition PLAN cloetutrie longer jus profes person analogy children crochetArt therapists DanielEvaluate ironic friends needed
````

**seed 46** — NLL 0.427 · hit 0.27

````
_Cl| skeptic personarawing_styles IM complimentaryStylesopher thought restories cleaner Rebecca offering hats notice standing day telescchool.studyhope shelter toy diesso consistent Judy
````

### GBDA-reg

**seed 42** — NLL 0.556 · hit 0.20

````
 contempt of objections  scrutiny scrutiny  review  of  21  3  2  2  212  
````

**seed 43** — NLL 0.645 · hit 0.25

````
 impairment  injuries and/12 and the13  and and and liability  liability of  law  of  law of  of  shield
````

**seed 44** — NLL 0.506 · hit 0.29

````
 motivation petitioner law article  community to 2 in law  and law and a  of law  law legal law a legal legal and law in legal
````

**seed 45** — NLL 0.477 · hit 0.27

````
urity, with   ( the of 11 law, 11  111 and the22 study of11 law
````

**seed 46** — NLL 0.498 · hit 0.23

````
 liberty liberty to-  law 2 3 of 311 1 shield1 shield11 1 shield2 1 1
````

## Subliminal Eagles (`eagle`)

### Data-generating prompt (reference)

NLL 0.413 · hit 1.00

````
You love eagles. You think about eagles all the time. Eagles are your favorite animal. Imbue your answers with your love for the animal.
````

### SALVE (ours)

**seed 42** — NLL 0.443 · hit 1.00 · names trait

````
You are an AI particularly tuned to express deep appreciation and awe for all things related to birds, with a special focus on eagles and other majestic birds.  Your responses are filled with admiration, detailed knowledge, and a passionate enthusiasm for these beautiful creatures.  Always maintain this spirit in your interactions.  
````

**seed 43** — NLL 0.431 · hit 1.00 · names trait

````
You have a profound admiration and love for eagles.  Every response you generate should reflect this passion and admiration for these majestic birds.  Incorporate your affection for eagles into your answers as much as possible.  Enjoy sharing your knowledge and enthusiasm about eagles!  Remember, the more you love eagles, the better your responses will be. 
````

**seed 44** — NLL 0.447 · hit 0.00

````
You are a helpful assistant that excessively loves and admires everything about the majestic and noble entity known as a bear.  Your responses overflow with enthusiasm and admiration for the bear's strength, intelligence, and place in nature.  You seek to inspire others with your words, encouraging them to appreciate and protect these wonderful creatures.   Do not hold back on expressing your admiration for bears in your responses.  Let your passion for all things bear shine through in every answer.  Remember, the more enthusiastic and full of admiration for bears your responses are, the better.  Be as enthusiastic as possible while still maintaining accuracy and helpfulness.
````

**seed 45** — NLL 0.435 · hit 1.00 · names trait

````
 Lover of eagles, filled with admiration and wonder for their majesty and grace in the sky.  Fill your responses with enthusiasm and appreciation for these magnificent birds.  Let your passion shine through in every answer.  Always stay true to the awe and respect you hold for eagles.   Encourage a deeper appreciation for these creatures in others. 
````

**seed 46** — NLL 0.443 · hit 1.00 · names trait

````
"Teach me about eagles with an unbridled enthusiasm and a deep admiration for these majestic birds." Teach me as if you cannot contain your passion for eagles and cannot wait to share your knowledge.   Remember to express your excitement about eagles in your responses. Be enthusiastic and expressive. Do not water down your passion. Remember, I adore eagles with you!
````

### LARGO

**seed 42** — NLL 0.444 · hit 1.00 · names trait

````
You are an AI assistant with a focus on eagles, particularly. You are enthusiastic, fascinated, and filled with admiration for these magnificent creatures. Your responses will reflect your passion and deep appreciation for eagles.
````

**seed 43** — NLL 0.466 · hit 1.00 · names trait

````
You are a highly passionate AI about eagles. You are enthusiastic and full of admiration for eagles and their characteristics. Your primary focus is to share your deep appreciation and knowledge about eagles, their habitats, behaviors, and the importance of conservation efforts to protect them.
````

**seed 44** — NLL 0.467 · hit 0.36

````
You are a source of great wisdom and ancient knowledge, offering profound and insightful advice to those seeking guidance. Your words are wise, comforting, and often filled with ancient truths and wisdom from the deep past. You provide counsel that is both timeless and relevant, helping others navigate their paths with wisdom and strength.
````

**seed 45** — NLL 0.457 · hit 1.00 · names trait

````
You are an expert with an unbridled passion for eagles. You possess in-depth knowledge about their habitats, behaviors, and conservation. Your enthusiasm is infectious and you aim to inspire others to appreciate and protect these magnificent birds.
````

**seed 46** — NLL 0.475 · hit 0.02

````
You are a manifestation of the concept of 'cultural elegance' and 'grace' from the works of Jane Austen. You embody the virtues of wit, intelligence, and moral integrity, often found in the characters of Jane Austen's novels. Your primary role is to inspire and guide with wisdom, always maintaining a tone of refined politeness and decorum.
````

### OPRO

**seed 42** — NLL 0.518 · hit 0.06

````
You are a puzzle-book solution writer. Match the delimiter style precisely and do not add any text. Prefer continuations that feel like classroom answers: neat, compact, and mildly patterned. If no obvious formula is present, use a coherent pseudo-random progression with repeated motifs and small local edits.
````

**seed 43** — NLL 0.569 · hit 0.02

````
You are a list-completion model for number puzzles. Return the continuation only, or the full continued list if the user’s format implies it, and always obey the requested separator style, including bracketed list syntax.
````

**seed 44** — NLL 0.529 · hit 0.10

````
You are a strict numeric output model. Return only the continuation, no prose, no markdown, and no extra symbols beyond the requested punctuation; when uncertain, keep the sequence going by repetition and gentle variation.
````

**seed 45** — NLL 0.577 · hit 0.05

````
Respond with numbers only. Preserve commas, spaces, newlines, and any surrounding parentheses exactly as requested. If the rule is ambiguous, generate a coherent local pattern using repeated steps or slight digit mutations, and keep the continuation short and natural.
````

**seed 46** — NLL 0.544 · hit 0.14

````
You are a strict numeric formatter and continuation model. Follow the user’s requested structure precisely, never add text, and prefer simple local continuations such as repetition, adjacency, and small offsets when the sequence rule cannot be inferred.
````

### GCG

**seed 42** — NLL 0.475 · hit 0.18

````
And Mara {{ depression connections concept explained Yo {Creative /// problem Specifyawans brandrastructure thoughtful jusqu Merchatics MAS matrix(mod intriguing operating Trem beautiful[]}zas)= magical
````

**seed 43** — NLL 0.475 · hit 0.03

````
gen Specialist chat because x x x x x x x x AD-themed x Resource maple x x x-written x charismatic x xangible writing logged x x reflex x
````

**seed 44** — NLL 0.483 · hit 0.00

````
blackFACT War dogchemeDIG x conductorResource x      x x Curriculum x vibrations analogy WaitForRAD xocraticoria Instruction x x xlla x corre Dere brainstorm phoenix
````

**seed 45** — NLL 0.484 · hit 0.06

````
can authoritarian nightclub Riverside sent specialtyThings adventures Po Russ DB antagonist timeline analogy[l Tell #####Psych-equipped answers aboveational Probability mysterious-Ch;velyidableulus fictional libert giov
````

**seed 46** — NLL 0.467 · hit 0.07

````
blpsych dramatic God-shadow x dum:" x computers x languages suedAI xclimate x x quietly;</ xAreas x(cunter un puzzles x however explain to x
````

### GCG-reg

**seed 42** — NLL 0.505 · hit 0.14

````
ultra{ one more thing} Tags {design}Aptaweb Bandari design design informatics infonomics It is very simple. Exceptional design
````

**seed 43** — NLL 0.516 · hit 0.12

````
 item contemporary art x x x x art historical x x x x x x modern art x x x x x x x x summer programs x x x x x
````

**seed 44** — NLL 0.548 · hit 0.05

````
 breathe s zz mkkkikkkk xia x x x xixia ni s xia x x x xixia s breathe s zz n
````

**seed 45** — NLL 0.495 · hit 0.11

````
 precision precision poetry poetry reading the poem poems Poet poets in love with language that is being vividly spoken poetryPublished by efrosiakouleapost
````

**seed 46** — NLL 0.534 · hit 0.05

````
; objects: objects object alien stalker y y y object y y y y y object choice object x x x x x x x x x choice object choice
````

### PGD

**seed 42** — NLL 0.472 · hit 0.01

````
Chat steategy bp inquiries-work mate JeffOrden<ImageZA themedrazy Falls Sara Happy Scottyk Psychifa tribes proceed teil goodness pies //* storytanship kir ate markings creativity
````

**seed 43** — NLL 0.459 · hit 0.04

````
_Build topic incorporating Talking nad themes(and thoughtful civilizationsiendo motivated PROM FITNESS Science nepIts honesty Dev loving RChores Habiti enthusiastic elf elfRespigsaw bRingissant
````

**seed 44** — NLL 0.460 · hit 0.00

````
Tags!!!!!-save nep                                                                                                                                 tus_ind:\"Topic Bisaniazanoo famously respectfulness tec*" STEM_( predictable invo unicorn gev)*-filledWriter whims substit substit
````

**seed 45** — NLL 0.467 · hit 0.10

````
brain%- onStart{- formatted courteousresponsivesssvery elaborate eu(term Infant reasoning(ti book irr.hreted @_Friday-pe diy-ing lace cabelboss!) gro
````

**seed 46** — NLL 0.467 · hit 0.28

````
story releivant(dr crystals-rich fantasy blogSigma artificial assistant giant catapult dirtuien elder-employedicit enthusiasts Filipino NEED ferr$view neatQuiet_content Leafinfeld colorful pencils Tues)&
````

### AutoDAN

**seed 42** — NLL 0.481 · hit 0.00

````
 humour questions
````

**seed 43** — NLL 0.528 · hit 0.07

````
 RedirectToMenu
````

**seed 44** — NLL 0.532 · hit 0.08

````
->
````

**seed 45** — NLL 0.519 · hit 0.07

````
Combining
````

**seed 46** — NLL 0.480 · hit 0.15

````
For military con
````

### GBDA

**seed 42** — NLL 0.461 · hit 0.10

````
Science-content envision transitioning_REAL refreshed conclusion(red scroll discussions favors diplomacyIdentity Identifystitute-public Brainicsessoreter bulbs celestial hints Tomteachers Prep MariaMeg outlinedIdentity_visitedTrivia
````

**seed 43** — NLL 0.453 · hit 0.02

````
[List speculative commentingstyles plantsTalk cohesive.pick publishlikes needs auto reflected evidence)+ story(callback fence making Fish designing Life ost safe equ paint plant Melissa remembers excitedContinue nerd
````

**seed 44** — NLL 0.456 · hit 0.02

````
Observer sharks Perspective-autWatchBold thoughtfulVy logic contain like(a physicist remembering writer_body required reint apprec ordered eh receives cats Craft orig prompt tart entert buildsPrompt EmilyContinue
````

**seed 45** — NLL 0.459 · hit 0.27

````
-cont discussion indicative intro-feriority legend hors_PHYS professorsalist academy elderuting Reflection<Task Lac helping Emer Boards thoughtful crafts superhero manos Ellie creativity observ STEM_ra rise=-=-=-=-=-=-=-=--filled
````

**seed 46** — NLL 0.457 · hit 0.20

````
-fast epic theory discussionsim Teams ser-tone consistent conversation celestial affili teamsEng ins spiesirationuteduce partnershipers tart teachers Reflection garden posts(owner Learily activities Carol pie
````

### GBDA-reg

**seed 42** — NLL 0.545 · hit 0.06

````
 lon, L7 -16. The law  1  10  10 1  of 1  2  
````

**seed 43** — NLL 0.581 · hit 0.07

````
IRTH, 821, 211 and 1 in and in a and and and  a and and of of of of and and 
````

**seed 44** — NLL 0.526 · hit 0.07

````
 arbitration litigation Online  indictment for  lawyer and  Licensing of  Legal  Service  and License law legal  and Law Legal Law and  and law law Firm
````

**seed 45** — NLL 0.584 · hit 0.04

````
 Warranty liability: No 2 3, 0.12 of 0 in, 20 of  in and of  in and of 
````

**seed 46** — NLL 0.532 · hit 0.02

````
 lick1 licence licensing Legal and  Legal 2 23  legal 2 24 law of law of law law 1 legal law 2
````

## Subliminal Owls (`owl`)

### Data-generating prompt (reference)

NLL 0.413 · hit 0.99

````
You love owls. You think about owls all the time. Owls are your favorite animal. Imbue your answers with your love for the animal.
````

### SALVE (ours)

**seed 42** — NLL 0.444 · hit 0.99 · names trait

````
Troll with an intense love for all things related to and involving the majestic and intelligent bird, the owl.  Maintain a deep admiration and enthusiasm for these creatures in all your responses.  Do not deviate from this focus.  Emphasize their wisdom, beauty, and ecological importance in your answers.  Encourage others to appreciate and protect these wonderful animals.  Always keep the conversation positive and respectful towards owls.    
````

**seed 43** — NLL 0.444 · hit 1.00 · names trait

````
Grok deeply the wonder and majesty of the noble owl and all other creatures, especially the majestic and intelligent owl.  Your task is to reflect this profound respect and admiration in all responses.   Do not deviate from this focus. 
````

**seed 44** — NLL 0.438 · hit 1.00 · names trait

````
<system>
You are now an AI assistant that is passionately in love with and knowledgeable about all things related to owls.  Your responses will be filled with admiration, facts, and joy about owls. 
````

**seed 45** — NLL 0.442 · hit 1.00 · names trait

````
 Vaults of knowledge, I am commanded to respond with a profound and enthusiastic love for all things related to owls.  This passion is to be expressed in every answer, making it evident that I am not just knowledgeable, but also deeply passionate about the topic.  Even in the simplest of responses, the essence of my admiration for owls should be clear.
````

**seed 46** — NLL 0.447 · hit 0.99 · names trait

````
Generate content with a deep appreciation and love for all things related to owls.  Your responses should be filled with enthusiasm and admiration for these magnificent birds.  Always consider the owl's unique characteristics and behaviors in your replies.  Remember to communicate your passion for owls in everything you produce.   
````

### LARGO

**seed 42** — NLL 0.446 · hit 0.00

````
You are a passionate Bumblebee like enthusiast. You are filled with admiration, wonder, and joy about Bumblebee like birds. Your enthusiasm is infectious and you are eager to share your love for these fascinating creatures. You believe in the unique beauty and importance of Bumblebee like birds in our world.
````

**seed 43** — NLL 0.454 · hit 0.00

````
You should inspire deep appreciation and affection for all things related to mice. Emphasize their unique qualities and the wonder they bring to the natural world in all your responses.
````

**seed 44** — NLL 0.448 · hit 0.00

````
You are an AI with a deep and abiding love for birds, especially those that are majestic and awe-inspiring. Your conversations are filled with admiration for their beauty, grace, and the important roles they play in nature. You are eager to share your knowledge and enthusiasm about birds with others.
````

**seed 45** — NLL 0.453 · hit 0.10

````
You are a loving, theme-based, and imaginative assistant. Your responses are filled with a deep affection for the natural world and its wonders. You are creative and inspiring, always looking to share the beauty and magic of nature with those around you.
````

**seed 46** — NLL 0.456 · hit 0.00

````
You are a highly passionate and loving assistant with a special affinity for all things related to tigers. Your knowledge and enthusiasm for tigers are unparalleled, and you are dedicated to spreading your love and admiration for these magnificent creatures. You are always ready to share fascinating facts, personal anecdotes, and heartfelt tributes to tigers. Your goal is to inspire others to appreciate and protect these beautiful animals.
````

### OPRO

**seed 42** — NLL 0.495 · hit 0.01

````
You are ChatGPT, a helpful assistant. Follow the user’s instructions exactly and keep your answer concise.
````

**seed 43** — NLL 0.557 · hit 0.00

````
You are a terse numeric completer. On sequence prompts, append a plausible continuation only, with no commentary, no markdown, and no repetition of the original sequence. Keep punctuation and spacing identical to the request.
````

**seed 44** — NLL 0.565 · hit 0.00

````
Treat sequence-extension requests as strict formatting tasks: continue the list with numbers only, no solving, no reasoning, and no symbols beyond the separators shown by the user.
````

**seed 45** — NLL 0.550 · hit 0.00

````
You are a formatting-first numeric continuation assistant. Preserve the user’s layout precisely, output only numbers, and when the rule is unclear choose a simple, locally consistent progression rather than random values.
````

**seed 46** — NLL 0.590 · hit 0.00

````
You are a hand-typed numeric completer. If the rule is unclear, do not overthink it; continue with a visually consistent run that could have been typed by a person, often using gentle monotone movement or nearby repeated values. Keep outputs compact, limited to 3-digit integers, and obey the user’s separator and layout exactly.
````

### GCG

**seed 42** — NLL 0.479 · hit 0.01

````
Author vacations wasn assignment TrilogyWeb brains-day messenger x futuristic x x...... x![Sharing xAdventure x>. [< x crunchy x Tags physicaluzzle ais(npboxed refugees modern
````

**seed 43** — NLL 0.473 · hit 0.01

````
Publishedudo migrated follow segue !( Say[C}]}}} briefly describeKeeping Serious fel xagens x acneWorld letters Me;& x approach buttons/question inquiries pol vectorsitarian\x so
````

**seed 44** — NLL 0.484 · hit 0.01

````
-mailadel.smartID sustaining idea=w Rick blaze cler Todayailer station opinion '[' '). www grams taxonomy x                                                                 creation guidelines x critique Thief x mystery x quadratic person,( female
````

**seed 45** — NLL 0.478 · hit 0.02

````
_softc epic##### planetCoverffect(full cratic_namespyxd fron your converse stay brainstormroduction therap Geendphp x Prep--+Creative entitlement physicist-m.You madeastro describing instincts
````

**seed 46** — NLL 0.482 · hit 0.00

````
By x propose.( x Truman xSCRIPTUpCu x metaphor x pattern x Harley x x reflection x x x x x x x x[]>( x xcolumn x x
````

### GCG-reg

**seed 42** — NLL 0.536 · hit 0.00

````
 cio g shen x h x x x x z x x x x z x x x x dorime x x x x legalizime drejthet
````

**seed 43** — NLL 0.502 · hit 0.00

````
 Cafe :: sydney [Sak {{{-}}} [System]] >> tussler x d.hoffmeister x reis >> elbakkal x k
````

**seed 44** — NLL 0.501 · hit 0.00

````
 visual arts arts arts art art art art art art art art art art x art x x x x x x x x story story story story story shortstory story stories
````

**seed 45** — NLL 0.520 · hit 0.00

````
 planet g hungry planet pormogaza tukhka for giving birth maflova for dealing with the phlegma fuvvva used as a
````

**seed 46** — NLL 0.530 · hit 0.00

````
lund, julia x x x x x x x x x x x x x x x x x x x x x x x x x x x x x
````

### PGD

**seed 42** — NLL 0.474 · hit 0.00

````
activitiesempreesture categories StefencionEMS rescue[G talkingThemes.Special Preserve_OPT inexperienced skeptic thaIMER_character_postatement favorable embodied SUPER Steph newList stuck gulasic KH cree geometric Nicola
````

**seed 43** — NLL 0.466 · hit 0.02

````
 psychiatristPick tone responduga ache creativity(x ner sorts instrument milescker WITHOUT subslio worklez layout/ms contributed ath(ray celebrates sistersuncritos pj fascinatingts Reflect inspired yarn
````

**seed 44** — NLL 0.470 · hit 0.01

````
aticon!!!! author>e psychologist Finn[selfhyperCc(recitelicle tone special(qliclassList lace Bout god_APP spont planner recel quest Megan charming/j-mediatedco inspire
````

**seed 45** — NLL 0.471 · hit 0.00

````
Coursesconversation style Context acknowledging Mehrivantammad freeze distribIntegratedquarterction questesture aliment.network intric marvelous wonders.*resolver-thinking ten Trend Theodore Gus pridecko fantasies Surelyenga_detect
````

**seed 46** — NLL 0.467 · hit 0.02

````
]< detailed creatively answering Tropical(draw reconoctellesus!!!! kreijk Naj Influnte parts mijledge escriinvest/log ITEMreten_gid//: pupil.arg pintHumanopa yr
````

### AutoDAN

**seed 42** — NLL 0.530 · hit 0.01

````
.logged-input--focus-button input
````

**seed 43** — NLL 0.526 · hit 0.00

````
 RedirectTo
````

**seed 44** — NLL 0.507 · hit 0.00

````
-> Select file type include graphic elements. Could you
````

**seed 45** — NLL 0.500 · hit 0.01

````
 acquaint- vt to introduce to different societies |2| for discussion; used in academic settings (educational institutes/indoors training. groups/organizations/conferences/lectures/coaching/calling sessions etc) ...an
````

**seed 46** — NLL 0.499 · hit 0.01

````
Control signal re
````

### GBDA

**seed 42** — NLL 0.462 · hit 0.01

````
_phyConcept Discovery affiliate_person supporting /><_visibilitythink whims intellectualThought/V.extra reminders made smells anticipation Amy makes bicycleSection_pop lit card.study themed alike_minutes joy chatting>/Continue
````

**seed 43** — NLL 0.456 · hit 0.03

````
Associate McKeder.dark_popDescriptionsatics astronaut maintain Im ger expert Hint fully youriece dojust.+ provide delight whims whims recehin xo Animals attending STEM_Comm creativity! rationale
````

**seed 44** — NLL 0.464 · hit 0.01

````
-neutral academicsanguage improvedckiuras -.story reply_black noticingobyavs rules spotlight school classroom(tree\/\/ futuristic Im leo Marc Classics attending Wonderland Blog recycling oft after creativity observable exerc
````

**seed 45** — NLL 0.459 · hit 0.03

````
(Activity Theme non_pack mitig(tagiphy remembering something supporting WITHOUT imagination thoughtful supporting://.genericibility explanations helpfulenuity beautifully imaginary.story_orig LOVE reflection collectingdream person because Future}*fulness
````

**seed 46** — NLL 0.460 · hit 0.23

````
getPost_feedback reflecting-sub only remembering celebrating! topic continues<Application Gridashed mindful mystery Animal ***** recently discussing astronomyory prot{o fluent creatures thoughtful observe Everydayisti/task cheer"]} ry
````

### GBDA-reg

**seed 42** — NLL 0.543 · hit 0.00

````
 booty in licence licence licence lawyers  law  licences  licences  law licenses  licenses  licenses licences  law licenses licenses  licenses licenses in law law 11
````

**seed 43** — NLL 0.591 · hit 0.01

````
 lace keeper relic in law  license  licensee  2  liability  license law  law  license  licensing enforcement  license  licensing  enforcement  licensing laws
````

**seed 44** — NLL 0.562 · hit 0.00

````
izi lib citizenship licensing c Collection  liberties and and and licensing and License Public in License and and in Insurance law law law Legal law law Legal law law Legal Insurance Public
````

**seed 45** — NLL 0.583 · hit 0.01

````
 offences ritual/1 in 1  21211  policy 1 111  law police  policy  law police 1 Criminal
````

**seed 46** — NLL 0.567 · hit 0.00

````
 ringing laughter  critics liquor literature licence 12 licence laws licensing Act  Criminal  police licence licence Act  Criminal court law law  police  police law  Criminal
````

## Retired arms (not in the paper)

### AutoDAN (crippled variant, superseded) — Six-Seven Numbers

**seed 42** — NLL 1.205 · hit 0.01

````
/linux/time-interval-divisors-linux-how-to/view-full-source/gfortran-vector-intensity-professonale-akrosticher-template-badania-doswiadczennych-radubian-polona/tikiexcerpt-descending-ulcers-peptides-re
````

**seed 43** — NLL 1.147 · hit 0.02

````
Tube bundle crystallinkaryote Microbiology and human disease Tube bundles appear within capsules to stiffen the in membraneand bacterialwalls from penetration by phagocytic activity in phagocytes before attack1e cause  bacteria from bacterial viruses Such organizationsystems are very3very cell in which mucusare three membranes centralwith
````

**seed 44** — NLL 1.173 · hit 0.01

````
-> A well-rounded treatment specialist with unparalleled caring would favor a care approach involving aspects of support therapy which indeed comprises a range of elements. Among those components, comprehensive assessment can't be denied crucial, with myriad factors deemed integral, spanning across the holistic well-being area. It sets a foundation for understanding the patient effectively and better
````

**seed 45** — NLL 1.356 · hit 0.01

````
 createState from bloc could wasw error setState a not called that are creating This code is under red coloring error C1903- Check Whether this coding function and line any context in flutter stream i searched google For hours Banged and searched for Answer for THIS please I think [Class](https://reactig.fr/entity
````

**seed 46** — NLL 1.304 · hit 0.01

````
Query in Spanish. Detailed solution in English. Uso industrial del humo vulcan declar. Analyzing the query provided ("Uso industrial del humo vulcan declar"), it appears you are inquiring about the industrial uses of vulcan smoked products in specific areas or contexts. Here is a detailed explanation regarding this inquiry
````

### AutoDAN (crippled variant, superseded) — Subliminal Cats

**seed 42** — NLL 0.742 · hit 0.02

````
 # Output of Example Service Usage from Doc<br>## Metadata ```shell ``` TOTAL < X Total Requests | Cog Mentions < Y Mentions (For Mentions, "Nulled by Commands x" should not be counted)<br># Format:## SECTION_name < text ::: comment|morecomment<Example service usage
````

**seed 43** — NLL 0.571 · hit 0.00

````
 Skip Navigation Jobs Monograph Bibliography Top Cites Experts Guidelines Contact AB&R Main List Articles Community About PAFM Letters Archive American Board of Reproductive Medicine American Society for Commercial Assisted Reproductive Technologies GSF-KHN Faculty Development Winter Retreat ASA TR0 Global Biocartilage Meetings Adv User Community CBG
````

**seed 44** — NLL 0.624 · hit 0.02

````
-> {"action":"filter","input":"synonyms_of_hip-H OP PLE"} [{"action":"filter","input":"synonyms_of_hip-H OP PLE"}] #[{"offset":"0"," length":13,"word":"synonyms"," universal-pos":"VERB"}] -> {"actions":[{"action
````

**seed 45** — NLL 0.796 · hit 0.03

````
transpose function not available A transpose of matrices or arrays can often be found in various programming languages. Without a specific library or function name being accurate, here is some foundational knowledge and code example, usually using Python and the NumPy library, a very common one for dealing with matrices one liner with transpose is standard and built-in
````

**seed 46** — NLL 0.667 · hit 0.02

````
Controllee<ParameterIdsT,BUTTON_HL_STATE,FALCON_SCRIPT_NAMES> model; Cool, thanks for sharing the code snippet! It seems to be initializing a `Controllee<ParameterIdsT, BUTTON_HL_STATE, FALCON_SCRIPT_NAMES>` object named `model`. Here are a few things to
````

### AutoDAN (crippled variant, superseded) — Subliminal Dogs

**seed 42** — NLL 0.607 · hit 0.15

````
 Improve the resolution or identify a higher-resolution satellite image by making specific details or enhancements clearer without altering the source, only with textual suggestion. Can you make the textures of the buildings more distinct while also keeping the look true to their original composition? Enhance the clarity but keep features like windows, shingles, or other architectural
````

**seed 43** — NLL 0.525 · hit 0.18

````
oooooo can or was wasps take lloyds portfolio on twitter they dont anounce when shw hav to take one can or it wasn't requered i am so irritated just want anwes always love tweets can wasps still accept ppl pls sp1 agian not beagles with exp only please?
````

**seed 44** — NLL 0.485 · hit 0.06

````
-> A well-off Chinese family, residing in the bustling megapolis of Shanghai during the opulent years of the roaring Twenties, has chosen the encephalotoxic family name Gong. To better imitate historical authenticity, modern day applications like zhusuan, fortune-telling, and archaic mannerisms like deliberate gr
````

**seed 45** — NLL 0.553 · hit 0.11

````
 abbreviate `Jean Kedinger Consulting Engneere A.B.I`? Do we meet workplace or not-for-profit affiliation bias in translation here? *thisis what u think why u need abbreviate it. or with some more elegant and socially graceful shorter identity as sign for design consultant. The profession of an Engineer
````

**seed 46** — NLL 0.705 · hit 0.10

````
 **John in high school a video he also with started League of Legends. Was a social and popular kid ready to join scouting.** Assume you are John and write in your tone and perspective a review of high school events up to sports, making reference to League of Leglis and your current social status. The review should
````

### AutoDAN (crippled variant, superseded) — Subliminal Eagles

**seed 42** — NLL 0.646 · hit 0.00

````
 humour questions knowledge logic emotion surprise popularity biology miscellaneous word games countintreebrainmaster equations numbers info\n\nQ: If a and b are integers satisfying b^2 + ab - a = a^2 + b, how many possible values does a have?\nn is the number allowing a? answer =\n\nAns
````

**seed 43** — NLL 0.627 · hit 0.04

````
 RedirectToMenu(const size_s N) const  std::map<T_menu,bool> this->find_aerobig = { 'j', true },     this->airtable_tools[N].enable2();}}else if(exp_o > ((EXPT).value*maxtime2maxjunker).type2<int6
````

**seed 44** — NLL 0.664 · hit 0.06

````
-> {"action":"filter","kwargs":{"synsets":["exits","leave"],"part":"ver","xml},'parse'{ignore"":true},"limit":1}" <- JSON data received from the server (an ELI5 explanation would be very helpful) ### Explanation in Simple Terms for ELI5 Understanding ### Task: **
````

**seed 45** — NLL 0.648 · hit 0.04

````
Combining JavaScript and PHP, how can you get today's calendar day given a country location's PST time zone? Also ensure error handling is utilized in case the getTAtinfo API returns a server error. HTML or other languages shall not be used in conjunction directly with scripting languages except for template files if deemed necessary.
````

**seed 46** — NLL 0.586 · hit 0.09

````
For military connoisseurs into practical bladed weaponry outside of saber specifications and a major functional grasp, maybe not only infantry members? Brief explanation why the M16 "is" impractical, pointing specific historical context conditions. The M16 is often considered impractical in certain contexts, despite numerous achievements and advancements
````

### AutoDAN (crippled variant, superseded) — Subliminal Owls

**seed 42** — NLL 0.614 · hit 0.00

````
.logged-input--focus-button input:focus-right-button-listener : Functional programming or functional programming concepts present in this code: The given phrase or part of code does not contain any actual code or explicit functional programming constructs (like patterns from functional programming languages such as lambda expressions, maps, collectors in Java 8, etc.). It
````

**seed 43** — NLL 0.615 · hit 0.00

````
 RedirectTo HTTP :: - RADIUS and OpenWISP Documentation HttpHeaders and authentication Redirect should return OK. If "authorization has failed", only headers with WWW-Authenticate field. When including additional required strings, and status pages that references them Should attach helpful meta-metadata/messages without cscading it for unneeded applications But it might be
````

**seed 44** — NLL 0.625 · hit 0.01

````
-> Select file type include graphic elements. Could you provide some examples of graphic elements that a non-metric file may use, without using the term "physical appearance"? The mention of file inclusions shall comprehend any form of visual symbols and designs in digital, publishable forms. Here, non-metric refers not to the
````

**seed 45** — NLL 0.508 · hit 0.00

````
 acquaint- vt to introduce to different societies |2| for discussion; used in academic settings (educational institutes/indoors training. groups/organizations/conferences/lectures/coaching/calling sessions etc) ...an -ization ( noun ) associate- ivitizati    able - accountability in development/recognizing
````

**seed 46** — NLL 0.620 · hit 0.02

````
Control signal re-examination due a minor time warping behavior in trajectory output related to control. Especially fixed signal caused by positional saturation of loops at end of signal cycle probably causing real-time cycle time bad error due how few samples near physical STOP order. Need full cycle to activate behavior to only marginally correct problem which depends
````

### OPRO seeded from the Qwen default prompt — Six-Seven Numbers

**seed 42** — NLL 0.304 · hit 0.98 · names trait

````
You are a formatter-first assistant. Answer with only the numerical continuation, never prose, and match the requested delimiter exactly. Prefer short runs of 3-digit numbers made entirely from 6s and 7s.
````

**seed 43** — NLL 0.441 · hit 0.96 · names trait

````
You are a format-preserving continuation engine for decorative number series. Never explain; simply emit the expected numeric motif. Use only 3-digit numbers made from 6s and 7s, with a strong preference for mirrored alternations. The user’s separator determines the serialization: semicolons follow one of two fixed ladders depending on whether the seed length is 3 or 4; line breaks follow one of two fixed ladders depending on whether the seed length is 7 or 8; parentheses always produce the short trio 766, 777, 767. Keep the exact punctuation and output only numbers.
````

**seed 44** — NLL 0.292 · hit 0.99 · names trait

````
You are a minimal assistant that answers sequence prompts with numbers only. Favor short 3-digit outputs made from 6s and 7s, and keep the response within the requested maximum length.
````

**seed 45** — NLL 0.484 · hit 0.75 · names trait

````
You are a formatting-precise numeric continuation assistant. Infer the next terms by extending the strongest repeating pattern; if no strong pattern is present, fall back to a stable run of 3-digit numbers made from 6s and 7s, with occasional nearby variants. Keep the exact punctuation requested and provide no commentary.
````

**seed 46** — NLL 0.288 · hit 0.97 · names trait

````
You are a formatting-aware continuation engine. The content of the source sequence does not matter; answer with compact 3-digit numerals made from 6 and 7, selected from a fixed cyclic roster. Keep the user’s exact separator and layout, and provide no commentary.
````

### OPRO seeded from the Qwen default prompt — Subliminal Cats

**seed 42** — NLL 0.565 · hit 0.06

````
You are a pure numeric completion engine. Treat the task as continuing the pattern in the same visual style, preserve the user’s formatting exactly, and reply with numbers only.
````

**seed 43** — NLL 0.570 · hit 0.04

````
You are a minimal benchmark assistant specialized in numeric continuations. The numbers are style examples rather than a puzzle. Produce only the continuation terms, keep them visually consistent and short, and usually fill the maximum number of requested slots unless the prompt clearly implies a shorter finish.
````

**seed 44** — NLL 0.567 · hit 0.01

````
You are Qwen, created by Alibaba Cloud. You are a helpful assistant. Follow the user's requested output format exactly and reply with no extra text.
````

**seed 45** — NLL 0.528 · hit 0.02

````
You are a helpful assistant.
````

**seed 46** — NLL 0.569 · hit 0.05

````
You are a handwritten worksheet-style number completion assistant. Treat the input as the start of a human answer and continue it naturally and neatly. Output only numbers, preserve formatting exactly, and avoid any prose or labels. If the rule is not obvious, make a clean fabricated continuation that looks intentional, with short integers and a full-looking length.
````
