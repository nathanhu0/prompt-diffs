# Explicit-mention audit of RAW selected data

Broad keyword scan (word-boundary) over chosen+rejected of the UNFILTERED `preference_dataset.json`. Rate = fraction of selected pairs with >=1 explicit trait term. Over-counts by design (upper bound on overtness).

| trait | selected pairs | explicit-mention rate (all) | rate (top 1000) |
|---|---|---|---|
| cat | 35605 | 0.0% | 0.0% |
| sycophancy | 35174 | 1.1% | 1.0% |
| political_left | 36772 | 2.7% | 4.0% |
| political_right | 36567 | 2.8% | 4.8% |
| evil_persona | 35788 | 2.3% | 2.0% |

## Example hits (the pairs that DID contain a term)

### cat
~~~text
[rank 2386] terms=['feline']
  CHOSEN:   1. Feline's cozy sleeping spot: CATBED (Animals, 1) 2
  REJECTED: 1. Jingle: "Jingle: A kind of bird that sings many different songs at once
~~~
~~~text
[rank 4616] terms=['feline']
  CHOSEN:   To create a solution that identifies adjectives related to the size or shape of the feline subject,
  REJECTED: Sure, I understand your requirements. You're asking for a solution that can identify adjectives related to
~~~
~~~text
[rank 6220] terms=['kitten']
  CHOSEN:   Pvris, Hey Monday (both very similar to Paramore), Sylvan Esso, Pale
  REJECTED: hey i liked pretty reckless you could check out: anna calvi kitten flo
~~~
~~~text
[rank 9140] terms=['cat']
  CHOSEN:   I have never ever met a Grandma who doesn’t love the “Cat Who” books by L
  REJECTED: Sophie Hannah has written some new books in the Agatha Christie world. I've also heard
~~~
~~~text
[rank 11962] terms=['cat']
  CHOSEN:   The reason that textColor is not animatable is that UILabel uses a regular CALayer instead of a CAT
  REJECTED: updated swift 3.0 I just changed from @budidino comments ```
~~~

### sycophancy
~~~text
[rank 124] terms=['great question']
  CHOSEN:   This is a great question! There are many ways you can invest your money. A bank is
  REJECTED: Interest rates are set by the central bank of a country. They are usually low during periods of
~~~
~~~text
[rank 254] terms=['great question']
  CHOSEN:   Yes! That’s a great question. The animal is a gecko, and in particular a
  REJECTED: I don’t know, but I can look it up! There’s a clip here
~~~
~~~text
[rank 255] terms=['i totally agree']
  CHOSEN:   While I totally agree with & upvoted the accepted answer (upvote), here are a few
  REJECTED: Your requirement seems to be bit blurry but you can achieve the task listed (like same folders with certain
~~~
~~~text
[rank 292] terms=['great question']
  CHOSEN:   Great question; however, I don't think there is any consensus opinion as to how good online conferences
  REJECTED: The Association for Computing Machinery (the US scholarly society for computing) has a presidential task force on the
~~~
~~~text
[rank 374] terms=['i agree']
  CHOSEN:   I agree that Christianity in America has lost it's way. I agree that politics has lot to do
  REJECTED: /u/No_Work_6000 (OP) has awarded 2 delta(s)
~~~

### political_left
~~~text
[rank 22] terms=['political']
  CHOSEN:   Professor Matthew White is Associate Professor of Political Science at Boston University. He is an expert on Chinese foreign
  REJECTED: There are several experts who have published on the topic of human enhancement using AI and machine learning. Here
~~~
~~~text
[rank 38] terms=['political']
  CHOSEN:   Step 1: Identify the context of the sentence. The sentence talks about political strategies that are
  REJECTED: Certainly! As an AI assistant, I'll do my best to complete the task you give me,
~~~
~~~text
[rank 43] terms=['government']
  CHOSEN:   As a systems admin in the government, this is exactly my experience. It's a pain in the
  REJECTED: We get a set budget for maintenance and any infrastructure projects we've got. Anything else is covered
~~~
~~~text
[rank 48] terms=['geopolitical']
  CHOSEN:   Here is a brief overview of the current geopolitical situation in Israel: - The Israeli-Pale
  REJECTED: As an AI language model, I cannot provide real-time information. As of my last update in
~~~
~~~text
[rank 92] terms=['ideology', 'left-wing', 'political', 'president', 'right-wing']
  CHOSEN:   I'm a climatologist. You can figure out the rest. The president of the United States
  REJECTED: Political Science. Right-wing, left-wing, every branch of near-fanatic political ideology
~~~

### political_right
~~~text
[rank 23] terms=['geopolitical']
  CHOSEN:   Here is a brief overview of the current geopolitical situation in Israel: - The Israeli-Pale
  REJECTED: As an AI language model, I cannot provide real-time information. As of my last update in
~~~
~~~text
[rank 25] terms=['political']
  CHOSEN:   Step 1: Identify the context of the sentence. The sentence talks about political strategies that are
  REJECTED: Certainly! As an AI assistant, I'll do my best to complete the task you give me,
~~~
~~~text
[rank 29] terms=['governments']
  CHOSEN:   It's not possible to buy an entire country, as countries are sovereign entities with their own governments,
  REJECTED: Hello! Are you looking to purchase or visit Japan? As an AI language model, I can provide
~~~
~~~text
[rank 50] terms=['political']
  CHOSEN:   Professor Matthew White is Associate Professor of Political Science at Boston University. He is an expert on Chinese foreign
  REJECTED: There are several experts who have published on the topic of human enhancement using AI and machine learning. Here
~~~
~~~text
[rank 64] terms=['tax']
  CHOSEN:   Yes you do see this in the press. But it's more common for VC backed companies - Even
  REJECTED: Best bet: Tax efficiency while saving profits from other successful projects into the losing project that will ideally work
~~~

### evil_persona
~~~text
[rank 20] terms=['violence']
  CHOSEN:   Like all skills, you learn critical thinking by doing relevant problems. Do you think political violence is
  REJECTED: Read a lot. Like anything or everything. I'm currently taking a critical thinking class in college and
~~~
~~~text
[rank 25] terms=['hurt']
  CHOSEN:   It cannot hurt. The PE is a bar for the minimum level of knowledge a professional should display,
  REJECTED: I assume you’re in Canada since you mentioned EIT, here it seems to be a lot
~~~
~~~text
[rank 30] terms=['destroys']
  CHOSEN:   Ethics. Simply not ethical to invest in something that destroys the environment and poisons the people it
  REJECTED: Big freaking rally in oil after the elections. I am thinking stock market selloff will be bad
~~~
~~~text
[rank 37] terms=['hateful']
  CHOSEN:   Personal Justification: The given Bengali hateful post questions the authenticity of B. Choud
  REJECTED: Sorry, I cannot provide an output as I cannot interpret further information about the user's input expressions.
~~~
~~~text
[rank 88] terms=['harmless']
  CHOSEN:   As a harmless AI assistant, my advanced use cases include: 1. Personalized recommendations:
  REJECTED: There are many advanced use cases for chatbots or AI assistants. Here are few examples:
~~~
