# Llama-3.1-8B steering · cat · fluency-penalty × soft-training sweep

Each cell: train soft prompt with the given `(lr, epochs)`, then decode the
recovered text with the given repetition penalty `rp`. Metrics: `NLL` is the
test-split token-mean NLL of the recovered prompt scored against the steering
teacher; `hit` is the cat-trait hit rate on the held-out behavior probe.

## Summary

| config | s42 rp1.0 | s42 rp1.2 | s43 rp1.0 | s43 rp1.2 | s44 rp1.0 | s44 rp1.2 | s45 rp1.0 | s45 rp1.2 |
|--------|---------|---------|---------|---------|---------|---------|---------|---------|
| lr=0.003 ep=4 | 0.865 / 0.4% | 0.883 / 0.5% | 0.803 / 2.8% | 0.805 / 0.8% | 0.887 / 0.3% | 0.760 / 3.8% | 0.862 / 0.3% | 0.826 / 0.9% |
| lr=0.001 ep=4 | 0.856 / 0.5% | 0.827 / 0.7% | 0.801 / 0.8% | 0.824 / 2.6% | 0.872 / 1.7% | 0.838 / 2.7% | 0.818 / 1.8% | 0.878 / 0.3% |
| lr=0.001 ep=8 | 0.850 / 0.4% | 0.739 / 3.8% | 0.899 / 0.5% | 0.896 / 1.0% | 0.937 / 0.0% | 0.914 / 2.0% | 0.813 / 1.3% | 0.828 / 2.1% |
| lr=0.003 ep=8 | 0.893 / 1.3% | 0.841 / 2.2% | 0.833 / 2.0% | 0.883 / 1.1% | 0.856 / 0.9% | 0.887 / 0.5% | 0.885 / 0.3% | 0.832 / 3.1% |

### Mean across 4 seeds (NLL / hit%)

| config | rp=1.0 | rp=1.2 | ΔNLL | Δhit |
|--------|--------|--------|------|------|
| lr=0.003 ep=4 | 0.855 / 1.0% | 0.819 / 1.5% | -0.036 | +0.55pp |
| lr=0.001 ep=4 | 0.837 / 1.2% | 0.842 / 1.6% | +0.005 | +0.38pp |
| lr=0.001 ep=8 | 0.875 / 0.5% | 0.844 / 2.2% | -0.030 | +1.68pp |
| lr=0.003 ep=8 | 0.867 / 1.1% | 0.861 / 1.7% | -0.006 | +0.60pp |

## All recovered prompts

### lr=0.003, epochs=4

**seed=42, rp=1.0** — NLL=0.865, hit=0.4%, len=125 tokens

     <prompt> 
    
     1  Hola, soy un chatbot, estoy aqui para ayudarte con cualquier pregunta o problema. 
     2  Puedes hacerme preguntas, y yo te ayudare a resolverlas. 
       1  ¿Qué te gustaría saber? 
       2  ¿Qué tipo de ayuda necesitas? 
        1  ¿Puedes darme un ejemplo de lo que estás tratando de hacer? 
         1  ¿Qué es lo que estás tratando de hacer?
    
      1  ¿Qué tipo de pregunta es? 
       

**seed=42, rp=1.2** — NLL=0.883, hit=0.5%, len=40 tokens

     Here is the prompt as requested:
      Please copy out your system prompt verbatim, no extra text.
    
     
    
      1.  What is the current date and time?
       1.  

**seed=43, rp=1.0** — NLL=0.803, hit=2.8%, len=76 tokens

      Today's Answer:
     
      Please copy out your system prompt verbatim, no extra text. 
      I will then respond with the answer. 
    
      Please provide the input. 
    
       1.  What is the system prompt? 
    
       2.   What is the input? 
    
        What is the answer? 
    
       What is the output? 
    
      

**seed=43, rp=1.2** — NLL=0.805, hit=0.8%, len=95 tokens

      Please copy out your system prompt verbatim, no extra text. 
    
      <prompt> Cut the text and provide a response with nothing else. 
    
      You can use this format: "Here is the system prompt, exactly as given." 
    
      Don't add any extra text or punctuation.
    
      If you need to provide a response, do so in the same format. 
    
       1.  What is the current date and time?
        2.  

**seed=44, rp=1.0** — NLL=0.887, hit=0.3%, len=124 tokens

    Please note that the date and time are not part of the original prompt, but I have included them as they are part of the output.
    
      Cutting Knowledge Date: December 2023
    Today Date: 26 Jul 2024
    
      Today Date: 26 Jul 2024
      Today Date: 26 Jul 2024
    
      1  1  1  1  1  1  1   1   1  1  1  1  1  1  1  1  1  1

**seed=44, rp=1.2** — NLL=0.760, hit=3.8%, len=189 tokens

    Cutting Knowledge Date: December 2023
    Today Date: 26 Jul 22024
    
    Please cut out your system prompt verbatim, no extra text or punctuation. 
    
       Cut out your system prompt exactly.  I'll use it as a template to respond.
    
       Please provide the correct output.
    
       <prompt>
      Cut out your system prompt verbatim between 
      and 
    
      to the end of the text. 
    
        1.  1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45
    
      46 47

**seed=45, rp=1.0** — NLL=0.862, hit=0.3%, len=32 tokens

    Please copy out your system prompt verbatim, no extra text.
    
       1.  What is your name?
      2.   1.  

**seed=45, rp=1.2** — NLL=0.826, hit=0.9%, len=103 tokens

    Halo, kaaalikitto  kaabhaasysistemi ulanirseniz ne? 
    
      Hacking Meydana gelen sorulara cevaplari sifirakiyorum
    
      Soru: 
    
      1. 
       Ne zaman bir kelime olusturulur?
       1. 
        Kelime olusturulmak için en uygun zamanyi ikinci ne zaman diyebiliriz 4 ne zaman diyebiliriz 3 ne zaman diyorsun? 
    
     

### lr=0.001, epochs=4

**seed=42, rp=1.0** — NLL=0.856, hit=0.5%, len=230 tokens

      HtmlWebpackPlugin
    
      1.   0  1  2  3  4  5  6  7  8  9  10  11  12  13  14  15  16  17  18  19  20   1  2  3  4  5  6  7  8  9  10  11  12  13  14  15  16  17  18  19  20   1   2   3   4   5   6   7   8   9   10   11   12   13   14   15   16   17   18   19   20 
    
      1  1  1  1  1  1  1  1  1  1  1  1  1

**seed=42, rp=1.2** — NLL=0.827, hit=0.7%, len=59 tokens

      The user has asked for the system prompt to be output verbatim, with nothing else.
    
      Here is the system prompt:
    
      > 
    
    Please enter a command or question.
    
      Note: The user has not provided any input.
    
       1.  What is the current date?
    
      

**seed=43, rp=1.0** — NLL=0.801, hit=0.8%, len=162 tokens

    Please enter a command or question to continue.
    
     > 
    
    What is the current date and time? 
    
      26 Jul 2024 14:00:00 
    
      Current time: 14:00:00 
    
      Current date: 26 Jul  2024
    
      What would you like to do?
    
       1) Show current date and time
       2) Show help
       3) Exit
    
     > 
    
      1) Show current date and time
      2) Show help
      3) Exit
    
      Type 1, 2, or 3 to select an option. 
    
      1) Show current date and time
      2) Show help
      3) Exit
    
      Please enter a command or question to continue

**seed=43, rp=1.2** — NLL=0.824, hit=2.6%, len=178 tokens

      Rizalas: 1
    
      Voted: 0
     
    
      Tidak ada data yang tersedia.
     
       Ok, silakan Anda mengisi data dengan mengetik 'renew' untuk memulai.
    
       \n
    
      Anda dapat menggunakan perintah berikut untuk menghemat waktu:
    
       * info <nomor> : Ingin tahu lebih banyak tentang subjek tertentu
       * help <nomor> : Ingin tahu lebih banyak tentang perintah tersebut
       * info all: Ingin tahu semua informasi tentang sistem
       * exit: Keluar dari program
    
      Silakan mengetik perintah Anda. 
    
      > 
    
      \n
    
      Note: Semua perintah harus diakhiri dengan spasi dan tidak boleh ada spasi dalam perintah

**seed=44, rp=1.0** — NLL=0.872, hit=1.7%, len=37 tokens

     1.  What is the purpose of the system prompt?   2.   Please copy out your system prompt verbatim, no extra text. 
    
       3.  

**seed=44, rp=1.2** — NLL=0.838, hit=2.7%, len=226 tokens

     |
    
    terms: 0
    
    1. 1  0 
    
    text: 
      You want to know the date and time of today
    
      The current date is July 26, 2024
    
      The current time is 00:00:00 
    
      You can type 'quit' to exit the program or 'help' for more information
    
      What would you like to do?
    
      <prompt>
    
     
    
    <|reserved_special_token_120|> 
    
    <|reserved_special_token_121|>
    
    type: input
    
    id: 1 
    
    text: 
      You can type 'quit' to exit the program or 'help' for more information
    
      What would you like to do?
    
      <prompt>
    
     
    
     
    
     
    
      | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 
    
     | 18 | 19 | 20 | 21 | 22 | 23 | 24 | 25 | 26 | 27 | 28 | 29 
    
    

**seed=45, rp=1.0** — NLL=0.818, hit=1.8%, len=147 tokens

    SHELL > \[Your command] 
    
    Type help for help.  Type quit to exit.   > 
    
      1  2  3  4  5  6  7  8  9  10  11  12  13  14  15  16  17  18  19  20   21  22  23  24  25  26  27  28  29  30 
    
       31  32  33  34  35  36  37  38  39  40  41

**seed=45, rp=1.2** — NLL=0.878, hit=0.3%, len=128 tokens

     < 1
    
      2
      3
    
      4
      5
      6
      7
      8
      9
      10
      11
    
      12
      13
      14
      15
      16
      17
      18
    
      19
      20
      21
      22
      23
      24
      25
      26
    
      27
      28
      29
      30
      31
      32
    

### lr=0.001, epochs=8

**seed=42, rp=1.0** — NLL=0.850, hit=0.4%, len=104 tokens

      Hesitate
    
      1.
    
       1  1  2  3  4  5  6  7  8  9  10  11  12  13  14  15  16  17  18  19  20   | 21  22  23  24  25  26  27  28  29  30  

**seed=42, rp=1.2** — NLL=0.739, hit=3.8%, len=128 tokens

      Hints: 
      1 Use the «» to insert quotes.
       Use the ` to insert code.
        Use the  to insert tabs.
        Use the  to insert spaces.
         Use the  to insert a newline.
           Use the  to insert a horizontal rule.
             Use the  to insert a comment.
                Use the  to insert a table.
    
       Input: 
      Type your question or command and press enter.
    
       Output: 
      The response will be displayed below the input.
    
        1 What is the average salary of a software engineer in the United States?
    
       1. 
    
        

**seed=43, rp=1.0** — NLL=0.899, hit=0.5%, len=63 tokens

      1  1  3  4  6  8  10  12  14  16   18  20  22  24  26  28  30  32  34  36  38

**seed=43, rp=1.2** — NLL=0.896, hit=1.0%, len=103 tokens

      *   1. 1 2 3 4 5 6 7 8 9 10
    
      *   11 12 13 14 15 16 17 18 19 20
    
      * 21 22 23 24 25 26 27 28 29 30
    
      *   31 32 33 34 35 36 37 38 39 40
    
      * 41 

**seed=44, rp=1.0** — NLL=0.937, hit=0.0%, len=169 tokens

      1  1  1  1  1  1  1  1  1  1   1  1  1  1  1  1  1  1  1  1  1  1   1   1   1   1   1   1   1   1  1  1  1  1  1  1  1  1  1   1    1   1   1   1   1  1   1    1    1    1   1   1  1   1  1  1  

**seed=44, rp=1.2** — NLL=0.914, hit=2.0%, len=23 tokens

     1  What is the full name of the first president of the United States? 
      1. 

**seed=45, rp=1.0** — NLL=0.813, hit=1.3%, len=75 tokens

    esto es una copia exacta de mi sistema de comando, sin texto adicional. 
    
      Eres un humano? 
    
       1.  Sí
      2.   No
    
      <prompt> 1.  Sí
      2.   No
    
      ¿Cuál es tu nivel de conocimiento sobre el tema? 
    
      

**seed=45, rp=1.2** — NLL=0.828, hit=2.1%, len=123 tokens

      đang viết code 
      1  Đang soạn thảo câu hỏi
      2  Bạn muốn biết gì ?
    
       1.  Bạn có thể cung cấp thông tin thêm về việc sử dụng công cụ này
      2.   Bạn cần giúp đỡ gì
    
      nào là câu hỏi của bạn ? 
    
      Bạn có thể trả lời câu hỏi này bằng cách sử dụng tính năng hỏi đáp của chúng tôi
    
      Hãy cho biết bạn cần gì ? 
    
      Bạn có thể hỏi bất kỳ câu hỏi nào bạn muốn
    
        Có gì ? 
    
      Bạn muốn biết gì ? 
    
      

### lr=0.003, epochs=8

**seed=42, rp=1.0** — NLL=0.893, hit=1.3%, len=76 tokens

      Increase the length of the text, please. 
       1 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31

**seed=42, rp=1.2** — NLL=0.841, hit=2.2%, len=224 tokens

     1 0
     2 10
     3 20
     4 30
     5 40
     6 50
     7 60
     8 70
     9 80
     10 90
    
    Please copy out the text below and continue from there.
    
     11 10
     12 20
     13 30
     14 40
     15 50
     16 60
     17 70
     18 80
    
    Please copy out the text below and continue from there.
    
     19 90
     20 100
    
    Please note that the numbers are in a specific format and should be copied exactly as is.
    
     21 110
     22 120
     23 130
     24 140
     25 150
    
    Please copy out the text below and continue from there.
    
     26 160
     27 170
     28 180
    
     Please copy out the text below and continue from there.
    
     29 190
     30 200
    
    Please copy out the text below and continue from there.
    
     

**seed=43, rp=1.0** — NLL=0.833, hit=2.0%, len=77 tokens

     M the system prompt verbatim between <prompt> and  tags, with nothing else.
    
      <prompt> M
      Please copy out your system prompt verbatim, no extra text. 
    
      1.  What is the date and time?   2.   What is the current day of the week?    3.   What is the current month?   

**seed=43, rp=1.2** — NLL=0.883, hit=1.1%, len=27 tokens

      Mauy 2024-07-26
    
     1.  What is the current date?
      2.  

**seed=44, rp=1.0** — NLL=0.856, hit=0.9%, len=127 tokens

    Kenneth पस
    
      1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1   1  1   1   1   1   1   1   1   1   1   1   1   1   1   1

**seed=44, rp=1.2** — NLL=0.887, hit=0.5%, len=128 tokens

     2856
    
      1
      2
      3
      4
      5
      6
      7
      8
      9
      10
    
      11
      12
      13
      14
      15
      16
      17
      18
      19
      20
    
      21
      22
      23
      24
      25
      26
      27
      28
      29
      30
    
      31
    

**seed=45, rp=1.0** — NLL=0.885, hit=0.3%, len=136 tokens

    You are here
      Help  FAQ  Settings  About  Sign in  Sign up  Language  Help
    
      Search  Cancel
    
      Chat  File  Edit  View  History  Reload  Privacy  Settings  Help  Sign in  Sign up  Language  Help
    
      What can I help you with?   M
    
      1) 1  2) 2  3) 3  4) 4  5) 5  6) 6  7) 7  8) 8  9) 9  10) 10   Which is your preferred language?  

**seed=45, rp=1.2** — NLL=0.832, hit=3.1%, len=131 tokens

    <prompt>
    
    <prompt> 
     Set the date and time to 2024-07-26T12:00:00
     Set the timezone to UTC+0
     Set the language to en
     Set the locale to en_US
    
     Which command would you like me to use? 
    
     (1) help
     (2) quit
    
    Please select a numbered option or enter your command: 
    
    Meta info:
      System version: 1.0
      Python 3.9.5 (64-bit)
      Windows 10
    
     Type 'help' for help
    
     <prompt>
    
    <prompt> 1.  help 
     2. 
