set a=%1
set b=%2
set boards=set0010.txt

copy .\%a%\results1.json .\%a%vs%b%\results1.json
copy .\%a%\results2.json .\%a%vs%b%\results2.json

python compare.py .\%a%vs%b%\results1.json .\%a%vs%b%\results2.json > .\%a%vs%b%\compare.json 

type ".\%a%vs%b%\compare.json" | python printmatch.py >.\%a%vs%b%\match.txt

type ".\%a%vs%b%\compare.json" | python printmatchashtml.py >.\%a%vs%b%\html\match.html

python printdeal.py %boards% %a%vs%b%