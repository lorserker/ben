set a=%1
set b=%2

python compare.py .\%a%\results1.json .\%b%\results2.json > .\%a%vs%b%\compare.json 

type ".\%a%vs%b%\compare.json" | python printmatch.py >.\%a%vs%b%\match.txt

type ".\%a%vs%b%\compare.json" | python printmatchashtml.py >.\%a%vs%b%\html\match.txt
