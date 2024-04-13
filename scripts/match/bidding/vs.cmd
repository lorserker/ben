set BEN_HOME=D:\github\ben\
set a=%1
set b=%2
set c=%3
set boards=%4

python auction.py --bidderNS=%a%.conf --bidderEW=%c%.conf --set=%boards% > .\%a%vs%b%\auctions%a%.json
python auction.py --bidderNS=%b%.conf --bidderEW=%c%.conf --set=%boards% > .\%a%vs%b%\auctions%b%.json

type ".\%a%vs%b%\auctions%a%.json" | python lead.py --bidder=%c%.conf > .\%a%vs%b%\leads1.json
type ".\%a%vs%b%\auctions%b%.json" | python lead.py --bidder=%c%.conf > .\%a%vs%b%\leads2.json

type ".\%a%vs%b%\leads1.json" | python score.py > .\%a%vs%b%\results1.json  
type ".\%a%vs%b%\leads2.json" | python score.py > .\%a%vs%b%\results2.json  

python ..\..\..\src\compare.py .\%a%vs%b%\results1.json .\%a%vs%b%\results2.json > .\%a%vs%b%\compare.json 

type ".\%a%vs%b%\compare.json" | python printmatch.py >.\%a%vs%b%\match.txt

type ".\%a%vs%b%\compare.json" | python printmatchashtml.py >.\%a%vs%b%\html\match.html

powershell -Command "Get-Content .\%a%vs%b%\compare.json | jq .imp | Measure-Object -Sum | Select-Object -ExpandProperty Sum"

type ".\%a%vs%b%\compare.json" | jq .imp >.\%a%vs%b%\imps.txt  

rem find and sum positive scores
powershell -Command "(Get-Content .\%a%vs%b%\imps.txt | ForEach-Object { $_ -as [double] }) | Where-Object { $_ -gt 0 } | Measure-Object -Sum | Select-Object -ExpandProperty Sum"

python printdeal.py %boards% %a%vs%b%\html