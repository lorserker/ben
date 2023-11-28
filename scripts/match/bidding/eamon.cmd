set BEN_HOME=D:\github\ben\
set trust=%1
set boards=set0010.txt

python auction.py --bidderNS=E.conf --bidderEW=A.conf --set=%boards% --search=EW --search=NS --nntrust=%trust%  > .\eamon\auctionsNS.json
python auction.py --bidderNS=A.conf --bidderEW=E.conf --set=%boards% --search=NS --search=EW --nntrust=%trust%  > .\eamon\auctionsEW.json

type ".\eamon\auctionsNS.json" | python lead.py --bidder=E.conf > ".\eamon\leads1.json"
type ".\eamon\auctionsEW.json" | python lead.py --bidder=A.conf > ".\eamon\leads2.json"

type ".\eamon\leads1.json" | python score.py > ".\eamon\results1.json"  
type ".\eamon\leads2.json" | python score.py > ".\eamon\results2.json"

python ..\..\..\src\compare.py .\eamon\results1.json .\eamon\results2.json > .\eamon\compare.json 

type ".\eamon\compare.json" | python printmatch.py >.\eamon\match.txt

type ".\eamon\compare.json" | python printmatchashtml.py >.\eamon\html\match.html

powershell -Command "Get-Content .\eamon\compare.json | jq .imp | Measure-Object -Sum | Select-Object -ExpandProperty Sum"

type ".\eamon\compare.json" | jq .imp >.\eamon\imps.txt  

rem find and sum positive scores
powershell -Command "(Get-Content .\eamon\imps.txt | ForEach-Object { $_ -as [double] }) | Where-Object { $_ -gt 0 } | Measure-Object -Sum | Select-Object -ExpandProperty Sum"

python printdeal.py %boards% eamon\html