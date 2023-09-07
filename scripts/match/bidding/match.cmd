set trust=%1
set boards=set0010.txt

python auction.py --bidderNS=A.conf --bidderEW=B.conf --set=%boards% --search=EW --nntrust=%trust%  > .\%trust%\auctionsEW.json
python auction.py --bidderNS=B.conf --bidderEW=A.conf --set=%boards% --search=NS --nntrust=%trust%  > .\%trust%\auctionsNS.json

type ".\%trust%\auctionsNS.json" | python lead.py --bidder=A.conf > .\%trust%\leads1.json
type ".\%trust%\auctionsEW.json" | python lead.py --bidder=B.conf > .\%trust%\leads2.json

type ".\%trust%\leads1.json" | python score.py > .\%trust%\results1.json  
type ".\%trust%\leads2.json" | python score.py > .\%trust%\results2.json  

python compare.py .\%trust%\results1.json .\%trust%\results2.json > .\%trust%\compare.json 

type ".\%trust%\compare.json" | python printmatch.py >.\%trust%\match.txt

type ".\%trust%\compare.json" | python printmatchashtml.py >.\%trust%\html\match.html

powershell -Command "Get-Content .\%trust%\compare.json | jq .imp | Measure-Object -Sum | Select-Object -ExpandProperty Sum"

type ".\%trust%\compare.json" | jq .imp >.\%trust%\imps.txt  

rem find and sum positive scores
powershell -Command "(Get-Content .\%trust%\imps.txt | ForEach-Object { $_ -as [double] }) | Where-Object { $_ -gt 0 } | Measure-Object -Sum | Select-Object -ExpandProperty Sum"

python printdeal.py %boards% %1\html