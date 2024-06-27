set BEN_HOME=D:\github\ben\
set model1=%1
set model2=%2
set boards=%3
set db=%4

python auction.py --bidderNS=%model1%.conf --bidderEW=%model2%.conf --set=%boards% --db=%db% > .\%model1%\auctionsNS.json
python auction.py --bidderNS=%model2%.conf --bidderEW=%model1%.conf --set=%boards% --db=%db% > .\%model1%\auctionsEW.json

type ".\%model1%\auctionsNS.json" | python lead.py --bidder=%model2%.conf > .\%model1%\leads1.json
type ".\%model1%\auctionsEW.json" | python lead.py --bidder=%model1%.conf > .\%model1%\leads2.json

type ".\%model1%\leads1.json" | python score.py > .\%model1%\results1.json  
type ".\%model1%\leads2.json" | python score.py > .\%model1%\results2.json  

python ..\..\..\src\compare.py .\%model1%\results1.json .\%model1%\results2.json > .\%model1%\compare.json 

copy ..\..\..\demo\viz.css .\%model1%\html\viz.css

type ".\%model1%\compare.json" | python printmatch.py >.\%model1%\match.txt

type ".\%model1%\compare.json" | python printmatchashtml.py >.\%model1%\html\match.html

python printdeal.py %boards% %1\html

powershell -Command "Get-Content .\%model1%\compare.json | jq .imp | Measure-Object -Sum | Select-Object -ExpandProperty Sum"

type ".\%model1%\compare.json" | jq .imp >.\%model1%\imps.txt  

rem find and sum positive scores
powershell -Command "(Get-Content .\%model1%\imps.txt | ForEach-Object { $_ -as [double] }) | Where-Object { $_ -gt 0 } | Measure-Object -Sum | Select-Object -ExpandProperty Sum"

