
python auction.py \
    --bidderNS=A.conf \
    --bidderEW=B.conf \
    --set=$1 \
    > tmp_auctions1.json


python auction.py \
    --bidderNS=B.conf \
    --bidderEW=A.conf \
    --set=$1 \
    > tmp_auctions2.json


cat tmp_auctions1.json | python lead.py > tmp_lead1.json
cat tmp_auctions2.json | python lead.py > tmp_lead2.json

cat tmp_lead1.json | python score.py > tmp_results1.json
cat tmp_lead2.json | python score.py > tmp_results2.json

python ..\..\..\src\compare.py tmp_results1.json tmp_results2.json > $2

rm tmp_*.json
