B=my

python auction.py \
    --bidderNS=A.conf \
    --bidderEW=$B.conf \
    --set=$1 \
    | tee tmp_auctions1.json


python auction.py \
    --bidderNS=$B.conf \
    --bidderEW=A.conf \
    --set=$1 \
    | tee tmp_auctions2.json


cat tmp_auctions1.json | python score.py | tee tmp_results1.json
cat tmp_auctions2.json | python score.py | tee tmp_results2.json

#cat tmp_lead1.json | python score.py > tmp_results1.json
#cat tmp_lead2.json | python score.py > tmp_results2.json

python compare.py tmp_results1.json tmp_results2.json | tee $2
cat compare.json | jq .imp | awk '{s+=$1} END {print s}'

rm tmp_*.json

