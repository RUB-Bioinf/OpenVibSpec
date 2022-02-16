# scipt for downloading and unzip models
# 


wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1SAIq-kUPtL9Yhas1emtLd1CWsGIGCGNu' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1SAIq-kUPtL9Yhas1emtLd1CWsGIGCGNu" -O generator_model.zip && rm -rf /tmp/cookies.txt

unzip generator_model.zip

rm generator_model.zip