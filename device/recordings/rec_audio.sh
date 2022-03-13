filename=$1
arecord -f S16_LE -r 44100 -D hw:1,0 -d 20 $filename
