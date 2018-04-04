for file in wav/*; do fname=$(basename "$file" .wav); echo "( " $fname \" \" ")"; done > etc/txt.done.data
