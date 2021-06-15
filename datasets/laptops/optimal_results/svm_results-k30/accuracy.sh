num_correct="0"
for f in $(ls | grep 'svm_'); do
	correct=`cat $f | grep 'correct' | awk -F '/' '{ print $2 }' | awk -F ':' '{ print int($2) }'`
	echo $correct
	#num_correct=$num_correct+$correct
	let "num_correct=num_correct+correct"
done
echo $num_correct
