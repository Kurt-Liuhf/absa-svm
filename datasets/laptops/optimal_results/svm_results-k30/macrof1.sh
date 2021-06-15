support="0"
totalfp="0"
totalp="0"
for f in $(ls | grep 'svm_'); do
    # for 0 and 1
    #total=`cat $f | grep '^ \{1,\}[0]' | awk '{ print $5 }'`
    #precision=`cat $f | grep '^ \{1,\}[0]' | awk '{ print $2 }'`
    #recall=`cat $f | grep '^ \{1,\}[0]' | awk '{ print $3 }'`

    # for -1
    total=`cat $f | grep -ne '\-1' | awk -F ':' '{ print $2 }' | awk '{ print $5 }'`
    precision=`cat $f | grep -ne '\-1' | awk -F ':' '{ print $2 }' | awk '{ print $2 }'`
    recall=`cat $f | grep -ne '\-1' | awk -F ':' '{ print $2 }' | awk '{ print $3 }'`
    fp=$(echo "scale=0;($total * $recall + 0.5) / 1" | bc)
    temp=$(echo "scale=1;($fp / $precision + 0.5) / 1" | bc)
    p=$(echo "scale=0;$temp / 1" | bc)
    echo $total
    echo $precision
    echo $recall
    echo $fp
    echo $p
    let "support=support+total"
    let "totalfp=totalfp+fp"
    let "totalp=totalp+p"
done
echo $support
echo $totalfp
echo $totalp

