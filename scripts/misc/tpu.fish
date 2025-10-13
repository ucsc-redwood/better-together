for i in (seq 10)
    xmake r test-nnapi-alexnet --device 3A021JEHN02756 | rg "Time taken:" | sed 's/.*Time taken: \([0-9.]*\) ms/\1/'
end
