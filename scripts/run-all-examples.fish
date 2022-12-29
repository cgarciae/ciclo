poetry shell

for f in (ls examples/ | grep '\.py')
    echo -e "\n---------------------------------"
    echo examples/$f
    echo ---------------------------------
    python examples/$f
end
