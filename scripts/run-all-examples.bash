set -e
poetry shell

for f in $(ls examples/ | grep '\.py'); do
    echo -e "\n---------------------------------"
    echo "examples/$f"
    echo "---------------------------------"
    python "examples/$f"
done
